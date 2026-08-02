"""
nlp-service/pain_points.py
───────────────────────────
Pain point extraction — two-pass pipeline.

What is a pain point?
─────────────────────
A pain point is a STRUCTURED CONCERN that blocks or slows progress.
It is NOT simply any negative sentence.

Pain point: "We are blocked on the auth service — every deploy breaks
             the token refresh and we've had three incidents this week."

Not a pain point: "I don't think we should use that approach."
                  "This is a bit complicated."

Pass 1 — Fine-tuned BERT binary classifier
  Model: fine-tuned distilbert-base-uncased (or loaded from models/ directory)
  Classifies each sentence as PAIN_POINT (1) or NOT_PAIN_POINT (0).
  Falls back to a heuristic keyword scorer if the model file is absent
  (useful during development before training data is collected).

Pass 2 — LLM structured extraction (Ollama)
  For each confirmed pain point, the LLM fills in the full schema:
  severity, category, quote, and a cleaned description.

Output schema (per pain point)
──────────────────────────────
{
  "pain_point" : "API rate limit causing nightly failures in reporting pipeline",
  "speaker"    : "SPEAKER_01",
  "timestamp"  : 143.20,
  "severity"   : "high",          # high | medium | low
  "category"   : "technical_blocker",
  "quote"      : "we have been hitting it every evening...",
  "confidence" : 0.87
}

Severity
────────
  high   — blocking progress (cannot proceed without resolution)
  medium — slowing progress (significant friction but workaround exists)
  low    — noted concern (minor friction, FYI)

Categories
──────────
  technical_blocker | resource_constraint | process_inefficiency |
  external_dependency | unclear_requirements | team_communication | other
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

SEVERITY_VALUES   = {"high", "medium", "low"}
CATEGORY_VALUES   = {
    "technical_blocker", "resource_constraint", "process_inefficiency",
    "external_dependency", "unclear_requirements", "team_communication", "other",
}

MODEL_DIRS = [
    Path(__file__).parent / "models" / "pain_point" / "best",
    Path(__file__).parent / "models" / "pain_point_classifier",
]

# ─── Heuristic fallback (used when fine-tuned model is not yet trained) ───────

# High-signal pain point keywords — presence strongly suggests a pain point
HIGH_SIGNAL_KEYWORDS = [
    r"\bblocked?\b", r"\bbreaking\b", r"\bbroken\b", r"\bfailing\b",
    r"\bfailure\b", r"\bincident\b", r"\boutage\b", r"\bcrash(ing|es)?\b",
    r"\bcannot (proceed|continue|deploy|release|ship)\b",
    r"\bcan'?t (proceed|continue|deploy|release|ship)\b",
    r"\bno(t)? able to\b", r"\bstuck\b", r"\bblocker\b",
    r"\bdeadlock\b", r"\bbottleneck\b", r"\bregression\b",
    r"\bevery (night|deploy|time|release)\b",
    r"\bkeep(s)? (breaking|failing|crashing)\b",
    r"\brepeat(ed)? (failure|incident|issue)\b",
    r"\bthree (incidents|failures|times)\b",
    r"\brate limit\b", r"\btimeout\b", r"\blatency (issue|problem|spike)\b",
    r"\bmissing (documentation|spec|requirement|owner|resource)\b",
    r"\bno (budget|resource|capacity|time|ownership|clarity)\b",
    r"\bunderstaffed\b", r"\boverwhelmed\b", r"\bburnout\b",
    r"\bunclear (requirement|spec|ownership|priority)\b",
    r"\bconflicting (requirement|priority|message)\b",
    r"\bnobody (knows|owns|is responsible)\b",
    r"\bno owner\b", r"\black of (clarity|direction|ownership)\b",
    r"\binsurance (issue|application|implication|concern|cost)\b",
    r"\bhealth and safety\b",
    r"\bliab(le|ility)\b",
    r"\bunfair\b", r"\bnot fair\b", r"\balienat(ing|e)\b",
    r"\bovercrowded\b", r"\bphones ring all the time\b",
    r"\bdifficult to concentrate\b", r"\bextremely difficult to concentrate\b",
    r"\bbusy office\b", r"\bnoisy office\b",
    r"\bpolicy\b.{0,30}\b(ad hoc|unclear|missing|absent)\b",
    r"\bstaff morale\b", r"\blow morale\b", r"\bsickness absence\b",
    r"\black of training\b", r"\black of effective appraisals?\b",
    r"\bpoor management\b",
    r"\bcar parking\b.{0,30}\b(problem|issue|fight|argument|row|dispute|shortage)\b",
    r"\bparking\b.{0,30}\b(problem|issue|fight|argument|row|dispute|shortage)\b",
    r"\btaken up two parking spaces\b",
]

# Medium-signal — suggests friction but needs context
MEDIUM_SIGNAL_KEYWORDS = [
    r"\bslow(er|ing down)?\b", r"\bdelay(ed|ing)?\b", r"\bbehind schedule\b",
    r"\bmissed (deadline|milestone|sprint)\b",
    r"\bdifficult\b", r"\bchallenging\b", r"\bfrustrat(ed|ing)\b",
    r"\bnot working\b", r"\bissue\b", r"\bproblem\b", r"\bconcern\b",
    r"\bworkaround\b", r"\bhack\b", r"\bmanual process\b",
    r"\btoo many\b", r"\brepetitive\b", r"\btoil\b",
    r"\bcost(ing|s)?\b", r"\bbudget\b", r"\bfinancial\b",
    r"\bprocedural issue(s)?\b", r"\bpractical thing(s)?\b",
    r"\bpolicy\b", r"\bworking from home\b", r"\bflexible working\b",
    r"\bfamily[- ]friendly working\b", r"\bad hoc\b", r"\bguilty\b",
    r"\bfeel differently\b", r"\bother colleagues\b",
    r"\bcar parking\b", r"\bparking\b", r"\bbad reputation\b",
    r"\barguments?\b", r"\bfight\b", r"\browing\b",
]

_HIGH_RE   = re.compile("|".join(HIGH_SIGNAL_KEYWORDS),   re.IGNORECASE)
_MEDIUM_RE = re.compile("|".join(MEDIUM_SIGNAL_KEYWORDS), re.IGNORECASE)

CONTEXTUAL_PATTERNS = {
    "resource_constraint": [
        r"\bcost(ing|s)?\b", r"\bbudget\b", r"\bfinancial\b", r"\bcomputer(s)?\b",
        r"\binsurance\b",
    ],
    "process_inefficiency": [
        r"\bpolicy\b", r"\bprocedure\b", r"\bpractical\b", r"\bad hoc\b",
    ],
    "team_communication": [
        r"\bunfair\b", r"\bnot fair\b", r"\balienat(ing|e)\b",
        r"\bfeel differently\b", r"\bguilty\b",
        r"\bstaff morale\b", r"\blow morale\b", r"\bpoor management\b",
        r"\bbad reputation\b",
    ],
    "other": [
        r"\bworking from home\b", r"\bflexible working\b", r"\bfamily[- ]friendly\b",
        r"\bdifficult to concentrate\b", r"\bovercrowded\b", r"\bphones ring\b",
        r"\bcar parking\b", r"\bparking\b", r"\bsickness absence\b",
        r"\black of training\b", r"\bappraisals?\b",
    ],
}

_DISCOURSE_RE = re.compile(
    r"^\s*(?:and|so|okay|ok|well|right|yeah|oh|now|alright|listen|look)\b[:,]?\s*",
    re.IGNORECASE,
)
_MEETING_ADMIN_RE = re.compile(
    "|".join([
        r"\bthanks for coming\b",
        r"\bapologies? for absence\b",
        r"\bemailed apologies\b",
        r"\brunning a bit late\b",
        r"\boff sick\b",
        r"\bcan't (?:make it|be here)\b",
        r"\bnew team member\b",
        r"\bintroduce ourselves\b",
        r"\bon to the next item\b",
        r"\bnext item on the agenda\b",
        r"\bitem on the agenda\b",
        r"\bget back to the agenda\b",
        r"\bplease,\s*carry on\b",
        r"^\s*(?:hi|hello)\b.*\bi(?:'m| am)\b",
        r"^\s*i(?:'m| am)\s+[A-Z][a-z]+\b",
        r"^\s*i(?:'ll| will)\s+start\b",
        r"^\s*where(?:'s| is)\b",
    ]),
    re.IGNORECASE,
)
_ACTIONISH_RE = re.compile(
    r"^\s*(?:how about|let'?s|i(?:'ll| will)|we(?:'ll| will| need to)|why don't i|please\b|(?:can|could|would|will)\s+you\b|[A-Z][a-z]+,\s*(?:can|could|would|will)\s+you\b)",
    re.IGNORECASE,
)
_META_DISCUSSION_RE = re.compile(
    r"\b(?:what has this got to do with|what does everyone think about that|can we all agree)\b",
    re.IGNORECASE,
)
_SOLUTION_HINT_RE = re.compile(
    r"\b(?:parks?\s+by|put a sign|send an email|send a list|circulate|put it to a vote|come up with|coordinate|allocate|let you know|speak with|ask the staff|give them some options)\b",
    re.IGNORECASE,
)
_PAIN_NOUNS_RE = re.compile(
    r"\b(problem|issue|concern|morale|absence|lack of|restructuring|poor management|bad reputation|argument|fight)\b",
    re.IGNORECASE,
)


def _strip_leading_discourse(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = _DISCOURSE_RE.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _heuristic_score(sentence: str) -> float:
    """Return a confidence score [0, 1] based on keyword density."""
    high_hits   = len(_HIGH_RE.findall(sentence))
    medium_hits = len(_MEDIUM_RE.findall(sentence))
    score = min(1.0, high_hits * 0.35 + medium_hits * 0.12)
    lowered = sentence.lower()
    if "concern" in lowered and any(
        phrase in lowered for phrase in (
            "cost", "insurance", "health and safety", "fair", "policy", "concentrate"
        )
    ):
        score = max(score, 0.34)
    if "working from home" in lowered and any(
        phrase in lowered for phrase in ("policy", "fair", "cost", "insurance", "concentrate")
    ):
        score = max(score, 0.36)
    return round(score, 4)


# ─── Fine-tuned BERT classifier ───────────────────────────────────────────────

_classifier_pipeline = None
_llm_unavailable = False


def _get_classifier():
    """
    Load fine-tuned DistilBERT pain point classifier from models/ directory.
    Falls back to None if model not trained yet.
    """
    global _classifier_pipeline
    if _classifier_pipeline is not None:
        return _classifier_pipeline

    try:
        from transformers import pipeline
        for model_dir in MODEL_DIRS:
            if not model_dir.exists():
                continue
            _classifier_pipeline = pipeline(
                "text-classification",
                model=str(model_dir),
                tokenizer=str(model_dir),
                device=0 if _cuda_available() else -1,
                truncation=True,
                max_length=512,
            )
            logger.info("Pain point classifier loaded from %s.", model_dir)
            return _classifier_pipeline
    except Exception as exc:
        logger.warning("Failed to load classifier: %s. Using heuristic.", exc)
        return None

    logger.warning(
        "Pain point classifier model not found at any expected path (%s). Using heuristic fallback.",
        ", ".join(str(path) for path in MODEL_DIRS),
    )
    return None


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _positive_class_score(classifier, sentence: str) -> float:
    try:
        raw = classifier(sentence[:512], top_k=None)
    except TypeError:
        raw = classifier(sentence[:512])

    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    if isinstance(raw, dict):
        raw = [raw]

    positive_labels = {"LABEL_1", "pain_point", "1"}
    best_score = 0.0
    for row in raw or []:
        label = str(row.get("label", ""))
        score = float(row.get("score", 0.0))
        if label in positive_labels:
            best_score = max(best_score, score)
    return best_score


# ─── LLM structured extraction ────────────────────────────────────────────────

_PAIN_PROMPT = """You are an expert at analysing meeting transcripts for pain points and blockers.

A pain point is a STRUCTURAL CONCERN that blocks or slows team progress — not just any negative sentence.

Given this sentence from a meeting transcript, extract a structured pain point.

Sentence: "{sentence}"
Speaker: {speaker}
Context: "{context}"

If this is NOT a genuine pain point (just an opinion, preference, or minor comment), respond with:
{{"is_pain_point": false}}

If it IS a pain point, respond ONLY with valid JSON:
{{
  "is_pain_point": true,
  "pain_point": "concise description of the blocker or friction (1-2 sentences)",
  "severity": "high" or "medium" or "low",
  "category": one of [technical_blocker, resource_constraint, process_inefficiency, external_dependency, unclear_requirements, team_communication, other],
  "quote": "the most relevant phrase from the sentence (verbatim, max 20 words)",
  "confidence": 0.0 to 1.0
}}

Severity guide:
  high   = blocking progress, team cannot proceed without resolving this
  medium = slowing progress significantly, but a workaround exists
  low    = minor friction, noted for awareness

Respond ONLY with JSON. No explanation."""


def _llm_extract(
    sentence: str,
    speaker: str,
    context: str,
    ollama_url: str,
    model: str,
    timeout: int,
) -> dict:
    global _llm_unavailable
    if _llm_unavailable:
        return {}

    prompt = _PAIN_PROMPT.format(
        sentence=sentence, speaker=speaker, context=context
    )
    models_to_try = [model]
    if model != "llama3.2:1b":
        models_to_try.append("llama3.2:1b")

    for candidate in models_to_try:
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": candidate, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match:
                clean = json_match.group(0)
            return json.loads(clean)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404 and candidate != models_to_try[-1]:
                logger.warning("Pain-point LLM model '%s' not found, retrying with fallback model.", candidate)
                continue
            logger.warning("LLM pain point extraction failed: %s", exc)
            return {}
        except Exception as exc:
            _llm_unavailable = True
            logger.warning("LLM pain point extraction failed: %s", exc)
            return {}

    return {}


def _heuristic_structured_extract(sentence: str, score: float) -> dict[str, Any]:
    category = "other"
    for cat, patterns in CONTEXTUAL_PATTERNS.items():
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
            category = cat
            break

    if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in HIGH_SIGNAL_KEYWORDS):
        severity = "high"
    elif score >= 0.45:
        severity = "medium"
    else:
        severity = "low"

    cleaned = re.sub(r"^(well|okay|so|i mean)\b[:,]?\s*", "", sentence.strip(), flags=re.IGNORECASE)
    return {
        "is_pain_point": True,
        "pain_point": cleaned[:220],
        "severity": severity,
        "category": category,
        "quote": sentence.strip()[:120],
        "confidence": round(max(score, 0.55 if severity != "low" else 0.42), 4),
    }


# ─── PainPointExtractor ───────────────────────────────────────────────────────

class PainPointExtractor:
    """
    Detects and structures pain points from diarised transcript segments.

    Parameters
    ----------
    threshold        : float  Minimum heuristic score OR classifier probability
                              to send a sentence to the LLM pass (default 0.30).
    ollama_url       : str    Ollama API base URL.
    model            : str    Ollama model name (e.g. "mistral", "llama3.2:1b").
    """

    def __init__(
        self,
        threshold:   float = 0.30,
        classifier_threshold: float = 0.62,
        ollama_url:  str   = "http://ollama:11434",
        model:       str   = "llama3.2:1b",
        timeout:     int   = 120,
    ):
        self.threshold  = threshold
        self.classifier_threshold = classifier_threshold
        self.ollama_url = ollama_url
        self.model      = model
        self.timeout    = timeout

    def extract(self, segments: list[dict]) -> list[dict]:
        """
        Run pain point extraction on all transcript segments.

        Returns
        -------
        list[dict]  — pain point records, sorted by timestamp.
        """
        classifier = _get_classifier()
        results:  list[dict] = []
        seen_quotes: set[str] = set()   # deduplicate near-identical pain points

        for segment in segments:
            text      = segment.get("text", "").strip()
            speaker   = segment.get("speaker", "UNKNOWN")
            timestamp = segment.get("start", 0.0)

            if not text:
                continue

            sentences = self._split_sentences(text)

            for i, sentence in enumerate(sentences):
                if len(sentence) < 20:   # too short to be a real pain point
                    continue

                analysis_sentence = _strip_leading_discourse(sentence)
                if _MEETING_ADMIN_RE.search(analysis_sentence):
                    continue
                if _META_DISCUSSION_RE.search(analysis_sentence):
                    continue
                if _ACTIONISH_RE.search(analysis_sentence) and not _PAIN_NOUNS_RE.search(analysis_sentence):
                    continue
                if _SOLUTION_HINT_RE.search(analysis_sentence) and not _PAIN_NOUNS_RE.search(analysis_sentence):
                    continue

                context = " ".join(sentences[max(0, i-1): i+2])

                # ── Pass 1: score the sentence ────────────────────────────
                heuristic_score = _heuristic_score(analysis_sentence)
                classifier_score = 0.0
                if classifier is not None:
                    try:
                        classifier_score = _positive_class_score(classifier, analysis_sentence)
                    except Exception:
                        classifier_score = 0.0

                score = max(classifier_score, heuristic_score)

                if classifier_score < self.classifier_threshold and heuristic_score < self.threshold:
                    continue

                # ── Pass 2: LLM structured extraction ─────────────────────
                high_signal = bool(_HIGH_RE.search(analysis_sentence))
                strong_signal = (
                    heuristic_score >= max(self.threshold + 0.12, 0.46)
                    or (high_signal and score >= 0.4)
                )

                if strong_signal:
                    llm_result = _heuristic_structured_extract(analysis_sentence, score)
                else:
                    llm_result = _llm_extract(
                        analysis_sentence, speaker, context,
                        self.ollama_url, self.model, self.timeout,
                    )

                    if not llm_result:
                        if heuristic_score >= self.threshold:
                            llm_result = _heuristic_structured_extract(analysis_sentence, score)
                        else:
                            continue

                if not llm_result.get("is_pain_point"):
                    # LLM rejected it — trust LLM over classifier/heuristic
                    continue

                # Validate and sanitise LLM output
                severity = llm_result.get("severity", "medium")
                if severity not in SEVERITY_VALUES:
                    severity = "medium"

                category = llm_result.get("category", "other")
                if category not in CATEGORY_VALUES:
                    category = "other"

                quote = llm_result.get("quote", analysis_sentence[:120])

                # Deduplicate by quote fingerprint
                fingerprint = re.sub(r"\W+", "", quote.lower())[:60]
                if fingerprint in seen_quotes:
                    continue
                seen_quotes.add(fingerprint)

                confidence = float(llm_result.get("confidence", score))

                results.append({
                    "pain_point": llm_result.get("pain_point", analysis_sentence[:200]),
                    "speaker":    speaker,
                    "timestamp":  round(timestamp, 2),
                    "severity":   severity,
                    "category":   category,
                    "quote":      quote,
                    "confidence": round(confidence, 4),
                })

        results.sort(key=lambda r: r["timestamp"])
        logger.info("Pain point extraction complete: %d pain points.", len(results))
        return results

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]
