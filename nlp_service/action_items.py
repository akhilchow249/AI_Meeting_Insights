"""
Action item and decision detection for meeting transcripts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


ACTION_MODEL_DIR = Path(__file__).parent / "models" / "action_item" / "best"
DECISION_MODEL_DIR = Path(__file__).parent / "models" / "decision" / "best"

_action_clf = None
_decision_clf = None
_llm_unavailable = False


def _get_action_classifier():
    global _action_clf
    if _action_clf is not None:
        return _action_clf
    if not ACTION_MODEL_DIR.exists():
        logger.warning("Action item model not found at %s. Using regex only.", ACTION_MODEL_DIR)
        return None
    try:
        from transformers import pipeline

        _action_clf = pipeline(
            "text-classification",
            model=str(ACTION_MODEL_DIR),
            tokenizer=str(ACTION_MODEL_DIR),
            device=0 if _cuda_available() else -1,
            truncation=True,
            max_length=256,
        )
        return _action_clf
    except Exception as exc:
        logger.warning("Failed to load action classifier: %s", exc)
        return None


def _get_decision_classifier():
    global _decision_clf
    if _decision_clf is not None:
        return _decision_clf
    if not DECISION_MODEL_DIR.exists():
        logger.warning("Decision model not found at %s. Using regex only.", DECISION_MODEL_DIR)
        return None
    try:
        from transformers import pipeline

        _decision_clf = pipeline(
            "text-classification",
            model=str(DECISION_MODEL_DIR),
            tokenizer=str(DECISION_MODEL_DIR),
            device=0 if _cuda_available() else -1,
            truncation=True,
            max_length=256,
        )
        return _decision_clf
    except Exception as exc:
        logger.warning("Failed to load decision classifier: %s", exc)
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


ACTION_PATTERNS = [
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwill you\b",
    r"\bcan i ask that you\b",
    r"\bi ask that you\b",
    r"\bplease\b.{0,40}\b(do|fix|check|update|send|review|handle|complete|finish|add|remove|create|write|deploy|test)\b",
    r"\baction\s*item\b",
    r"\btake ownership\b",
    r"\bassigned to\b",
    r"\bi('ll| will| am going to)\b.{0,80}\b(research|investigate|review|check|share|send|email|prepare|schedule|follow up|book|arrange|confirm|provide|draft|present|update|fix|remove|add|create|test|deploy|meet|discuss|inform|notify|tell|call|speak|talk|allocate|park|ask|put|coordinate)\b",
    r"\bwe('ll| will| are going to)\b.{0,80}\b(research|investigate|review|check|share|send|email|prepare|schedule|follow up|book|arrange|confirm|provide|draft|present|update|fix|remove|add|create|test|deploy|meet|discuss|inform|notify|tell|call|speak|talk|allocate|park|ask|put|coordinate)\b",
    r"\blook at (researching|other organisations|the costs|the proposal|this proposal)\b",
    r"\b(research|investigate)\b.{0,60}\b(further|proposal|cost|insurance|option|organisation|issue|practice)\b",
    r"\bpresent (us )?with\b",
    r"\bgive a joint presentation\b",
    r"\bset a date\b",
    r"\bmeet up before\b",
    r"\bget it in the diary\b",
    r"\bsomeone (needs|should|must|has to)\b",
    r"\bmake sure (someone|we|you|they)\b",
    r"\bneed(s)? to\b.{0,40}\b(by|before|until|end of)\b",
    r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|end of (day|week|month|sprint))\b",
    r"\bbefore (next|the|this)\b.{0,30}\b(sprint|meeting|week|deadline|release)\b",
    r"\bat the next meeting\b",
    r"\bin a fortnight('?s)? time\b",
    r"\bdue (date|by|on)\b",
    r"\btodo\b",
    r"\bfollow.?up\b",
    r"\b([A-Z][a-z]+|speaker[_\s-]?\d+|we|i|you|they)\s+(will|should|need(?:s)? to|must|has to|have to)\b.{0,100}\b(research|investigate|review|check|share|send|email|prepare|schedule|follow up|book|arrange|confirm|provide|draft|present|update|fix|remove|add|create|test|deploy|meet|discuss|inform|notify|tell|call|speak|talk|allocate|park|ask|put|coordinate)\b",
    r"\blet'?s\b.{0,80}\b(schedule|follow up|meet|review|check|draft|prepare|share|send|email|research|discuss|present|come up with|put|coordinate)\b",
]

DECISION_PATTERNS = [
    r"\bwe have decided\b",
    r"\bwe('ve| have) agreed\b",
    r"\bthe team agrees?\b",
    r"\bgoing forward\b",
    r"\bwe will (adopt|use|move|migrate|switch|implement|roll out|deprecate|drop|replace)\b",
    r"\bdecision (is|was|has been)\b",
    r"\bwe are (moving|switching|adopting|dropping|implementing)\b",
    r"\bfinal(ly|,)? (decided|agreed|resolved|settled)\b",
    r"\bthe plan is\b",
    r"\bwe've settled on\b",
    r"\bconsensus (is|was)\b",
    r"\bapproved\b",
    r"\bsigned off\b",
]

_ACTION_RE = re.compile("|".join(ACTION_PATTERNS), re.IGNORECASE)
_DECISION_RE = re.compile("|".join(DECISION_PATTERNS), re.IGNORECASE)
_NAME_RE = re.compile(r"\b([A-Z][a-z]+)\b")
_INTRO_RE = re.compile(r"^\s*(?:(?:hi|hello|hey)\s*,?\s*)?i(?:'m| am)\s+([A-Z][a-z]+)\b", re.IGNORECASE)
_ACTION_VERB_RE = re.compile(
    r"\b(research|investigate|review|check|share|send|email|prepare|schedule|follow up|book|arrange|confirm|provide|draft|present|update|fix|remove|add|create|test|deploy|meet|discuss|inform|notify|tell|call|speak|talk|allocate|park|ask|put|coordinate|document|submit|vote|come up with)\b",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"\b("
    r"today(?!'s)|tomorrow(?!'s)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"next week|next meeting|next sprint|next release|"
    r"next (?:one|two|three|four|five|\d+) days?|next (?:one|two|three|four|five|\d+) weeks?|"
    r"end of day|end of week|end of month|end of sprint|"
    r"before the next meeting|at the next meeting|"
    r"in a fortnight('?s)? time"
    r")\b",
    re.IGNORECASE,
)

_WEAK_ACTION_PHRASES = [
    "give us a little bit of a background",
    "give us a little background",
    "what do you think",
    "how do you feel",
    "ask the others what they think",
    "can you answer for sure",
    "i'm matthew",
    "i'm john",
    "i'm alex",
    "i'm rachel",
    "it's something we should continue to explore",
    "something we should continue to explore",
    "something we should look at",
    "thanks for coming",
    "apologies for absence",
    "i'll start",
    "new team member",
    "introduce ourselves",
    "carry on",
    "next item on the agenda",
]

_STRONG_ACTION_HINTS = [
    "research",
    "present",
    "follow up",
    "set a date",
    "get it in the diary",
    "give a joint presentation",
    "meet up before",
    "schedule",
    "share",
    "send",
    "prepare",
    "review",
    "check",
    "investigate",
    "confirm",
    "arrange",
    "email",
    "notify",
    "inform",
    "coordinate",
    "put a sign",
    "come up with",
    "park by",
]

_DISCOURSE_RE = re.compile(
    r"^\s*(?:and|so|okay|ok|well|right|yeah|oh|listen|alright|now)\b[:,]?\s*",
    re.IGNORECASE,
)
_NON_ACTION_RE = re.compile(
    "|".join(
        [
            r"\bthanks for coming\b",
            r"\bapologies? for absence\b",
            r"\brunning a bit late\b",
            r"\bemailed apologies\b",
            r"\bcan't (?:make it|be here)\b",
            r"\boff sick\b",
            r"\bnew team member\b",
            r"\bintroduce ourselves\b",
            r"\bon to the next item\b",
            r"\bnext item on the agenda\b",
            r"\bitem on the agenda\b",
            r"\bget back to the agenda\b",
            r"\bplease,\s*carry on\b",
            r"\bput your phones away\b",
            r"^\s*(?:hi|hello)\b.*\bi(?:'m| am)\b",
            r"^\s*i(?:'m| am)\s+[A-Z][a-z]+\b",
            r"^\s*i(?:'ll| will)\s+start\b",
            r"^\s*where(?:'s| is)\b",
        ]
    ),
    re.IGNORECASE,
)
_OWNER_STOPWORDS = {
    "and",
    "as",
    "dammit",
    "delighted",
    "excited",
    "hello",
    "hi",
    "if",
    "it",
    "let",
    "listen",
    "look",
    "now",
    "oh",
    "okay",
    "please",
    "pleased",
    "respect",
    "right",
    "some",
    "someone",
    "sorry",
    "thanks",
    "thank",
    "that",
    "the",
    "there",
    "to",
    "well",
    "yeah",
    "yes",
}
_COMMITMENT_RE = re.compile(
    r"^(?:[^,]{0,50},\s*)?"
    r"(?:(?:can|could|would|will)\s+you\b|please\b|let'?s\b|"
    r"(?:i|we)(?:'ll| will)\b|"
    r"(?:someone|you|they|he|she|[A-Z][a-z]+)\s+(?:need(?:s)? to|must|should|have to|has to|will)\b|"
    r"we\s+(?:need(?:s)? to|must|should|have to|has to)\b)",
    re.IGNORECASE,
)
_DIRECTED_QUESTION_RE = re.compile(
    r"^(?:[A-Z][a-z]+,\s*)?(?:can|could|would|will)\s+you\b",
    re.IGNORECASE,
)
_IMPERATIVE_RE = re.compile(
    r"^(?:come up with|email|send|share|prepare|schedule|arrange|confirm|review|check|draft|present|update|fix|add|remove|create|test|deploy|meet|discuss|call|talk|speak|inform|notify|tell|put|ask|allocate|park|book|follow up|make sure|coordinate)\b",
    re.IGNORECASE,
)
_GROUP_PROCESS_RE = re.compile(
    r"\b(?:can we|shall we|let'?s)\s+(?:all\s+)?(?:agree|decide|introduce|start|make)\b",
    re.IGNORECASE,
)
_CONDITIONAL_VAGUE_ACTION_RE = re.compile(
    r"^\s*if\b.*\b(?:i|we)(?:'ll| will)\s+(?:send|do|handle|take care of)\s+(?:it|that)\b",
    re.IGNORECASE,
)


def _strip_leading_discourse(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = _DISCOURSE_RE.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _is_owner_stopword(token: str | None) -> bool:
    if not token:
        return True
    return token.strip().lower() in _OWNER_STOPWORDS


_ACTION_PROMPT_TEMPLATE = """You are an assistant that extracts structured action items from meeting transcripts.

Given this sentence from a meeting, extract the action item if present.
If this is NOT a real actionable task, respond with {{"is_action": false}}.

Sentence: "{sentence}"
Speaker: {speaker}
Context (surrounding text): "{context}"

Respond ONLY with valid JSON in this exact format:
{{
  "is_action": true,
  "action": "concise description of the task",
  "owner": "speaker label or person name if mentioned",
  "deadline": "deadline if mentioned, else null",
  "confidence": 0.0 to 1.0
}}
or
{{"is_action": false}}"""

_DECISION_PROMPT_TEMPLATE = """You are an assistant that extracts decisions from meeting transcripts.

Given this sentence, extract the decision if one was made.
If this is NOT a real decision, respond with {{"is_decision": false}}.

Sentence: "{sentence}"
Speaker: {speaker}

Respond ONLY with valid JSON:
{{
  "is_decision": true,
  "decision": "concise description of the decision",
  "confidence": 0.0 to 1.0
}}
or
{{"is_decision": false}}"""


def _llm_call(prompt: str, ollama_url: str, model: str, timeout: int) -> str | None:
    global _llm_unavailable
    if _llm_unavailable:
        return None

    models_to_try = [model]
    if model != "llama3":
        models_to_try.append("llama3")

    for candidate in models_to_try:
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": candidate, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404 and candidate != models_to_try[-1]:
                logger.warning("LLM model '%s' not found, retrying with fallback model.", candidate)
                continue
            logger.warning("LLM call failed: %s", exc)
            return None
        except Exception as exc:
            _llm_unavailable = True
            logger.warning("LLM call failed: %s", exc)
            return None

    return None


def _parse_llm_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            clean = json_match.group(0)
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


class ActionItemExtractor:
    def __init__(
        self,
        ollama_url: str = "http://ollama:11434",
        model: str = "llama3.2:1b",
        timeout: int = 120,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout

    def extract(self, segments: list[dict]) -> dict[str, list[dict]]:
        action_items = self.extract_action_items(segments)
        decisions = self.extract_decisions(segments)

        logger.info("Extracted %d action items, %d decisions.", len(action_items), len(decisions))
        return {"action_items": action_items, "decisions": decisions}

    def extract_action_items(self, segments: list[dict]) -> list[dict]:
        action_items: list[dict] = []
        speaker_names = self._build_speaker_name_map(segments)

        for segment in segments:
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "UNKNOWN")
            timestamp = segment.get("start", 0.0)

            if not text:
                continue

            sentences = self._split_sentences(text)
            for i, sentence in enumerate(sentences):
                context = " ".join(sentences[max(0, i - 1) : i + 2])
                regex_hit = bool(_ACTION_RE.search(sentence))
                classifier_support = False
                classifier_score = 0.0
                clf = _get_action_classifier()
                if clf is not None:
                    pred = clf(sentence[:256])[0]
                    is_positive = pred["label"] in ("LABEL_1", "action_item", "1")
                    classifier_score = float(pred.get("score", 0.0))
                    classifier_support = is_positive and classifier_score >= 0.58
                heuristic = self._heuristic_action(sentence, speaker, context, speaker_names)
                heuristic_support = bool(heuristic.get("is_action"))
                if not heuristic_support and not (regex_hit and classifier_support):
                    continue
                if clf is not None and not classifier_support and not heuristic_support:
                    continue

                strong_heuristic = float(heuristic.get("confidence", 0.0)) >= 0.86
                strong_signal = (
                    strong_heuristic
                    or (
                        heuristic_support
                        and regex_hit
                        and float(heuristic.get("confidence", 0.0)) >= 0.8
                    )
                    or (
                        heuristic_support
                        and classifier_support
                        and classifier_score >= 0.88
                    )
                )

                result = heuristic if strong_signal else self._verify_action(sentence, speaker, context, heuristic)
                if result.get("is_action"):
                    action_items.append(
                        {
                            "action": result.get("action", heuristic.get("action", sentence[:120])),
                            "owner": result.get("owner", heuristic.get("owner", speaker)),
                            "owner_name": result.get("owner_name", heuristic.get("owner_name")),
                            "deadline": result.get("deadline", heuristic.get("deadline")),
                            "confidence": round(
                                float(result.get("confidence", heuristic.get("confidence", 0.6))), 4
                            ),
                            "quote": sentence.strip(),
                            "speaker": speaker,
                            "timestamp": round(timestamp, 2),
                        }
                    )

        return self._dedupe_action_items(action_items)

    def extract_decisions(self, segments: list[dict]) -> list[dict]:
        decisions: list[dict] = []

        for segment in segments:
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "UNKNOWN")
            timestamp = segment.get("start", 0.0)

            if not text:
                continue

            for sentence in self._split_sentences(text):
                regex_hit = bool(_DECISION_RE.search(sentence))
                classifier_support = False
                classifier_score = 0.0
                clf = _get_decision_classifier()
                if clf is not None:
                    pred = clf(sentence[:256])[0]
                    is_positive = pred["label"] in ("LABEL_1", "decision", "1")
                    classifier_score = float(pred.get("score", 0.0))
                    score = classifier_score if is_positive else 1.0 - classifier_score
                    classifier_support = is_positive and classifier_score >= 0.7
                    if score < 0.70:
                        continue
                elif not regex_hit:
                    continue

                if regex_hit or (classifier_support and classifier_score >= 0.88):
                    result = {
                        "is_decision": True,
                        "decision": self._normalize_decision(sentence),
                        "confidence": round(max(classifier_score, 0.72), 4),
                    }
                else:
                    result = self._verify_decision(sentence, speaker)
                if result.get("is_decision"):
                    decisions.append(
                        {
                            "decision": result.get("decision", sentence[:120]),
                            "speaker": speaker,
                            "timestamp": round(timestamp, 2),
                            "confidence": round(float(result.get("confidence", 0.6)), 4),
                            "quote": sentence.strip(),
                        }
                    )

        return self._dedupe_decisions(decisions)

    def _verify_action(
        self,
        sentence: str,
        speaker: str,
        context: str,
        heuristic: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = _ACTION_PROMPT_TEMPLATE.format(sentence=sentence, speaker=speaker, context=context)
        raw = _llm_call(prompt, self.ollama_url, self.model, self.timeout)
        result = _parse_llm_json(raw)
        if not result:
            return heuristic
        if result.get("is_action") is False:
            if heuristic.get("is_action") and float(heuristic.get("confidence", 0.0)) >= 0.82:
                return heuristic
            return {"is_action": False}
        if not result.get("owner"):
            result["owner"] = heuristic.get("owner", speaker)
        if not result.get("owner_name"):
            result["owner_name"] = heuristic.get("owner_name")
        if not result.get("deadline"):
            result["deadline"] = heuristic.get("deadline")
        if not result.get("action"):
            result["action"] = heuristic.get("action", sentence[:120])
        if not result.get("confidence"):
            result["confidence"] = heuristic.get("confidence", 0.6)
        return result

    def _verify_decision(self, sentence: str, speaker: str) -> dict[str, Any]:
        prompt = _DECISION_PROMPT_TEMPLATE.format(sentence=sentence, speaker=speaker)
        raw = _llm_call(prompt, self.ollama_url, self.model, self.timeout)
        result = _parse_llm_json(raw)
        if not result:
            return {"is_decision": True, "decision": sentence[:120], "confidence": 0.55}
        return result

    def _heuristic_action(
        self,
        sentence: str,
        speaker: str,
        context: str,
        speaker_names: dict[str, str],
    ) -> dict[str, Any]:
        analysis_sentence = _strip_leading_discourse(sentence)
        lowered = analysis_sentence.lower()
        if any(phrase in lowered for phrase in _WEAK_ACTION_PHRASES):
            return {"is_action": False}

        if _NON_ACTION_RE.search(analysis_sentence):
            return {"is_action": False}

        if re.match(r"^\s*i(?:'m| am)\s+[A-Z][a-z]+", analysis_sentence, re.IGNORECASE):
            return {"is_action": False}

        owner, owner_name = self._extract_owner(analysis_sentence, context, speaker, speaker_names)
        deadline = self._extract_deadline(analysis_sentence, context)
        action = self._normalize_action(analysis_sentence)
        has_task_verb = bool(_ACTION_VERB_RE.search(analysis_sentence)) or bool(
            re.search(r"\blet\b.{0,20}\bknow\b", analysis_sentence, re.IGNORECASE)
        )
        commitment = bool(_COMMITMENT_RE.search(analysis_sentence))
        directed_question = bool(_DIRECTED_QUESTION_RE.search(analysis_sentence))
        imperative = bool(_IMPERATIVE_RE.search(analysis_sentence))
        strong = any(hint in lowered for hint in _STRONG_ACTION_HINTS) or (commitment and has_task_verb)

        if _GROUP_PROCESS_RE.search(analysis_sentence) and deadline is None:
            return {"is_action": False}

        if not has_task_verb:
            return {"is_action": False}

        if not (commitment or imperative or directed_question or deadline):
            return {"is_action": False}

        if analysis_sentence.endswith("?") and not directed_question and deadline is None:
            return {"is_action": False}

        if re.search(r"\b(background|proposal)\b", lowered) and not strong and deadline is None:
            return {"is_action": False}

        if "should continue to explore" in lowered and deadline is None and owner == speaker:
            return {"is_action": False}

        if _CONDITIONAL_VAGUE_ACTION_RE.search(analysis_sentence) and deadline is None:
            return {"is_action": False}

        if len(re.findall(r"[A-Za-z']+", analysis_sentence)) <= 3 and deadline is None:
            return {"is_action": False}

        confidence = 0.72
        if deadline:
            confidence += 0.08
        if owner_name or owner != speaker:
            confidence += 0.08
        if strong:
            confidence += 0.07

        return {
            "is_action": True,
            "action": action,
            "owner": owner,
            "owner_name": owner_name,
            "deadline": deadline,
            "confidence": min(confidence, 0.95),
        }

    @staticmethod
    def _build_speaker_name_map(segments: list[dict]) -> dict[str, str]:
        speaker_names: dict[str, str] = {}
        for segment in segments:
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "UNKNOWN")
            if not text or speaker in speaker_names or speaker == "UNKNOWN":
                continue
            match = _INTRO_RE.search(_strip_leading_discourse(text))
            if match:
                candidate = match.group(1).title()
                if not _is_owner_stopword(candidate):
                    speaker_names[speaker] = candidate
        return speaker_names

    @staticmethod
    def _extract_owner(
        sentence: str,
        context: str,
        speaker: str,
        speaker_names: dict[str, str],
    ) -> tuple[str, str | None]:
        known_names = {name: label for label, name in speaker_names.items()}
        analysis_sentence = _strip_leading_discourse(sentence)

        if re.match(r"^\s*let'?s\b", analysis_sentence, re.IGNORECASE):
            return "GROUP", None

        direct_address = re.match(r"^\s*([A-Z][a-z]+),", analysis_sentence)
        if direct_address:
            name = direct_address.group(1)
            if not _is_owner_stopword(name):
                return known_names.get(name, name), name

        trailing_name = re.search(r",\s*([A-Z][a-z]+)[?.!]?$", analysis_sentence)
        if trailing_name:
            name = trailing_name.group(1)
            if not _is_owner_stopword(name):
                return known_names.get(name, name), name

        asked_person = re.search(r"\b([A-Z][a-z]+)\s+can i ask that you\b", analysis_sentence, re.IGNORECASE)
        if asked_person:
            name = asked_person.group(1)
            if not _is_owner_stopword(name):
                return known_names.get(name, name), name

        assigned_owner = re.match(
            r"^\s*([A-Z][a-z]+)\s+(?:will|need(?:s)? to|must|should|has to|have to)\b",
            analysis_sentence,
            re.IGNORECASE,
        )
        if assigned_owner:
            name = assigned_owner.group(1)
            if name.lower() == "we":
                return "GROUP", None
            if not _is_owner_stopword(name):
                return known_names.get(name, name), name

        for text in (analysis_sentence, context):
            for name in _NAME_RE.findall(text):
                if name in known_names and not _is_owner_stopword(name):
                    return known_names[name], name

        if re.search(
            r"\bwe('ll| will| need(?:s)? to| must| should| have to| has to)\b",
            analysis_sentence,
            re.IGNORECASE,
        ):
            return "GROUP", None

        return speaker, speaker_names.get(speaker)

    @staticmethod
    def _extract_deadline(sentence: str, context: str) -> str | None:
        combined = f"{sentence} {context}"
        match = _DEADLINE_RE.search(combined)
        if match:
            trailing = combined[match.end() : match.end() + 2]
            if trailing.startswith("'s"):
                return None
            return match.group(1)
        return None

    @staticmethod
    def _normalize_action(sentence: str) -> str:
        cleaned = " ".join(_strip_leading_discourse(sentence).strip().split())
        lowered = cleaned.lower()

        if "look at researching this proposal further" in lowered:
            return "Research the proposal further"
        if "look at other organisations that have taken this practice on board" in lowered:
            return "Research how other organisations have implemented this practice"
        if "present us with that information at the next meeting" in lowered:
            return "Present the findings at the next meeting"
        if "give a joint presentation" in lowered:
            return "Give a joint presentation at the next meeting"
        if "set a date to meet up before the next meeting" in lowered:
            return "Set a date to meet before the next meeting"
        if "get it in the diary today" in lowered:
            return "Schedule the follow-up meeting today"

        cleaned = re.sub(r"^can i ask that you\b[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^let'?s\s+all\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^let'?s\b[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^(?:[A-Z][a-z]+,\s*)?(?:can|could|would|will)\s+you\b[:,]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:[A-Z][a-z]+,\s*)?you\s+(?:need(?:s)? to|must|should|have to|has to|will)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:[A-Z][a-z]+|Speaker\s*\d+)\s+(?:will|should|need(?:s)? to|must|has to)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:we|i)(?:'ll| will| should| need(?:s)? to| must| have to| has to)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^please\b[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r",\s*[A-Z][a-z]+$", "", cleaned)
        return cleaned.rstrip(".?!")

    @staticmethod
    def _normalize_decision(sentence: str) -> str:
        cleaned = " ".join(sentence.strip().split())
        cleaned = re.sub(r"^(well|okay|so|moving forward)\b[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^(?:we have decided|we've decided|we have agreed|we've agreed|the decision is|the plan is|we've settled on|consensus is)\b[:,]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.rstrip(".")

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if len(p.strip()) >= 6]

    @staticmethod
    def _dedupe_action_items(action_items: list[dict]) -> list[dict]:
        seen: set[tuple[str, str, str | None]] = set()
        deduped: list[dict] = []
        for item in action_items:
            key = (
                item.get("action", "").lower(),
                item.get("owner", ""),
                item.get("deadline"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _dedupe_decisions(decisions: list[dict]) -> list[dict]:
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in decisions:
            key = item.get("decision", "").lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
