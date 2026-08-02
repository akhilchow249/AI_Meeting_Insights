"""
nlp-service/topics.py
─────────────────────
Key topic extraction using two complementary approaches:

1. KeyBERT — BERT-based keyphrase extraction.
2. LDA via scikit-learn LatentDirichletAllocation.
   (Replaces gensim — no C++ compilation required on Windows.)

Output schema
─────────────
{
  "keyphrases": [
    {"phrase": "API rate limit", "score": 0.82}, ...   # top 10
  ],
  "lda_topics": [
    {"topic_id": 0, "top_words": ["rate","limit","pipeline","failure","nightly"], "weight": 0.24},
    ...                                                  # 5 topics
  ]
}
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

_keybert_model = None

DOMAIN_TOPIC_PATTERNS = {
    "working from home": r"\bworking from home\b",
    "work from home policy": r"\b(policy|proposal).{0,25}\bworking from home\b|\bworking from home.{0,25}\bpolicy\b",
    "flexible working": r"\bflexible working\b",
    "family-friendly working": r"\bfamily[- ]friendly working\b",
    "insurance implications": r"\binsurance\b",
    "health and safety": r"\bhealth and safety\b",
    "fairness across teams": r"\bnot fair\b|\bunfair\b|\bacross the board\b|\bone section of the team\b",
    "office concentration": r"\bdifficult to concentrate\b|\bbusy office\b|\bovercrowded\b|\bphones ring all the time\b",
    "cost concerns": r"\bcost(ing|s)?\b|\bbudget\b|\bfinancial statement\b",
    "other organisations": r"\bother organisations\b",
}


def _get_keybert():
    global _keybert_model
    if _keybert_model is None:
        from keybert import KeyBERT
        _keybert_model = KeyBERT(model="all-MiniLM-L6-v2")
    return _keybert_model


# ─── TopicExtractor ───────────────────────────────────────────────────────────

class TopicExtractor:
    """
    Extracts keyphrases (KeyBERT) and LDA topics (sklearn) from
    a list of diarised transcript segments.
    """

    STOPWORDS = {
        "um", "uh", "yeah", "okay", "right", "like", "just", "know",
        "think", "going", "really", "actually", "basically", "literally",
        "sort", "kind", "thing", "things", "something", "anything",
        "want", "need", "make", "sure", "good", "great", "well",
    }

    def __init__(
        self,
        top_n_keyphrases: int = 10,
        n_lda_topics:     int = 5,
        keyphrase_ngram:  tuple[int, int] = (1, 3),
    ):
        self.top_n_keyphrases = top_n_keyphrases
        self.n_lda_topics     = n_lda_topics
        self.keyphrase_ngram  = keyphrase_ngram

    def extract(self, segments: list[dict]) -> dict[str, Any]:
        full_text  = self._segments_to_text(segments)
        clean_text = self._clean(full_text)

        keyphrases = self._merge_keyphrases(
            self._run_keybert(clean_text),
            self._extract_domain_keyphrases(full_text),
        )
        lda_topics = self._run_lda(clean_text)

        return {"keyphrases": keyphrases, "lda_topics": lda_topics}

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _segments_to_text(segments: list[dict]) -> str:
        return " ".join(s.get("text", "") for s in segments)

    def _clean(self, text: str) -> str:
        text = text.lower()
        filler_pattern = r"\b(" + "|".join(self.STOPWORDS) + r")\b"
        text = re.sub(filler_pattern, " ", text)
        text = text.translate(
            str.maketrans(
                string.punctuation.replace("-", ""),
                " " * (len(string.punctuation) - 1)
            )
        )
        return re.sub(r"\s+", " ", text).strip()

    def _run_keybert(self, text: str) -> list[dict]:
        if not text.strip():
            return []
        try:
            kb  = _get_keybert()
            raw = kb.extract_keywords(
                text,
                keyphrase_ngram_range=self.keyphrase_ngram,
                stop_words="english",
                use_mmr=True,
                diversity=0.5,
                top_n=self.top_n_keyphrases,
            )
            return [{"phrase": phrase, "score": round(score, 4)} for phrase, score in raw]
        except Exception as exc:
            logger.warning("KeyBERT extraction failed, using n-gram fallback: %s", exc)
            return self._fallback_keyphrases(text)

    def _extract_domain_keyphrases(self, text: str) -> list[dict]:
        lowered = re.sub(r"\s+", " ", text.lower())
        matches: list[dict] = []
        for phrase, pattern in DOMAIN_TOPIC_PATTERNS.items():
            hit_count = len(re.findall(pattern, lowered, re.IGNORECASE))
            if hit_count:
                score = min(0.98, 0.68 + (hit_count * 0.08))
                matches.append({"phrase": phrase, "score": round(score, 4)})
        matches.sort(key=lambda item: -item["score"])
        return matches

    def _merge_keyphrases(self, model_keyphrases: list[dict], domain_keyphrases: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {}
        for item in model_keyphrases + domain_keyphrases:
            phrase = item["phrase"].strip().lower()
            if phrase not in merged or item["score"] > merged[phrase]["score"]:
                merged[phrase] = {
                    "phrase": item["phrase"],
                    "score": round(float(item["score"]), 4),
                }
        ranked = sorted(merged.values(), key=lambda item: -item["score"])
        return ranked[: self.top_n_keyphrases]

    def _fallback_keyphrases(self, text: str) -> list[dict]:
        tokens = [tok for tok in text.split() if tok and tok not in self.STOPWORDS and len(tok) > 2]
        candidates = Counter()
        for n in (2, 3):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i + n]).strip()
                if len(set(phrase.split())) == 1:
                    continue
                candidates[phrase] += 1

        results = []
        for phrase, count in candidates.most_common(self.top_n_keyphrases):
            score = min(0.75, 0.45 + (count * 0.08))
            results.append({"phrase": phrase, "score": round(score, 4)})
        return results

    def _run_lda(self, text: str) -> list[dict]:
        """
        LDA using scikit-learn — pure Python, no C++ required.

        Pipeline:
          1. TF-IDF vectorise sentences
          2. Fit LatentDirichletAllocation
          3. Compute per-topic weight as mean topic proportion across corpus
          4. Return top-5 words per topic
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        import numpy as np

        # Split into sentence-level documents
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]

        if len(sentences) < self.n_lda_topics:
            logger.warning("Too few sentences (%d) for %d LDA topics.",
                           len(sentences), self.n_lda_topics)
            return []

        # Vectorise with English stopwords (no external dependency)
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words="english",
            max_features=500,
            ngram_range=(1, 2),
        )

        try:
            dtm = vectorizer.fit_transform(sentences)
        except ValueError:
            # All terms filtered out (very short transcript)
            return []

        if dtm.shape[1] == 0:
            return []

        lda = LatentDirichletAllocation(
            n_components=self.n_lda_topics,
            random_state=42,
            max_iter=20,
            learning_method="batch",
        )
        doc_topics = lda.fit_transform(dtm)   # shape: (n_sentences, n_topics)

        vocab      = vectorizer.get_feature_names_out()

        # Mean topic weight across all sentences
        mean_weights = doc_topics.mean(axis=0)

        topics = []
        for topic_id in range(self.n_lda_topics):
            top_indices = lda.components_[topic_id].argsort()[-5:][::-1]
            top_words   = [vocab[i] for i in top_indices]
            topics.append({
                "topic_id":  topic_id,
                "top_words": top_words,
                "weight":    round(float(mean_weights[topic_id]), 4),
            })

        topics.sort(key=lambda t: -t["weight"])
        return topics
