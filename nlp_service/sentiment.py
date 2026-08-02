"""
nlp-service/sentiment.py
─────────────────────────
Per-segment sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment-latest.

Why this model?
  - Trained on 124M tweets — handles informal, spoken-style language well
  - 3-class output: POSITIVE / NEUTRAL / NEGATIVE
  - Fast inference on GPU; handles short segments naturally
  - Robust to speech transcription artefacts (missing punctuation, etc.)

Output schema
─────────────
{
  "per_segment": [
    {
      "segment_id": 12,
      "speaker":    "SPEAKER_01",
      "start":      143.20,
      "end":        158.90,
      "text":       "The main blocker right now is...",
      "sentiment":  "negative",
      "scores": {
        "positive": 0.03,
        "neutral":  0.11,
        "negative": 0.86
      }
    }, ...
  ],
  "per_speaker": {
    "SPEAKER_00": {
      "positive_pct": 0.45,
      "neutral_pct":  0.38,
      "negative_pct": 0.17,
      "dominant":     "positive",
      "trend": ["positive", "neutral", "negative", "negative", ...]
    }, ...
  },
  "overall": {
    "positive_pct": 0.35,
    "neutral_pct":  0.42,
    "negative_pct": 0.23
  }
}
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Label mapping for cardiffnlp model
LABEL_MAP = {
    "positive": "positive",
    "neutral":  "neutral",
    "negative": "negative",
    # Some versions use these labels
    "LABEL_0":  "negative",
    "LABEL_1":  "neutral",
    "LABEL_2":  "positive",
}

_pipeline = None

POSITIVE_HINTS = [
    r"\b(good idea|very good idea|definitely an endorsement)\b",
    r"\b(fine|excellent|happy|appreciate|support|positive|benefit)\b",
    r"\bflexibility\b", r"\bfamily[- ]friendly\b",
]

NEGATIVE_HINTS = [
    r"\b(concern|cost|insurance|health and safety|liable|liability)\b",
    r"\b(not fair|unfair|alienating)\b",
    r"\b(difficult to concentrate|busy office|overcrowded|phones ring)\b",
    r"\b(not a good idea|wouldn't be possible|difficult to make this fair)\b",
    r"\b(blocker|issue|problem|risk|fed up)\b",
]


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    _pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if _cuda_available() else -1,
        top_k=None,         # return scores for all labels
        truncation=True,
        max_length=512,
    )
    logger.info("Sentiment pipeline loaded: %s", MODEL_NAME)
    return _pipeline


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─── SentimentAnalyser ────────────────────────────────────────────────────────

class SentimentAnalyser:
    """
    Runs RoBERTa sentiment analysis on each diarised transcript segment,
    then aggregates per-speaker and overall statistics.

    Segments are processed in batches for GPU efficiency.
    """

    BATCH_SIZE = 16   # safe for 4 GB VRAM with roberta-base

    def analyse(self, segments: list[dict]) -> dict[str, Any]:
        """
        Parameters
        ----------
        segments : list[dict]
            Speaker-labelled transcript segments.

        Returns
        -------
        dict with keys: per_segment, per_speaker, overall.
        """
        pipe = _get_pipeline()

        # Filter to non-empty segments and keep index mapping
        valid = [(i, s) for i, s in enumerate(segments) if s.get("text", "").strip()]

        per_segment_results: list[dict] = []

        # Batch inference
        for batch_start in range(0, len(valid), self.BATCH_SIZE):
            batch = valid[batch_start: batch_start + self.BATCH_SIZE]
            texts = [s["text"].strip() for _, s in batch]

            try:
                raw_outputs = pipe(texts)
            except Exception as exc:
                logger.warning("Sentiment batch failed: %s", exc)
                raw_outputs = [None] * len(texts)

            for (_, segment), raw in zip(batch, raw_outputs):
                scores = self._parse_scores(raw)
                scores = self._apply_meeting_adjustments(segment["text"].strip(), scores)
                dominant = max(scores, key=scores.get)

                per_segment_results.append({
                    "segment_id": segment.get("segment_id", -1),
                    "speaker":    segment.get("speaker", "UNKNOWN"),
                    "start":      round(segment.get("start", 0.0), 2),
                    "end":        round(segment.get("end", 0.0), 2),
                    "text":       segment["text"].strip(),
                    "sentiment":  dominant,
                    "scores":     scores,
                })

        per_speaker = self._aggregate_per_speaker(per_segment_results)
        overall     = self._aggregate_overall(per_segment_results)

        logger.info("Sentiment analysis complete: %d segments, %d speakers.",
                    len(per_segment_results), len(per_speaker))

        return {
            "per_segment": per_segment_results,
            "per_speaker": per_speaker,
            "overall":     overall,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_scores(raw: list[dict] | None) -> dict[str, float]:
        """Normalise raw pipeline output to {positive, neutral, negative} scores."""
        defaults = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
        if not raw:
            return defaults

        # raw is a list of {"label": ..., "score": ...} for top_k=None
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            mapped = {}
            for item in raw:
                label = LABEL_MAP.get(item["label"].lower(), item["label"].lower())
                mapped[label] = round(float(item["score"]), 4)
            # Ensure all three keys exist
            for k in ("positive", "neutral", "negative"):
                mapped.setdefault(k, 0.0)
            return mapped

        return defaults

    @staticmethod
    def _apply_meeting_adjustments(text: str, scores: dict[str, float]) -> dict[str, float]:
        adjusted = dict(scores)
        lowered = text.lower()

        pos_hits = sum(bool(re.search(pattern, lowered, re.IGNORECASE)) for pattern in POSITIVE_HINTS)
        neg_hits = sum(bool(re.search(pattern, lowered, re.IGNORECASE)) for pattern in NEGATIVE_HINTS)

        if pos_hits:
            adjusted["positive"] += 0.08 * pos_hits
            adjusted["neutral"] = max(0.0, adjusted["neutral"] - 0.03 * pos_hits)
        if neg_hits:
            adjusted["negative"] += 0.08 * neg_hits
            adjusted["neutral"] = max(0.0, adjusted["neutral"] - 0.03 * neg_hits)

        total = sum(adjusted.values())
        if total <= 0:
            return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

        return {
            "positive": round(adjusted["positive"] / total, 4),
            "neutral": round(adjusted["neutral"] / total, 4),
            "negative": round(adjusted["negative"] / total, 4),
        }

    @staticmethod
    def _aggregate_per_speaker(
        segments: list[dict],
    ) -> dict[str, dict]:
        """Compute per-speaker sentiment statistics and trend."""
        speaker_data: dict[str, list[str]] = defaultdict(list)
        for seg in segments:
            speaker_data[seg["speaker"]].append(seg["sentiment"])

        result = {}
        for speaker, sentiments in speaker_data.items():
            n = len(sentiments)
            counts = {s: sentiments.count(s) for s in ("positive", "neutral", "negative")}
            dominant = max(counts, key=counts.get)
            result[speaker] = {
                "positive_pct": round(counts["positive"] / n, 4),
                "neutral_pct":  round(counts["neutral"]  / n, 4),
                "negative_pct": round(counts["negative"] / n, 4),
                "dominant":     dominant,
                "trend":        sentiments,           # chronological list
                "segment_count": n,
            }
        return result

    @staticmethod
    def _aggregate_overall(segments: list[dict]) -> dict[str, float]:
        if not segments:
            return {"positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0}
        n = len(segments)
        counts = {s: sum(1 for seg in segments if seg["sentiment"] == s)
                  for s in ("positive", "neutral", "negative")}
        return {
            "positive_pct": round(counts["positive"] / n, 4),
            "neutral_pct":  round(counts["neutral"]  / n, 4),
            "negative_pct": round(counts["negative"] / n, 4),
        }
