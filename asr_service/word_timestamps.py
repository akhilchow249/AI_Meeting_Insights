"""
asr-service/word_timestamps.py
───────────────────────────────
Post-processes the raw RawWord list from WhisperChunker into the final
canonical word schema required by downstream services (diarisation, UI).

Output schema per word
──────────────────────
{
    "word":        str,    # cleaned word text
    "start":       float,  # seconds from file start
    "end":         float,
    "confidence":  float,  # [0, 1]  whisper word probability
    "segment_id":  int     # monotonically increasing sentence-level grouping
}

Segment assignment
──────────────────
A new segment_id is assigned whenever:
  • A gap > SEGMENT_GAP_THRESHOLD seconds exists between consecutive words, OR
  • A sentence-ending punctuation mark (. ? !) terminates the previous word.

This gives a coarse sentence segmentation that downstream services can refine
with speaker boundaries from pyannote.
"""

from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisper_chunker import RawWord  # type: ignore

# ─── Constants ────────────────────────────────────────────────────────────────

SEGMENT_GAP_THRESHOLD = 1.5   # seconds of silence → new segment
_SENTENCE_END_RE      = re.compile(r"[.?!]['\"]?\s*$")
_STRIP_RE             = re.compile(r"[^\w''\-]+")   # keep contractions & hyphens


# ─── Public API ───────────────────────────────────────────────────────────────

def build_transcript(raw_words: "list[RawWord]") -> list[dict]:
    """
    Convert a list of RawWord objects into the canonical word dict schema.

    Steps
    -----
    1. Clean each word (strip leading/trailing whitespace; skip empties).
    2. Assign monotonically increasing segment_ids based on silence gaps
       and sentence-ending punctuation.
    3. Return sorted list of dicts.

    Parameters
    ----------
    raw_words : list[RawWord]
        Output of WhisperChunker.transcribe() — already sorted by start time.

    Returns
    -------
    list[dict]  matching the canonical schema above.
    """
    result: list[dict] = []
    segment_id = 0
    prev_end: float | None = None
    prev_word_text: str = ""

    for rw in raw_words:
        text = rw.word.strip()
        if not text:
            continue

        # Detect segment boundary
        if prev_end is not None:
            gap = rw.start - prev_end
            sentence_ended = bool(_SENTENCE_END_RE.search(prev_word_text))
            if gap > SEGMENT_GAP_THRESHOLD or sentence_ended:
                segment_id += 1

        result.append({
            "word":       text,
            "start":      rw.start,
            "end":        rw.end,
            "confidence": rw.confidence,
            "segment_id": segment_id,
        })

        prev_end       = rw.end
        prev_word_text = text

    return result


def compute_avg_confidence(words: list[dict]) -> float:
    """
    Return the mean word-level confidence across the transcript.
    Used as the quality indicator surfaced in the UI.

    Returns 0.0 for an empty transcript.
    """
    if not words:
        return 0.0
    scores = [w["confidence"] for w in words if isinstance(w.get("confidence"), (int, float))]
    return statistics.mean(scores) if scores else 0.0


def words_to_segments(words: list[dict]) -> list[dict]:
    """
    Group the flat word list into segment-level dicts.
    Each segment contains its constituent words, the full text, start/end
    timestamps, and an average confidence score.

    Useful for rendering a sentence-per-line transcript preview before
    diarisation completes.

    Returns
    -------
    list[dict] ::
        [
          {
            "segment_id" : int,
            "start"      : float,
            "end"        : float,
            "text"       : str,
            "words"      : list[dict],
            "confidence" : float,
          },
          ...
        ]
    """
    if not words:
        return []

    segments: list[dict] = []
    current_seg_id  = words[0]["segment_id"]
    current_words: list[dict] = []

    def _flush(seg_id: int, seg_words: list[dict]) -> dict:
        text  = " ".join(w["word"] for w in seg_words)
        conf  = compute_avg_confidence(seg_words)
        return {
            "segment_id": seg_id,
            "start":      seg_words[0]["start"],
            "end":        seg_words[-1]["end"],
            "text":       text,
            "words":      seg_words,
            "confidence": round(conf, 4),
        }

    for word in words:
        if word["segment_id"] != current_seg_id:
            if current_words:
                segments.append(_flush(current_seg_id, current_words))
            current_seg_id = word["segment_id"]
            current_words  = []
        current_words.append(word)

    if current_words:
        segments.append(_flush(current_seg_id, current_words))

    return segments
