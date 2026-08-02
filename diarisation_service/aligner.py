"""
Align Whisper word timestamps with pyannote speaker segments.
"""

from __future__ import annotations

import bisect
import logging
import statistics
from typing import NamedTuple

logger = logging.getLogger(__name__)


class AlignmentError(RuntimeError):
    """Raised when alignment cannot proceed."""


class _SpeakerInterval(NamedTuple):
    start: float
    end: float
    speaker: str


class Aligner:
    """
    Merge Whisper word timestamps with diarisation turns.
    """

    def __init__(self, conflict_threshold_s: float = 0.5):
        self.conflict_threshold_s = conflict_threshold_s
        self.alignment_conflicts = 0
        self.total_segments = 0

    def assign_speakers_to_words(
        self,
        words: list[dict],
        speaker_segments: list[dict],
    ) -> list[dict]:
        if not speaker_segments:
            raise AlignmentError("Speaker segment list is empty.")

        intervals: list[_SpeakerInterval] = sorted(
            (_SpeakerInterval(s["start"], s["end"], s["speaker"]) for s in speaker_segments),
            key=lambda iv: iv.start,
        )
        starts = [iv.start for iv in intervals]

        self.alignment_conflicts = 0
        self.total_segments = len(speaker_segments)

        labelled: list[dict] = []
        for word in words:
            speaker, conflict = self._find_speaker(
                word["start"],
                word["end"],
                (word["start"] + word["end"]) / 2.0,
                intervals,
                starts,
            )
            if conflict:
                self.alignment_conflicts += 1
            labelled.append({**word, "speaker": speaker})

        logger.info(
            "Assigned speakers to %d words with %d alignment conflicts.",
            len(labelled),
            self.alignment_conflicts,
        )
        return labelled

    def group_into_segments(self, labelled_words: list[dict]) -> list[dict]:
        if not labelled_words:
            return []

        merge_gap_threshold = 3.0
        segments: list[dict] = []
        segment_id = 0

        current_speaker = labelled_words[0]["speaker"]
        current_words: list[dict] = [labelled_words[0]]

        for word in labelled_words[1:]:
            gap = word["start"] - current_words[-1]["end"]
            speaker_change = word["speaker"] != current_speaker
            long_pause = gap > merge_gap_threshold

            if speaker_change or long_pause:
                segments.append(self._flush_segment(segment_id, current_speaker, current_words))
                segment_id += 1
                current_speaker = word["speaker"]
                current_words = []

            current_words.append(word)

        if current_words:
            segments.append(self._flush_segment(segment_id, current_speaker, current_words))

        logger.info("Grouped into %d speaker segments.", len(segments))
        return segments

    def _find_speaker(
        self,
        w_start: float,
        w_end: float,
        w_mid: float,
        intervals: list[_SpeakerInterval],
        starts: list[float],
    ) -> tuple[str, bool]:
        candidates: list[tuple[float, str]] = []

        idx = bisect.bisect_right(starts, w_end) - 1
        if idx < 0:
            idx = 0

        for direction in (-1, 0, 1):
            i = idx + direction
            if i < 0 or i >= len(intervals):
                continue
            iv = intervals[i]
            overlap = min(w_end, iv.end) - max(w_start, iv.start)
            if overlap > 0:
                candidates.append((overlap, iv.speaker))

        if candidates:
            candidates.sort(key=lambda candidate: -candidate[0])
            return candidates[0][1], False

        nearest = min(
            intervals,
            key=lambda iv: abs(((iv.start + iv.end) / 2.0) - w_mid),
        )

        if w_end < nearest.start:
            disagreement = nearest.start - w_end
        elif w_start > nearest.end:
            disagreement = w_start - nearest.end
        else:
            disagreement = 0.0

        conflict = disagreement > self.conflict_threshold_s
        logger.debug(
            "Assigned nearest speaker %s for uncovered word %.3f-%.3f (gap %.3fs).",
            nearest.speaker,
            w_start,
            w_end,
            disagreement,
        )
        return nearest.speaker, conflict

    @staticmethod
    def _flush_segment(
        segment_id: int,
        speaker: str,
        words: list[dict],
    ) -> dict:
        text = " ".join(w["word"] for w in words)
        conf_scores = [
            w["confidence"]
            for w in words
            if isinstance(w.get("confidence"), (int, float))
        ]
        avg_conf = statistics.mean(conf_scores) if conf_scores else 0.0

        word_list = [
            {"word": w["word"], "start": w["start"], "end": w["end"]}
            for w in words
        ]

        return {
            "segment_id": segment_id,
            "speaker": speaker,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": text,
            "words": word_list,
            "confidence": round(avg_conf, 4),
        }
