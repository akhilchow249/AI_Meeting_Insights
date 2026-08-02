"""
vad.py — Voice Activity Detection using Silero-VAD
Identifies speech vs silence timestamps in a denoised 16kHz mono WAV.

Output format (list of dicts):
  [{"start": 1.24, "end": 4.87}, {"start": 6.10, "end": 12.03}, ...]
  (seconds, relative to the start of the audio file)

This segment list is consumed downstream by the transcription service
so that Whisper only processes speech-containing windows, significantly
reducing both latency and hallucinations on silent segments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# ─── Model singleton ──────────────────────────────────────────────────────────

_silero_model: Any = None
_silero_utils: Any = None


def _get_silero() -> tuple[Any, Any]:
    """
    Lazy-load the Silero VAD model exactly once per process.
    Uses torch.hub; the model is cached at ~/.cache/torch/hub.

    Returns (model, utils) where utils = (get_speech_timestamps,
    save_audio, read_audio, VADIterator, collect_chunks).
    """
    global _silero_model, _silero_utils
    if _silero_model is None:
        logger.info("Loading Silero VAD model from torch.hub…")
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,          # use PyTorch backend for simplicity
        )
        _silero_model.eval()
        logger.info("Silero VAD model loaded.")
    return _silero_model, _silero_utils


# ─── VoiceActivityDetector ────────────────────────────────────────────────────

class VoiceActivityDetector:
    """
    Wraps Silero VAD to produce timestamped speech segments from a WAV file.

    Key design decisions
    --------------------
    • Silero VAD requires **16 kHz** audio (exactly). The extractor already
      produces this, but we validate and resample defensively.
    • Silero operates on 30 ms (512-sample) or 60 ms (1024-sample) windows at
      16 kHz.  We use 512 for finer temporal resolution.
    • min_speech_duration_ms / min_silence_duration_ms are tuned for typical
      meeting speech patterns (≥ 250 ms speech, ≥ 300 ms silence gap).
    • Segments are expanded by *padding_ms* on each side so that the
      transcription model receives a small context window around each utterance.
    """

    REQUIRED_SAMPLE_RATE: int = 16_000
    WINDOW_SIZE_SAMPLES: int = 512  # 32 ms at 16 kHz

    def __init__(
        self,
        threshold: float = 0.50,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        padding_ms: int = 100,
    ):
        """
        Parameters
        ----------
        threshold : float
            VAD confidence threshold (0–1). Higher → fewer false positives.
            0.5 is the recommended default from the Silero authors.
        min_speech_duration_ms : int
            Discard speech segments shorter than this (avoids noise bursts).
        min_silence_duration_ms : int
            Silence gaps shorter than this are merged into adjacent speech.
        padding_ms : int
            Milliseconds of audio to add before/after each detected segment.
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.padding_ms = padding_ms

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_speech_segments(
        self, wav_path: Path
    ) -> list[dict[str, float]]:
        """
        Run VAD on *wav_path* and return a list of speech segments.

        Each segment: {"start": <float seconds>, "end": <float seconds>}

        Raises
        ------
        ValueError  : if the file has an incompatible sample rate and
                      on-the-fly resampling is unavailable.
        RuntimeError: if the model fails unexpectedly.
        """
        audio, sr = self._load_audio(wav_path)

        if sr != self.REQUIRED_SAMPLE_RATE:
            audio, sr = self._resample(audio, sr, self.REQUIRED_SAMPLE_RATE)

        model, utils = _get_silero()
        get_speech_timestamps, *_ = utils

        # Convert to torch tensor (float32, shape [N])
        audio_tensor = torch.from_numpy(audio).float()

        logger.info("Running VAD on %d samples (%.1f s)…",
                    len(audio_tensor), len(audio_tensor) / sr)

        raw_timestamps: list[dict[str, int]] = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sr,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            window_size_samples=self.WINDOW_SIZE_SAMPLES,
            return_seconds=False,   # we convert manually for consistency
        )

        segments = self._timestamps_to_seconds(raw_timestamps, sr, len(audio))

        logger.info("VAD complete: %d speech segments detected.", len(segments))
        return segments

    def compute_speech_ratio(self, segments: list[dict[str, float]], duration: float) -> float:
        """
        Return the fraction of *duration* that is classified as speech.
        Useful for detecting silent / corrupted recordings early.
        """
        if not duration:
            return 0.0
        total_speech = sum(s["end"] - s["start"] for s in segments)
        return min(total_speech / duration, 1.0)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_audio(wav_path: Path) -> tuple[np.ndarray, int]:
        """Read a WAV file and return (float32 mono array, sample_rate)."""
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as exc:
            raise RuntimeError(f"Cannot read '{wav_path.name}': {exc}") from exc

        if audio.ndim == 2:
            audio = audio.mean(axis=1)   # stereo → mono

        # Normalise to [-1, 1]
        peak = np.abs(audio).max()
        if peak > 0:
            audio /= peak

        return audio, sr

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
        """
        Resample *audio* from *orig_sr* to *target_sr* using librosa if available,
        otherwise raise a descriptive error.
        """
        try:
            import librosa  # optional dependency
            logger.warning(
                "Audio is %d Hz; resampling to %d Hz with librosa.", orig_sr, target_sr
            )
            resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            return resampled, target_sr
        except ImportError:
            raise ValueError(
                f"Audio sample rate is {orig_sr} Hz but Silero VAD requires "
                f"{target_sr} Hz. Install librosa for automatic resampling, "
                "or ensure the extractor produces 16 kHz WAV files."
            )

    def _timestamps_to_seconds(
        self,
        raw: list[dict[str, int]],
        sr: int,
        total_samples: int,
    ) -> list[dict[str, float]]:
        """
        Convert sample-index timestamps → seconds, applying *padding_ms*.

        Clamps values to [0, total_duration] so padded segments never exceed
        file boundaries.
        """
        padding_samples = int(self.padding_ms * sr / 1000)
        total_duration = total_samples / sr
        segments = []

        for ts in raw:
            start_s = max(0, ts["start"] - padding_samples) / sr
            end_s = min(total_samples, ts["end"] + padding_samples) / sr
            end_s = min(end_s, total_duration)

            if end_s > start_s:  # safety guard
                segments.append({
                    "start": round(start_s, 3),
                    "end": round(end_s, 3),
                })

        return segments
