"""
asr-service/whisper_chunker.py
──────────────────────────────
Handles:
  1. Splitting a long audio file into 30-second segments with 5-second overlap.
  2. Transcribing each chunk in parallel via ProcessPoolExecutor.
  3. Converting relative chunk timestamps → absolute file timestamps.
  4. Merging chunks: detecting and removing duplicate words in the overlap zone.

Why chunking?
─────────────
Whisper large-v3 has an internal 30-second context window. For audio longer
than 30 s it silently truncates or hallucinates. Chunking with overlap prevents
word cuts at boundaries and allows parallel CPU/GPU utilisation.

Overlap merge strategy
──────────────────────
Given:
  chunk i  covers [A,  A+30]   step = chunk_duration − overlap = 25 s
  chunk i+1 covers [A+25, A+55]

The overlap zone is [A+25, A+30].  After converting to absolute times,
chunk i+1 may produce duplicate words for this zone.  We resolve this by:
  • Keeping ALL words from chunk i (they have full left-context).
  • For chunk i+1, discarding words whose absolute start time falls
    strictly before the chunk's "safe start" = chunk_start + overlap.
  • Exception: if a word straddles the safe-start boundary (it starts
    just before but ends after), we keep the later chunk's version
    because it has more right-context.

This is the same strategy used by production systems like AssemblyAI.
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from observability import configure_json_logging, log_event

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000   # Whisper requires 16 kHz mono


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RawWord:
    """A single word emitted by Whisper with absolute timestamps."""
    word:       str
    start:      float   # seconds, absolute within the full audio file
    end:        float
    confidence: float   # probability in [0, 1]
    chunk_idx:  int     # which chunk produced this word


@dataclass
class ChunkSpec:
    """Describes one audio chunk to be transcribed."""
    idx:          int
    start_sample: int
    end_sample:   int
    start_sec:    float     # = start_sample / SAMPLE_RATE
    safe_start_sec: float   # words before this are dropped during merge
                            # (= start_sec + overlap, except for chunk 0)


@dataclass
class ChunkTranscriptionResult:
    words: list["RawWord"]
    chunk_duration_s: float
    transcription_time_ms: int
    confidence_avg: float


# ─── Errors ───────────────────────────────────────────────────────────────────

class ChunkerError(RuntimeError):
    """Raised on unrecoverable chunking or transcription errors."""


# ─── Worker initialiser (runs once per process) ───────────────────────────────

_worker_model = None   # module-level singleton inside each worker process


def _init_worker(model_size: str, device: str, compute_type: str) -> None:
    """
    Called by ProcessPoolExecutor when each worker process starts.
    Loads the Whisper model ONCE per process to avoid repeated disk I/O.
    """
    global _worker_model
    from faster_whisper import WhisperModel
    configure_json_logging("asr_service")
    logger.info("Worker PID %d: loading faster-whisper '%s' on %s/%s…",
                os.getpid(), model_size, device, compute_type)
    _worker_model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        num_workers=1,              # intra-model parallelism (keep at 1 per process)
        cpu_threads=max(1, os.cpu_count() // 2),
    )
    logger.info("Worker PID %d: model loaded.", os.getpid())


def _transcribe_chunk(
    audio_bytes: bytes,
    chunk_spec: ChunkSpec,
    language: str | "en",
    session_id: str | None = None,
) -> ChunkTranscriptionResult:
    """
    Executed inside a worker process.
    Deserialises the audio chunk from bytes, runs Whisper, returns RawWord list.

    Parameters
    ----------
    audio_bytes : bytes
        Raw float32 PCM bytes at SAMPLE_RATE Hz (serialised numpy array).
    chunk_spec  : ChunkSpec
        Metadata about this chunk (used for absolute timestamp offsetting).
    language    : str | None
        BCP-47 language code (e.g. "en"). None → auto-detect per chunk.

    Returns
    -------
    list[RawWord]  with absolute timestamps.
    """
    if _worker_model is None:
        raise RuntimeError("Worker not initialised — model is None.")

    started = time.perf_counter()
    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

    segments_gen, _info = _worker_model.transcribe(
        audio_np,
        language="en",
        task="transcribe",
        word_timestamps=True,
        vad_filter=False,        # VAD already done in ingestion-service
        beam_size=5,
        best_of=5,
        temperature=0.0,         # greedy; reduces hallucinations
        condition_on_previous_text=False,  # chunks are independent
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    words: list[RawWord] = []
    for segment in segments_gen:
        if segment.words is None:
            continue
        for w in segment.words:
            # Convert chunk-relative timestamps → absolute
            abs_start = round(chunk_spec.start_sec + w.start, 3)
            abs_end   = round(chunk_spec.start_sec + w.end,   3)
            words.append(RawWord(
                word=w.word.strip(),
                start=abs_start,
                end=abs_end,
                confidence=round(float(w.probability), 4),
                chunk_idx=chunk_spec.idx,
            ))

    chunk_duration_s = round(len(audio_np) / SAMPLE_RATE, 3)
    confidence_avg = round(
        sum(word.confidence for word in words) / len(words),
        4,
    ) if words else 0.0
    transcription_time_ms = int((time.perf_counter() - started) * 1000)

    log_event(
        logger,
        "transcription.chunk_complete",
        session_id=session_id,
        stage="transcription",
        duration_ms=transcription_time_ms,
        metadata={
            "chunk_id": chunk_spec.idx,
            "chunk_duration_s": chunk_duration_s,
            "confidence_avg": confidence_avg,
            "word_count": len(words),
        },
        message=f"Transcribed chunk {chunk_spec.idx}",
    )

    return ChunkTranscriptionResult(
        words=words,
        chunk_duration_s=chunk_duration_s,
        transcription_time_ms=transcription_time_ms,
        confidence_avg=confidence_avg,
    )


# ─── WhisperChunker ───────────────────────────────────────────────────────────

class WhisperChunker:
    """
    Orchestrates chunked parallel transcription of a long audio file.

    Usage
    -----
    chunker = WhisperChunker(model_size="medium", device="cuda", ...)
    words   = chunker.transcribe(wav_path, progress_callback=cb)
    """

    def __init__(
        self,
        model_size:       str   = "medium",
        device:           str   = "cuda",
        compute_type:     str   = "int8_float16",
        chunk_duration:   float = 30.0,
        overlap_duration: float = 5.0,
        num_workers:      int   = 2,
        language:         str | None = "en",
    ):
        if overlap_duration >= chunk_duration:
            raise ValueError("overlap_duration must be less than chunk_duration.")

        self.model_size       = model_size
        self.device           = device
        self.compute_type     = compute_type
        self.chunk_duration   = chunk_duration
        self.overlap_duration = overlap_duration
        self.step_duration    = chunk_duration - overlap_duration   # 25 s
        self.num_workers      = num_workers
        self.language         = language

    # ── Public ────────────────────────────────────────────────────────────────
    def transcribe(
        self,
        wav_path: Path,
        *,
        session_id: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        word_callback: Callable[[list[RawWord], int, int], None] | None = None,
    ) -> list[RawWord]:
        """
        Transcribe *wav_path* and return a merged list of RawWord objects
        sorted by absolute start time.
        """
        audio = self._load_audio(wav_path)
        specs = self._build_chunk_specs(len(audio))

        if not specs:
            raise ChunkerError(f"Audio file '{wav_path.name}' produced no chunks.")

        logger.info(
            "Transcribing %d chunks (%s each, %s overlap) with %d workers.",
            len(specs),
            f"{self.chunk_duration}s",
            f"{self.overlap_duration}s",
            self.num_workers,
        )

        chunk_results: dict[int, list[RawWord]] = {}

        # Celery workers are daemon processes and cannot create child processes.
        # So when num_workers <= 1, run sequentially in the same process.
        if self.num_workers <= 1:
            logger.info("Running chunk transcription sequentially inside Celery worker.")

            # Initialise the model once in this process
            global _worker_model
            if _worker_model is None:
                _init_worker(self.model_size, self.device, self.compute_type)

            completed = 0

            for spec in specs:
                try:
                    result = _transcribe_chunk(
                        audio[spec.start_sample:spec.end_sample].tobytes(),
                        spec,
                        self.language,
                        session_id,
                    )
                    chunk_results[spec.idx] = result.words
                except Exception as exc:
                    raise ChunkerError(
                        f"Chunk {spec.idx} transcription failed: {exc}"
                    ) from exc

                completed += 1

                if progress_callback:
                    progress_callback(completed, len(specs))

                if word_callback:
                    word_callback(result.words, completed, len(specs))

        else:
            # Original parallel path for non-Celery environments
            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_worker,
                initargs=(self.model_size, self.device, self.compute_type),
            ) as pool:
                futures = {
                    pool.submit(
                        _transcribe_chunk,
                        audio[spec.start_sample:spec.end_sample].tobytes(),
                        spec,
                        self.language,
                        session_id,
                    ): spec.idx
                    for spec in specs
                }

                completed = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        chunk_results[idx] = result.words
                    except Exception as exc:
                        raise ChunkerError(
                            f"Chunk {idx} transcription failed: {exc}"
                        ) from exc

                    completed += 1

                    if progress_callback:
                        progress_callback(completed, len(specs))

                    if word_callback:
                        word_callback(result.words, completed, len(specs))

        return self._merge_chunks(chunk_results, specs)
        
    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_audio(wav_path: Path) -> np.ndarray:
        """Read WAV → float32 mono numpy array at SAMPLE_RATE."""
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as exc:
            raise ChunkerError(f"Cannot read '{wav_path.name}': {exc}") from exc

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sr != SAMPLE_RATE:
            raise ChunkerError(
                f"Expected {SAMPLE_RATE} Hz audio, got {sr} Hz. "
                "Re-run the audio extractor."
            )

        return audio

    def _build_chunk_specs(self, total_samples: int) -> list[ChunkSpec]:
        """
        Divide *total_samples* into overlapping chunks and return ChunkSpec list.

        Chunk boundaries (in samples):
          chunk i: start = i * step_samples
                   end   = min(start + chunk_samples, total_samples)
        """
        chunk_samples   = int(self.chunk_duration   * SAMPLE_RATE)
        step_samples    = int(self.step_duration     * SAMPLE_RATE)
        overlap_samples = int(self.overlap_duration  * SAMPLE_RATE)

        specs = []
        idx   = 0
        start = 0

        while start < total_samples:
            end = min(start + chunk_samples, total_samples)

            # "safe start" = the point from which this chunk's words are trusted
            # For chunk 0 there is no preceding chunk, so safe_start = 0.
            safe_start_sec = (start / SAMPLE_RATE) + (
                0.0 if idx == 0 else self.overlap_duration
            )

            specs.append(ChunkSpec(
                idx=idx,
                start_sample=start,
                end_sample=end,
                start_sec=start / SAMPLE_RATE,
                safe_start_sec=safe_start_sec,
            ))

            if end == total_samples:
                break
            start += step_samples
            idx   += 1

        return specs

    @staticmethod
    def _merge_chunks(
        chunk_results: dict[int, list[RawWord]],
        specs: list[ChunkSpec],
    ) -> list[RawWord]:
        """
        Merge per-chunk word lists into a single de-duplicated list.

        Rule: for chunk i (i > 0), discard any word whose absolute start
        falls before that chunk's safe_start_sec — those words were already
        captured by chunk i-1.
        """
        merged: list[RawWord] = []

        for spec in sorted(specs, key=lambda s: s.idx):
            words = chunk_results.get(spec.idx, [])

            for word in words:
                # Drop empty / punctuation-only tokens
                if not word.word.strip():
                    continue

                # Drop overlap duplicates (except for the very first chunk)
                if spec.idx > 0 and word.start < spec.safe_start_sec:
                    continue

                merged.append(word)

        # Sort by absolute start time (chunks arrive out of order from the pool)
        merged.sort(key=lambda w: w.start)

        # Final deduplication pass: remove consecutive words that are
        # identical in text AND have overlapping timestamps (rare edge case).
        deduped: list[RawWord] = []
        for word in merged:
            if (deduped
                    and deduped[-1].word.lower() == word.word.lower()
                    and word.start < deduped[-1].end):
                # Keep the one with higher confidence
                if word.confidence > deduped[-1].confidence:
                    deduped[-1] = word
            else:
                deduped.append(word)

        logger.info("Merge complete: %d raw words → %d after deduplication.",
                    len(merged), len(deduped))
        return deduped
