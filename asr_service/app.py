"""
Celery application for Stage 3 speech-to-text.

Responsibilities
----------------
- Receive a session_id and path to the denoised WAV from ingestion.
- Run chunked Whisper transcription.
- Publish progress events to Redis for the live pipeline UI.
- Persist the final word-level transcript JSON.
- Trigger diarisation when transcription completes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import redis
import soundfile as sf
from celery import Celery

from observability import configure_json_logging, log_event
from whisper_chunker import ChunkerError, WhisperChunker
from word_timestamps import build_transcript, compute_avg_confidence

configure_json_logging("asr_service")
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/tmp/meeting_insights"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8_float16")
WORKER_PROCESSES = int(os.getenv("WHISPER_WORKERS", "1"))

celery_app = Celery(
    "asr_service",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,
)

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _publish(
    session_id: str,
    stage: str,
    status: str,
    percent: int = 0,
    detail: str = "",
) -> None:
    r = _get_redis()
    event = json.dumps({
        "session_id": session_id,
        "stage": stage,
        "status": status,
        "percent": percent,
        "detail": detail,
    })
    r.publish(f"progress:{session_id}", event)
    r.hset(f"session:{session_id}", mapping={
        "stage": stage,
        "status": status,
        "percent": str(percent),
        "detail": detail,
    })
    r.expire(f"session:{session_id}", 86400)


def _make_progress_callback(session_id: str):
    def _callback(chunks_done: int, total_chunks: int) -> None:
        pct = int((chunks_done / max(total_chunks, 1)) * 100)
        _publish(
            session_id,
            "transcription",
            "running",
            pct,
            f"Transcribed {chunks_done}/{total_chunks} chunks",
        )

    return _callback


def _make_word_callback(session_id: str):
    def _callback(chunk_words, chunks_done: int, total_chunks: int) -> None:
        if not chunk_words:
            return

        r = _get_redis()
        words_payload = [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence,
                "chunk_idx": w.chunk_idx,
            }
            for w in chunk_words
            if w.word.strip()
        ]

        if not words_payload:
            return

        event = json.dumps({
            "type": "transcript_preview",
            "session_id": session_id,
            "words": words_payload,
            "chunks_done": chunks_done,
            "total_chunks": total_chunks,
        })

        r.publish(f"progress:{session_id}", event)
        list_key = f"transcript:preview:{session_id}"
        r.rpush(list_key, json.dumps(words_payload))
        r.expire(list_key, 86400)

    return _callback


@celery_app.task(
    bind=True,
    name="asr_service.transcribe_audio",
    max_retries=2,
    default_retry_delay=10,
)
def transcribe_audio(self, session_id: str, wav_filename: str = "audio_denoised.wav") -> dict:
    """
    Transcribe a denoised WAV file and store the word-level transcript.
    """
    session_dir = STORAGE_DIR / session_id
    wav_path = session_dir / wav_filename
    stage_started = time.perf_counter()

    if not wav_path.exists():
        _publish(session_id, "transcription", "failed", 0, f"WAV file not found: {wav_path}")
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="transcription",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            metadata={"wav_path": str(wav_path)},
            message="Transcription input file missing",
            level=logging.ERROR,
        )
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    _publish(session_id, "transcription", "running", 0, "Initialising Whisper medium...")
    log_event(
        logger,
        "stage.started",
        session_id=session_id,
        stage="transcription",
        metadata={
            "wav_path": str(wav_path),
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE,
            "worker_processes": WORKER_PROCESSES,
        },
        message="Starting transcription stage",
    )

    try:
        audio_info = sf.info(str(wav_path))
        audio_duration_s = round(audio_info.frames / max(audio_info.samplerate, 1), 3)

        chunker = WhisperChunker(
            model_size=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
            chunk_duration=30.0,
            overlap_duration=5.0,
            num_workers=WORKER_PROCESSES,
        )

        raw_words = chunker.transcribe(
            wav_path=wav_path,
            session_id=session_id,
            progress_callback=_make_progress_callback(session_id),
            word_callback=_make_word_callback(session_id),
        )

    except ChunkerError as exc:
        _publish(session_id, "transcription", "failed", 0, str(exc))
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="transcription",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            metadata={"wav_path": str(wav_path)},
            message="Transcription stage failed",
            level=logging.ERROR,
            exc_info=exc,
        )
        raise self.retry(exc=exc)
    except Exception as exc:
        _publish(session_id, "transcription", "failed", 0, str(exc))
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="transcription",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            metadata={"wav_path": str(wav_path)},
            message="Unexpected transcription failure",
            level=logging.ERROR,
            exc_info=exc,
        )
        raise

    transcript_words = build_transcript(raw_words)
    avg_conf = compute_avg_confidence(transcript_words)
    transcription_time_s = round(time.perf_counter() - stage_started, 3)
    rtf = round(transcription_time_s / max(audio_duration_s, 0.001), 4)

    transcript_path = session_dir / "transcript_words.json"
    transcript_path.write_text(
        json.dumps(
            {"words": transcript_words, "avg_confidence": round(avg_conf, 4)},
            indent=2,
        )
    )

    r = _get_redis()
    r.set(
        f"result:asr:{session_id}",
        json.dumps({
            "transcript_path": str(transcript_path),
            "word_count": len(transcript_words),
            "avg_confidence": round(avg_conf, 4),
            "audio_duration_s": audio_duration_s,
            "transcription_time_s": transcription_time_s,
            "rtf": rtf,
        }),
        ex=86400,
    )

    _publish(
        session_id,
        "transcription",
        "complete",
        100,
        f"{len(transcript_words)} words | avg confidence {avg_conf:.2%}",
    )
    log_event(
        logger,
        "stage.completed",
        session_id=session_id,
        stage="transcription",
        duration_ms=transcription_time_s * 1000,
        metadata={
            "word_count": len(transcript_words),
            "avg_confidence": round(avg_conf, 4),
            "audio_duration_s": audio_duration_s,
            "transcription_time_s": transcription_time_s,
            "rtf": rtf,
        },
        message="Transcription stage complete",
    )

    try:
        celery_app.send_task(
            "diarisation_service.diarise_audio",
            kwargs={"session_id": session_id, "wav_filename": wav_filename},
            queue="diarisation",
        )
        log_event(
            logger,
            "pipeline.next_stage_enqueued",
            session_id=session_id,
            stage="transcription",
            metadata={"next_stage": "diarisation", "queue": "diarisation"},
            message="Diarisation task enqueued",
        )
    except Exception as exc:
        log_event(
            logger,
            "pipeline.next_stage_enqueue_failed",
            session_id=session_id,
            stage="transcription",
            metadata={"next_stage": "diarisation", "queue": "diarisation"},
            message="Failed to enqueue diarisation task",
            level=logging.WARNING,
            exc_info=exc,
        )

    return {
        "session_id": session_id,
        "word_count": len(transcript_words),
        "avg_confidence": round(avg_conf, 4),
        "audio_duration_s": audio_duration_s,
        "transcription_time_s": transcription_time_s,
        "rtf": rtf,
        "transcript_path": str(transcript_path),
    }
