"""
Celery application for Stage 4 speaker diarisation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import redis
import torch
from celery import Celery

from aligner import Aligner, AlignmentError
from observability import configure_json_logging, log_event

configure_json_logging("diarisation_service")
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/tmp/meeting_insights"))
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
DIAR_MODEL = os.getenv("DIAR_MODEL", "pyannote/speaker-diarization-3.1")
DIAR_DEVICE = os.getenv("DIAR_DEVICE", "cpu")
MIN_SPEAKERS = int(os.getenv("MIN_SPEAKERS", "1"))
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "6"))

celery_app = Celery(
    "diarisation_service",
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

_diar_pipeline = None
_redis_client: redis.Redis | None = None


def _get_pipeline():
    global _diar_pipeline
    if _diar_pipeline is not None:
        return _diar_pipeline

    if not HF_TOKEN:
        raise EnvironmentError(
            "HUGGINGFACE_TOKEN is not set. Accept the pyannote model terms and set the token."
        )

    from pyannote.audio import Pipeline

    logger.info("Loading pyannote pipeline '%s' on %s.", DIAR_MODEL, DIAR_DEVICE)
    pipeline = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=HF_TOKEN)
    if DIAR_DEVICE == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    else:
        pipeline.to(torch.device("cpu"))

    _diar_pipeline = pipeline
    logger.info("Pyannote pipeline loaded.")
    return _diar_pipeline


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


@celery_app.task(
    bind=True,
    name="diarisation_service.diarise_audio",
    max_retries=2,
    default_retry_delay=15,
)
def diarise_audio(
    self,
    session_id: str,
    wav_filename: str = "audio_denoised.wav",
) -> dict:
    session_dir = STORAGE_DIR / session_id
    wav_path = session_dir / wav_filename
    words_path = session_dir / "transcript_words.json"
    stage_started = time.perf_counter()

    log_event(
        logger,
        "stage.started",
        session_id=session_id,
        stage="diarisation",
        metadata={
            "wav_path": str(wav_path),
            "transcript_path": str(words_path),
            "model": DIAR_MODEL,
            "device": DIAR_DEVICE,
        },
        message="Starting diarisation stage",
    )

    for path in (wav_path, words_path):
        if not path.exists():
            _publish(session_id, "diarisation", "failed", 0, f"Required file not found: {path.name}")
            log_event(
                logger,
                "stage.failed",
                session_id=session_id,
                stage="diarisation",
                duration_ms=(time.perf_counter() - stage_started) * 1000,
                metadata={"missing_path": str(path)},
                message="Diarisation input file missing",
                level=logging.ERROR,
            )
            raise FileNotFoundError(str(path))

    transcript_data = json.loads(words_path.read_text())
    words: list[dict] = transcript_data.get("words", [])

    if not words:
        _publish(session_id, "diarisation", "failed", 0, "Transcript is empty; cannot diarise.")
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="diarisation",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            message="Diarisation cannot run on an empty transcript",
            level=logging.ERROR,
        )
        return {"session_id": session_id, "speaker_count": 0, "segment_count": 0}

    _publish(session_id, "diarisation", "running", 5, "Loading pyannote pipeline...")

    try:
        pipeline = _get_pipeline()
    except Exception as exc:
        _publish(session_id, "diarisation", "failed", 0, str(exc))
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="diarisation",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            message="Failed to load diarisation pipeline",
            level=logging.ERROR,
            exc_info=exc,
        )
        raise self.retry(exc=exc)

    _publish(session_id, "diarisation", "running", 10, "Running speaker diarisation on full audio...")
    diarisation_started = time.perf_counter()

    try:
        diarisation = pipeline(
            str(wav_path),
            min_speakers=MIN_SPEAKERS,
            max_speakers=MAX_SPEAKERS,
        )
    except Exception as exc:
        _publish(session_id, "diarisation", "failed", 0, f"Pyannote pipeline error: {exc}")
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="diarisation",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            message="Pyannote diarisation failed",
            level=logging.ERROR,
            exc_info=exc,
        )
        raise self.retry(exc=exc)

    raw_speaker_segments: list[dict] = [
        {
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": label,
        }
        for turn, _, label in diarisation.itertracks(yield_label=True)
    ]

    raw_diar_path = session_dir / "diarisation_raw.json"
    raw_diar_path.write_text(json.dumps(raw_speaker_segments, indent=2))

    _publish(
        session_id,
        "diarisation",
        "running",
        60,
        f"{len(raw_speaker_segments)} raw speaker turns detected.",
    )
    _publish(session_id, "diarisation", "running", 70, "Aligning speaker turns with word timestamps...")

    try:
        aligner = Aligner()
        labelled_words = aligner.assign_speakers_to_words(words, raw_speaker_segments)
        speaker_segments = aligner.group_into_segments(labelled_words)
    except AlignmentError as exc:
        _publish(session_id, "diarisation", "failed", 0, str(exc))
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="diarisation",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            metadata={"raw_turn_count": len(raw_speaker_segments)},
            message="Word-speaker alignment failed",
            level=logging.ERROR,
            exc_info=exc,
        )
        raise self.retry(exc=exc)

    unique_speakers = {segment["speaker"] for segment in speaker_segments}
    diarisation_time_ms = int((time.perf_counter() - diarisation_started) * 1000)

    final_path = session_dir / "transcript_speaker.json"
    final_path.write_text(
        json.dumps({
            "speaker_count": len(unique_speakers),
            "segments": speaker_segments,
        }, indent=2)
    )

    r = _get_redis()
    r.set(
        f"result:diarisation:{session_id}",
        json.dumps({
            "transcript_path": str(final_path),
            "speaker_count": len(unique_speakers),
            "segment_count": len(speaker_segments),
            "alignment_conflicts": aligner.alignment_conflicts,
            "raw_turn_count": len(raw_speaker_segments),
            "diarisation_time_ms": diarisation_time_ms,
        }),
        ex=86400,
    )

    _publish(
        session_id,
        "diarisation",
        "complete",
        100,
        f"{len(unique_speakers)} speakers | {len(speaker_segments)} segments",
    )
    log_event(
        logger,
        "stage.completed",
        session_id=session_id,
        stage="diarisation",
        duration_ms=(time.perf_counter() - stage_started) * 1000,
        metadata={
            "num_speakers_detected": len(unique_speakers),
            "total_segments": len(speaker_segments),
            "alignment_conflicts": aligner.alignment_conflicts,
            "raw_turn_count": len(raw_speaker_segments),
            "diarisation_time_ms": diarisation_time_ms,
        },
        message="Diarisation stage complete",
    )

    return {
        "session_id": session_id,
        "speaker_count": len(unique_speakers),
        "segment_count": len(speaker_segments),
        "alignment_conflicts": aligner.alignment_conflicts,
        "raw_turn_count": len(raw_speaker_segments),
        "diarisation_time_ms": diarisation_time_ms,
    }
