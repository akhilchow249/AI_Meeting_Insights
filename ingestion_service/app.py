"""
app.py — FastAPI Video Ingestion Endpoint
Stage 1: Upload → Validation → Metadata extraction → Session creation
Stage 2: Audio extraction → Noise reduction → VAD (triggered async)
"""

import asyncio
import json
import logging
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import redis.asyncio as aioredis
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from extractor import AudioExtractor, ExtractionError
from observability import configure_json_logging, log_event
from vad import VoiceActivityDetector

configure_json_logging("ingestion_service")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB
MAX_DURATION_SECONDS = 2 * 3600                 # 2 hours

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
ALLOWED_MIME_TYPES  = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/webm", "video/x-matroska",
}

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/tmp/meeting_insights"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ─── App & Redis ──────────────────────────────────────────────────────────────

app = FastAPI(title="Meeting Insights — Ingestion Service", version="1.0.0")
redis_client: Optional[aioredis.Redis] = None


@app.on_event("startup")
async def startup_event():
    global redis_client
    redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
    log_event(logger, "service.startup", stage="ingestion", message="Ingestion service ready")


@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.aclose()
    log_event(logger, "service.shutdown", stage="ingestion", message="Ingestion service stopped")


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class IngestionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    metadata: Optional[dict] = None


class ProgressEvent(BaseModel):
    session_id: str
    stage: str
    status: str
    percent: Optional[int] = None
    detail: Optional[str] = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def publish_progress(session_id: str, stage: str, status: str,
                           percent: int = 100, detail: str = ""):
    """Publish a progress event to Redis pub/sub channel."""
    if redis_client is None:
        return
    event = ProgressEvent(
        session_id=session_id,
        stage=stage,
        status=status,
        percent=percent,
        detail=detail,
    )
    channel = f"progress:{session_id}"
    await redis_client.publish(channel, event.model_dump_json())
    # Also persist latest state as a hash for late-joiners
    await redis_client.hset(f"session:{session_id}", mapping={
        "stage": stage,
        "status": status,
        "percent": str(percent),
        "detail": detail,
    })
    await redis_client.expire(f"session:{session_id}", 86400)  # 24 h TTL


def _validate_extension(filename: str) -> str:
    """Return lowercase extension or raise HTTPException."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_extension",
                "message": f"Extension '{ext}' is not supported.",
                "allowed": sorted(ALLOWED_EXTENSIONS),
            },
        )
    return ext


def _validate_mime(content_type: str):
    """Raise if MIME type is not a known video type."""
    # Browsers sometimes omit the exact subtype; accept any video/* as a hint
    if not content_type.startswith("video/") and content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_mime_type",
                "message": f"MIME type '{content_type}' is not a supported video type.",
                "allowed": sorted(ALLOWED_MIME_TYPES),
            },
        )


async def _stream_to_disk(upload: UploadFile, dest: Path) -> int:
    """Stream upload to disk in chunks; return total bytes written."""
    total = 0
    chunk_size = 1024 * 1024  # 1 MB
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(chunk_size):
            total += len(chunk)
            if total > MAX_FILE_SIZE_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "file_too_large",
                        "message": "File exceeds the 2 GB limit.",
                        "max_bytes": MAX_FILE_SIZE_BYTES,
                    },
                )
            await f.write(chunk)
    return total


# ─── Background pipeline ──────────────────────────────────────────────────────

async def run_pipeline(session_id: str, raw_video_path: Path):
    """
    Asynchronous pipeline executed after the upload completes.
    Runs Stage 1 validation + Stage 2 audio processing, emitting
    Redis progress events at each checkpoint.
    """
    extractor = AudioExtractor()
    vad = VoiceActivityDetector()

    session_dir = STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ── Validation & metadata ──────────────────────────────────────
    ingestion_started = time.perf_counter()
    log_event(
        logger,
        "stage.started",
        session_id=session_id,
        stage="ingestion",
        metadata={"video_path": str(raw_video_path)},
        message="Starting ingestion stage",
    )

    try:
        await publish_progress(session_id, "ingestion", "running", 10,
                               "Probing video metadata…")
        

        metadata = await asyncio.to_thread(extractor.probe_metadata, raw_video_path)

        # Duration check
        duration = metadata.get("duration", 0)
        if duration > MAX_DURATION_SECONDS:
            await publish_progress(session_id, "ingestion", "failed", 0,
                                   f"Duration {duration:.0f}s exceeds 2-hour limit.")
            return

        # Audio-track check
        if not metadata.get("has_audio"):
            await publish_progress(session_id, "ingestion", "failed", 0,
                                   "No audio track detected in video.")
            return

        # ── Extract first-frame thumbnail ─────────────────────────────────────
        # Required by the upload screen UI — shown immediately after upload.
        thumbnail_path = session_dir / "thumbnail.jpg"
        has_thumbnail = await asyncio.to_thread(
            extractor.extract_thumbnail, raw_video_path, thumbnail_path
        )
        metadata["thumbnail_path"] = str(thumbnail_path) if has_thumbnail else None
        metadata["has_thumbnail"]  = has_thumbnail

        # Persist metadata alongside the raw file
        meta_path = session_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        await publish_progress(session_id, "ingestion", "complete", 100,
                               "Metadata extracted successfully.")
        log_event(
            logger,
            "stage.completed",
            session_id=session_id,
            stage="ingestion",
            duration_ms=(time.perf_counter() - ingestion_started) * 1000,
            metadata={
                "duration_seconds": round(float(duration or 0), 2),
                "has_audio": bool(metadata.get("has_audio")),
                "has_thumbnail": bool(has_thumbnail),
            },
            message="Ingestion stage complete",
        )

    except ExtractionError as exc:
        await publish_progress(session_id, "ingestion", "failed", 0, str(exc))
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="ingestion",
            duration_ms=(time.perf_counter() - ingestion_started) * 1000,
            metadata={"video_path": str(raw_video_path)},
            message="Ingestion stage failed",
            level=logging.ERROR,
            exc_info=exc,
        )
        return
    except Exception as exc:
        await publish_progress(session_id, "ingestion", "failed", 0,
                               f"Unexpected error during ingestion: {exc}")
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="ingestion",
            duration_ms=(time.perf_counter() - ingestion_started) * 1000,
            metadata={"video_path": str(raw_video_path)},
            message="Unexpected ingestion failure",
            level=logging.ERROR,
            exc_info=exc,
        )
        return

    # ── Stage 2 ── Audio extraction → noise reduction ─────────────────────────
    wav_path = session_dir / "audio_raw.wav"
    denoised_path = session_dir / "audio_denoised.wav"

    audio_started = time.perf_counter()
    log_event(
        logger,
        "stage.started",
        session_id=session_id,
        stage="audio_extraction",
        metadata={"wav_filename": wav_path.name, "denoised_filename": denoised_path.name},
        message="Starting audio extraction stage",
    )

    try:
        await publish_progress(session_id, "audio_extraction", "running", 10,
                               "Extracting audio track with FFmpeg…")

        await asyncio.to_thread(
            extractor.extract_audio,
            raw_video_path,
            wav_path,
        )

        await publish_progress(session_id, "audio_extraction", "running", 40,
                               "Applying noise reduction…")

        await asyncio.to_thread(
            extractor.denoise_audio,
            wav_path,
            denoised_path,
        )

        await publish_progress(session_id, "audio_extraction", "running", 70,
                               "Running Voice Activity Detection…")

        speech_segments = await asyncio.to_thread(
            vad.detect_speech_segments,
            denoised_path,
        )

        segments_path = session_dir / "speech_segments.json"
        segments_path.write_text(json.dumps(speech_segments, indent=2))

        await publish_progress(session_id, "audio_extraction", "complete", 100,
                               f"{len(speech_segments)} speech segments identified.")
        log_event(
            logger,
            "stage.completed",
            session_id=session_id,
            stage="audio_extraction",
            duration_ms=(time.perf_counter() - audio_started) * 1000,
            metadata={
                "speech_segment_count": len(speech_segments),
                "raw_audio_path": str(wav_path),
                "denoised_audio_path": str(denoised_path),
            },
            message="Audio extraction stage complete",
        )

        if not denoised_path.exists():
            raise FileNotFoundError(f"Expected denoised audio not found: {denoised_path}")

    except ExtractionError as exc:
        await publish_progress(session_id, "audio_extraction", "failed", 0, str(exc))
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="audio_extraction",
            duration_ms=(time.perf_counter() - audio_started) * 1000,
            metadata={"video_path": str(raw_video_path)},
            message="Audio extraction stage failed",
            level=logging.ERROR,
            exc_info=exc,
        )
    except Exception as exc:
        await publish_progress(session_id, "audio_extraction", "failed", 0,
                               f"Unexpected error during audio extraction: {exc}")
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="audio_extraction",
            duration_ms=(time.perf_counter() - audio_started) * 1000,
            metadata={"video_path": str(raw_video_path)},
            message="Unexpected audio extraction failure",
            level=logging.ERROR,
            exc_info=exc,
        )


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestionResponse, status_code=202)
async def ingest_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a multipart video upload.  Validates extension and MIME type
    synchronously, streams the file to disk, then fires the processing
    pipeline as a background task and immediately returns 202 Accepted
    with the session_id the client can use to poll progress.
    """
    # ── Synchronous pre-checks (fast, no I/O) ──
    _validate_extension(file.filename or "")
    _validate_mime(file.content_type or "application/octet-stream")

    session_id = str(uuid.uuid4())
    session_dir = STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix.lower()
    raw_path = session_dir / f"raw{ext}"

    # ── Stream upload to disk ──
    try:
        file_size = await _stream_to_disk(file, raw_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    log_event(
        logger,
        "upload.accepted",
        session_id=session_id,
        stage="ingestion",
        metadata={
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size_bytes": file_size,
            "storage_path": str(raw_path),
        },
        message="Video upload accepted",
    )

    # ── Kick off pipeline ──
    background_tasks.add_task(run_pipeline, session_id, raw_path)

    return IngestionResponse(
        session_id=session_id,
        status="accepted",
        message="Video received. Processing has started.",
        metadata={"file_size_bytes": file_size, "original_filename": file.filename},
    )


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Fetch the latest processing state for a session."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available.")
    data = await redis_client.hgetall(f"session:{session_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    return JSONResponse(content={"session_id": session_id, **data})


@app.get("/thumbnail/{session_id}")
async def get_thumbnail(session_id: str):
    """
    Return the first-frame JPEG thumbnail extracted from the uploaded video.

    Used by the upload screen to show a preview card immediately after upload.
    Returns 404 if the session doesn't exist or the video has no video stream.
    """
    from fastapi.responses import FileResponse

    thumbnail_path = STORAGE_DIR / session_id / "thumbnail.jpg"
    if not thumbnail_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Thumbnail not available. Either the session does not exist "
                   "or the uploaded file has no video stream.",
        )
    return FileResponse(
        path=str(thumbnail_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=86400"},   # cache for 24 h in browser
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
