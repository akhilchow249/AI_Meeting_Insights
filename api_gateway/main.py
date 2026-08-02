"""
api-gateway/main.py
────────────────────
Central FastAPI gateway — the single entry point for the frontend.

Responsibilities
────────────────
1. Video upload  — proxy POST to ingestion service, register session
2. SSE progress  — subscribe to Redis progress:{session_id}, forward to browser
                   includes transcript_preview events (Stage 3 partial words)
3. Pipeline orchestration — watches ALL sessions via Redis pattern-subscribe
                            and triggers the next stage when current completes:
                              audio_extraction:complete → ASR  (Stage 3)
                              diarisation:complete      → NLP  (Stage 5)
                              nlp_analysis:complete     → marks genai ready
                              genai_report:complete     → persist + index (Stage 7)
4. Transcript endpoints — full transcript + partial preview words
5. Report stream — proxy GenAI service SSE to browser
6. Session persistence/indexing — stores transcript/report payloads in PostgreSQL
                                  and transcript words in Meilisearch
7. Session library — list / detail for all processed meetings
8. Prometheus /metrics endpoint

Pipeline chain (as implemented across services)
───────────────────────────────────────────────
  ingestion   Stage 1+2  FastAPI  :8000   writes denoised WAV
  asr         Stage 3    Celery   queue=asr         ← gateway triggers
  diarisation Stage 4    Celery   queue=diarisation ← asr service triggers
  nlp         Stage 5    Celery   queue=nlp         ← gateway triggers
  genai       Stage 6    FastAPI  :8001             ← gateway notifies ready,
                                                       frontend opens stream
  indexing    Stage 7    gateway  PostgreSQL + search index

Redis key reference
────────────────────
  progress:{session_id}              pub/sub channel (all progress events)
  session:{session_id}               hash  — stage/status/percent/detail
  session:meta:{session_id}          hash  — filename/created_at/duration
  sessions:index                     sorted set — score=created_at, member=session_id
  transcript:preview:{session_id}    list  — JSON word batches from Stage 3
  pipeline:triggered:{session_id}:*  string — idempotency locks
  stage:start:{session_id}:{stage}   string — Unix timestamp, for latency calc
  result:asr:{session_id}            string — JSON ASR result summary
  result:diarisation:{session_id}    string — JSON diarisation result summary
  result:nlp:{session_id}            string — JSON NLP result summary
  result:indexing:{session_id}       string — JSON persistence/index summary
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import inspect
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiohttp
import redis.asyncio as aioredis
from celery import Celery
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from metrics import (
    REGISTRY,
    record_action_items_total,
    record_genai_request,
    record_stage_latency,
    record_stage_completion,
    record_stage_failure,
    record_transcription_confidence,
    record_transcription_rtf,
    record_speaker_count,
    record_pain_point_confidence,
    record_pain_points_total,
    record_sentiment_score,
    set_active_processing_jobs,
    set_queue_depth,
    record_genai_first_token,
    increment_search_latency,
)
from observability import configure_json_logging, log_event
from persistence import SessionArtifactsError, SessionStore, load_session_artifacts

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
configure_json_logging("api_gateway")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

REDIS_URL      = os.getenv("REDIS_URL",      "redis://redis:6379/0")
INGESTION_URL  = os.getenv("INGESTION_URL",  "http://ingestion:8000")
GENAI_URL      = os.getenv("GENAI_URL",      "http://genai:8001")
STORAGE_DIR    = Path(os.getenv("STORAGE_DIR", "/data/sessions"))
DATABASE_URL   = os.getenv("DATABASE_URL", "")
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "none").strip().lower()
SEARCH_URL     = os.getenv("SEARCH_URL", "http://meilisearch:7700").rstrip("/")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "")
MEILI_INDEX_NAME = os.getenv("MEILI_INDEX_NAME", "meeting_transcript_words")

# SSE heartbeat interval — keeps the connection alive through proxies/load balancers
HEARTBEAT_INTERVAL = 15   # seconds

# ─── Celery (send-only — gateway never consumes tasks, only dispatches them) ──

_celery = Celery(broker=REDIS_URL, backend=REDIS_URL)

# ─── Redis connection pool ────────────────────────────────────────────────────
# Two pools:
#   _redis       — regular commands (HGET, SET, GET, LPUSH …)
#   _orchestrator_redis — dedicated pub/sub connection for the pipeline watcher
#
# Redis pub/sub connections must not interleave regular commands, so we keep
# them on separate client instances.

_redis: aioredis.Redis | None = None
_orchestrator_redis: aioredis.Redis | None = None
_session_store: SessionStore | None = None


def _get_redis() -> aioredis.Redis:
    assert _redis is not None, "Redis not initialised"
    return _redis


def _get_session_store() -> SessionStore | None:
    return _session_store


async def _safe_close(obj) -> None:
    """Close redis/pubsub clients across redis-py versions."""
    if obj is None:
        return

    close_fn = getattr(obj, "aclose", None) or getattr(obj, "close", None)
    if close_fn is None:
        return

    result = close_fn()
    if inspect.isawaitable(result):
        await result


def _build_stage_event(
    session_id: str,
    stage: str,
    status: str,
    percent: int,
    detail: str,
) -> dict:
    return {
        "session_id": session_id,
        "stage": stage,
        "status": status,
        "percent": percent,
        "detail": detail,
    }


async def _publish_stage_event(
    session_id: str,
    stage: str,
    status: str,
    percent: int,
    detail: str,
) -> dict:
    r = _get_redis()
    event = _build_stage_event(session_id, stage, status, percent, detail)
    await r.hset(f"session:{session_id}", mapping={
        "stage": stage,
        "status": status,
        "percent": str(percent),
        "detail": detail,
    })
    await r.expire(f"session:{session_id}", 86400)
    await r.publish(f"progress:{session_id}", json.dumps(event))
    return event


async def _fetch_persisted_session(session_id: str) -> dict | None:
    store = _get_session_store()
    if store is None:
        return None
    return await asyncio.to_thread(store.get_session, session_id)


async def _list_persisted_sessions() -> list[dict]:
    store = _get_session_store()
    if store is None:
        return []
    return await asyncio.to_thread(store.list_sessions)


def _persisted_summary(row: dict) -> dict:
    return {
        "session_id": row["session_id"],
        "filename": row.get("filename") or "unknown",
        "created_at": row.get("created_at", 0),
        "file_size_bytes": row.get("file_size_bytes", 0),
        "duration": int(row.get("duration_seconds") or 0),
        "speaker_count": row.get("speaker_count", 0),
        "num_speakers": row.get("speaker_count", 0),
        "word_count": row.get("word_count", 0),
        "avg_confidence": row.get("avg_confidence", 0),
        "stage": row.get("stage", "indexing"),
        "status": row.get("status", "complete"),
        "pain_point_count": row.get("pain_point_count", 0),
        "action_item_count": row.get("action_item_count", 0),
    }


def _persisted_pipeline(row: dict) -> dict:
    pipeline = row.get("pipeline_json") or {}
    if pipeline:
        return pipeline
    return {
        "stage": row.get("stage", "indexing"),
        "status": row.get("status", "complete"),
        "percent": 100 if row.get("status") == "complete" else 0,
        "detail": "Loaded from PostgreSQL",
    }


def _apply_nlp_counts(entry: dict, nlp_payload: dict | None) -> None:
    def _count_items(payload: dict, key: str) -> int:
        if key in payload:
            try:
                return int(payload.get(key) or 0)
            except (TypeError, ValueError):
                pass

        value = payload.get(key, {})
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            nested = value.get(key) if isinstance(value.get(key), list) else None
            if isinstance(nested, list):
                return len(nested)
            for alt in ("items", "entries", "results"):
                alt_value = value.get(alt)
                if isinstance(alt_value, list):
                    return len(alt_value)
        return 0

    if not isinstance(nlp_payload, dict):
        entry.setdefault("pain_point_count", 0)
        entry.setdefault("action_item_count", 0)
        return

    entry["pain_point_count"] = _count_items(nlp_payload, "pain_point_count")
    entry["action_item_count"] = _count_items(nlp_payload, "action_item_count")

    if entry["pain_point_count"] == 0:
        entry["pain_point_count"] = _count_items(nlp_payload, "pain_points")
    if entry["action_item_count"] == 0:
        entry["action_item_count"] = _count_items(nlp_payload, "action_items")


def _apply_report_counts(entry: dict, report_payload: dict | None) -> None:
    if not isinstance(report_payload, dict):
        return
    sections = report_payload.get("sections", [])
    if not isinstance(sections, list):
        return

    def _count_section(index: int) -> int:
        section = next((item for item in sections if int(item.get("index", -1)) == index), None)
        if not isinstance(section, dict):
            return 0
        content = str(section.get("content") or "")
        if not content.strip():
            return 0
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        table_rows = [line for line in lines if "|" in line]
        if len(table_rows) >= 3:
            return max(0, len(table_rows) - 2)
        return len([line for line in lines if line.startswith(("-", "*", "+"))])

    if int(entry.get("pain_point_count") or 0) == 0:
        entry["pain_point_count"] = _count_section(2)
    if int(entry.get("action_item_count") or 0) == 0:
        entry["action_item_count"] = _count_section(3)


async def _load_nlp_payload_for_metrics(session_id: str) -> dict | None:
    persisted = await _fetch_persisted_session(session_id)
    if persisted and persisted.get("nlp_json"):
        return persisted["nlp_json"]

    path = STORAGE_DIR / session_id / "nlp_results.json"
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


async def _refresh_runtime_metrics() -> None:
    r = _get_redis()
    active_jobs = 0
    session_ids = await r.zrange("sessions:index", 0, -1)

    for session_id in session_ids:
        state = await r.hgetall(f"session:{session_id}")
        if state.get("status") in {"running", "ready"}:
            active_jobs += 1

    set_active_processing_jobs(active_jobs)

    for queue_name in ("asr", "diarisation", "nlp"):
        try:
            set_queue_depth(queue_name, await r.llen(queue_name))
        except Exception:
            set_queue_depth(queue_name, 0)


# ─── App lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start background tasks on startup; clean up on shutdown."""
    global _redis, _orchestrator_redis, _session_store

    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    _orchestrator_redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    if DATABASE_URL:
        _session_store = SessionStore(DATABASE_URL)
        await asyncio.to_thread(_session_store.ensure_schema)
        log_event(
            logger,
            "service.startup",
            stage="gateway",
            metadata={"postgres_enabled": True, "search_backend": SEARCH_BACKEND},
            message="API gateway startup complete",
        )
    else:
        _session_store = None
        log_event(
            logger,
            "service.startup",
            stage="gateway",
            metadata={"postgres_enabled": False, "search_backend": SEARCH_BACKEND},
            message="API gateway startup complete without PostgreSQL persistence",
            level=logging.WARNING,
        )

    # Start the pipeline orchestrator as a background task
    orchestrator_task = asyncio.create_task(_pipeline_orchestrator())
    log_event(
        logger,
        "service.orchestrator_started",
        stage="gateway",
        message="Pipeline orchestrator running",
    )

    yield   # ← application runs here

    orchestrator_task.cancel()
    try:
        await orchestrator_task
    except asyncio.CancelledError:
        pass

    await _safe_close(_redis)
    await _safe_close(_orchestrator_redis)
    log_event(logger, "service.shutdown", stage="gateway", message="API gateway shutdown complete")


app = FastAPI(
    title="Meeting Insights — API Gateway",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pipeline Orchestrator ────────────────────────────────────────────────────

async def _pipeline_orchestrator() -> None:
    """
    Background coroutine — pattern-subscribes to ALL progress events across
    ALL sessions and triggers the next pipeline stage when appropriate.

    Stage trigger map
    ─────────────────
      audio_extraction:complete  →  trigger ASR   (Celery asr queue)
      diarisation:complete       →  trigger NLP   (Celery nlp queue)
      nlp_analysis:complete      →  publish genai_ready event
      *:failed                   →  record failure metric

    Idempotency: a Redis SETNX lock prevents double-triggering if multiple
    gateway instances are running or if the event is replayed.
    """
    assert _orchestrator_redis is not None

    pubsub = _orchestrator_redis.pubsub()
    await pubsub.psubscribe("progress:*")   # pattern subscribe — all sessions
    logger.info("Orchestrator: subscribed to progress:* pattern")

    try:
        async for message in pubsub.listen():
            if message["type"] != "pmessage":
                continue

            try:
                raw = message["data"]
                # Skip binary keepalive messages
                if not isinstance(raw, str):
                    continue
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            # transcript_preview events don't represent stage transitions
            if event.get("type") == "transcript_preview":
                continue

            session_id = event.get("session_id")
            stage      = event.get("stage", "")
            status     = event.get("status", "")

            if not session_id:
                continue

            # ── Record stage timing ──────────────────────────────────────────
            await _handle_stage_timing(session_id, stage, status)

            # ── Trigger next stage ───────────────────────────────────────────
            if stage == "audio_extraction" and status == "complete":
                await _trigger_asr(session_id)

            elif stage == "diarisation" and status == "complete":
                await _trigger_nlp(session_id)
                await _read_diarisation_metrics(session_id)

            elif stage == "transcription" and status == "complete":
                await _read_asr_metrics(session_id)

            elif stage == "nlp_analysis" and status == "complete":
                await _notify_genai_ready(session_id)
                await _read_nlp_metrics(session_id)

            # ── Record failures ──────────────────────────────────────────────
            elif status == "failed":
                record_stage_failure(stage)
                logger.warning("[%s] Stage '%s' failed.", session_id, stage)

    except asyncio.CancelledError:
        logger.info("Orchestrator: shutting down.")
    finally:
        await pubsub.punsubscribe("progress:*")
        await _safe_close(pubsub)


async def _handle_stage_timing(session_id: str, stage: str, status: str) -> None:
    """Record stage start time on 'running'; compute and record latency on 'complete'."""
    r   = _get_redis()
    key = f"stage:start:{session_id}:{stage}"

    if status == "running":
        # Only set if not already set (first running event)
        await r.setnx(key, str(time.time()))
        await r.expire(key, 86400)

    elif status == "complete":
        record_stage_completion(stage)
        start_raw = await r.get(key)
        if start_raw:
            latency_ms = int((time.time() - float(start_raw)) * 1000)
            record_stage_latency(stage, latency_ms)
            logger.info("[%s] Stage '%s' completed in %dms.",
                        session_id, stage, latency_ms)


async def _trigger_asr(session_id: str) -> None:
    """Dispatch the ASR Celery task — idempotent."""
    r   = _get_redis()
    key = f"pipeline:triggered:{session_id}:asr"

    if await r.setnx(key, "1"):
        await r.expire(key, 86400)
        try:
            _celery.send_task(
                "asr_service.transcribe_audio",
                kwargs={"session_id": session_id, "wav_filename": "audio_denoised.wav"},
                queue="asr",
            )
            logger.info("[%s] ASR task dispatched.", session_id)
        except Exception as exc:
            logger.error("[%s] Failed to dispatch ASR task: %s", session_id, exc)
            await r.delete(key)   # release lock so retry is possible
    else:
        logger.debug("[%s] ASR already triggered — skipping duplicate.", session_id)


async def _trigger_nlp(session_id: str) -> None:
    """Dispatch the NLP Celery task — idempotent."""
    r   = _get_redis()
    key = f"pipeline:triggered:{session_id}:nlp"

    if await r.setnx(key, "1"):
        await r.expire(key, 86400)
        try:
            _celery.send_task(
                "nlp_service.analyse",
                kwargs={"session_id": session_id},
                queue="nlp",
            )
            logger.info("[%s] NLP task dispatched.", session_id)
        except Exception as exc:
            logger.error("[%s] Failed to dispatch NLP task: %s", session_id, exc)
            await r.delete(key)
    else:
        logger.debug("[%s] NLP already triggered — skipping duplicate.", session_id)


async def _notify_genai_ready(session_id: str) -> None:
    """
    Publish a genai_ready event so the frontend knows it can open the
    report stream.  The genai service generates on demand when the
    frontend connects to /report/{session_id}/stream.
    """
    await _publish_stage_event(
        session_id,
        "genai_report",
        "ready",
        0,
        "NLP complete — AI report ready to stream",
    )
    logger.info("[%s] GenAI ready event published.", session_id)


def _meili_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MEILI_MASTER_KEY:
        headers["Authorization"] = f"Bearer {MEILI_MASTER_KEY}"
    return headers


async def _wait_for_meili_task(
    http: aiohttp.ClientSession,
    task_uid: int,
    *,
    timeout_s: float = 30.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        async with http.get(
            f"{SEARCH_URL}/tasks/{task_uid}",
            headers=_meili_headers(),
        ) as resp:
            payload = await resp.json()
            status = payload.get("status")
            if status == "succeeded":
                return
            if status == "failed":
                err = payload.get("error", {}).get("message") or "Meilisearch task failed"
                raise RuntimeError(err)
        await asyncio.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for Meilisearch task {task_uid}")


async def _ensure_meili_index(http: aiohttp.ClientSession) -> None:
    headers = _meili_headers()
    async with http.get(f"{SEARCH_URL}/indexes/{MEILI_INDEX_NAME}", headers=headers) as resp:
        if resp.status == 404:
            async with http.post(
                f"{SEARCH_URL}/indexes",
                headers=headers,
                json={"uid": MEILI_INDEX_NAME, "primaryKey": "id"},
            ) as create_resp:
                payload = await create_resp.json()
                if create_resp.status >= 400:
                    raise RuntimeError(payload.get("message") or "Failed to create Meilisearch index")
                await _wait_for_meili_task(http, int(payload["taskUid"]))
        elif resp.status >= 400:
            payload = await resp.json()
            raise RuntimeError(payload.get("message") or "Failed to inspect Meilisearch index")

    async with http.patch(
        f"{SEARCH_URL}/indexes/{MEILI_INDEX_NAME}/settings",
        headers=headers,
        json={
            "filterableAttributes": ["session_id"],
            "sortableAttributes": ["start_time", "word_index"],
        },
    ) as settings_resp:
        payload = await settings_resp.json()
        if settings_resp.status >= 400:
            raise RuntimeError(payload.get("message") or "Failed to configure Meilisearch index")
        await _wait_for_meili_task(http, int(payload["taskUid"]))


async def _update_search_index(artifacts) -> dict:
    if SEARCH_BACKEND in ("", "none", "disabled"):
        return {"backend": "disabled", "documents": 0}

    if SEARCH_BACKEND != "meilisearch":
        raise RuntimeError(f"Unsupported SEARCH_BACKEND '{SEARCH_BACKEND}'")

    docs = artifacts.to_meili_documents()
    if not docs:
        return {"backend": "meilisearch", "documents": 0}

    timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        await _ensure_meili_index(http)
        async with http.post(
            f"{SEARCH_URL}/indexes/{MEILI_INDEX_NAME}/documents",
            headers=_meili_headers(),
            json=docs,
        ) as resp:
            payload = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(payload.get("message") or "Failed to update transcript search index")
            await _wait_for_meili_task(http, int(payload["taskUid"]))

    return {"backend": "meilisearch", "documents": len(docs)}


async def _search_transcript_index(
    session_id: str,
    query: str,
    *,
    limit: int = 20,
) -> tuple[list[dict], str]:
    if SEARCH_BACKEND == "meilisearch":
        try:
            timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=10)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(
                    f"{SEARCH_URL}/indexes/{MEILI_INDEX_NAME}/search",
                    headers=_meili_headers(),
                    json={
                        "q": query,
                        "limit": limit,
                        "filter": [f'session_id = "{session_id}"'],
                        "sort": ["start_time:asc"],
                    },
                ) as resp:
                    payload = await resp.json()
                    if resp.status < 400:
                        return payload.get("hits", []), "meilisearch"
                    logger.warning(
                        "[%s] Meilisearch query failed, falling back to PostgreSQL: %s",
                        session_id,
                        payload.get("message") or resp.status,
                    )
        except Exception as exc:
            logger.warning("[%s] Meilisearch unavailable, falling back to PostgreSQL: %s", session_id, exc)

    store = _get_session_store()
    if store is None:
        return [], "disabled"

    hits = await asyncio.to_thread(
        store.search_transcript_words,
        session_id,
        query,
        limit=limit,
    )
    return hits, "postgres"


async def _run_indexing_stage(session_id: str) -> None:
    r = _get_redis()
    store = _get_session_store()
    stage_started = time.perf_counter()
    if store is None:
        await _publish_stage_event(
            session_id,
            "indexing",
            "failed",
            0,
            "PostgreSQL persistence is not configured.",
        )
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="indexing",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            message="Indexing failed because PostgreSQL persistence is not configured",
            level=logging.ERROR,
        )
        return

    lock_key = f"pipeline:triggered:{session_id}:indexing"
    if not await r.setnx(lock_key, "1"):
        logger.debug("[%s] Indexing already triggered — skipping duplicate.", session_id)
        return
    await r.expire(lock_key, 86400)

    persisted_row = False
    try:
        log_event(
            logger,
            "stage.started",
            session_id=session_id,
            stage="indexing",
            metadata={"search_backend": SEARCH_BACKEND},
            message="Starting indexing stage",
        )
        await _publish_stage_event(
            session_id,
            "indexing",
            "running",
            10,
            "Preparing persistence payload…",
        )

        redis_meta = await r.hgetall(f"session:meta:{session_id}")
        pipeline_state = {
            "stage": "indexing",
            "status": "running",
            "percent": 55,
            "detail": "Persisting session to PostgreSQL…",
        }
        artifacts = await asyncio.to_thread(
            load_session_artifacts,
            STORAGE_DIR,
            session_id,
            redis_meta,
            pipeline_state,
        )

        await _publish_stage_event(
            session_id,
            "indexing",
            "running",
            55,
            "Persisting session to PostgreSQL…",
        )
        await asyncio.to_thread(store.upsert_session_artifacts, artifacts)
        persisted_row = True

        await _publish_stage_event(
            session_id,
            "indexing",
            "running",
            82,
            "Updating transcript search index…",
        )
        search_result = await _update_search_index(artifacts)

        detail = "Session stored in PostgreSQL"
        if search_result["backend"] == "meilisearch":
            detail += f" and indexed in Meilisearch ({search_result['documents']} words)."
        else:
            detail += "; search indexing skipped."

        await asyncio.to_thread(
            store.update_session_state,
            session_id,
            stage="indexing",
            status="complete",
            percent=100,
            detail=detail,
        )

        await r.set(
            f"result:indexing:{session_id}",
            json.dumps({
                "status": "complete",
                "backend": search_result["backend"],
                "documents": search_result["documents"],
            }),
            ex=86400,
        )
        await _publish_stage_event(session_id, "indexing", "complete", 100, detail)
        log_event(
            logger,
            "stage.completed",
            session_id=session_id,
            stage="indexing",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            metadata={
                "search_backend": search_result["backend"],
                "documents": search_result["documents"],
            },
            message="Indexing stage complete",
        )

    except SessionArtifactsError as exc:
        detail = str(exc)
        if persisted_row:
            try:
                await asyncio.to_thread(
                    store.update_session_state,
                    session_id,
                    stage="indexing",
                    status="failed",
                    percent=0,
                    detail=detail,
                )
            except Exception:
                logger.exception("[%s] Failed to mark persisted row as failed.", session_id)
        await r.delete(lock_key)
        await _publish_stage_event(session_id, "indexing", "failed", 0, detail[:200])
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="indexing",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            message="Indexing aborted because artifacts were incomplete",
            level=logging.ERROR,
            exc_info=exc,
        )

    except Exception as exc:
        detail = str(exc) or "Unexpected indexing error"
        if persisted_row:
            try:
                await asyncio.to_thread(
                    store.update_session_state,
                    session_id,
                    stage="indexing",
                    status="failed",
                    percent=0,
                    detail=detail[:200],
                )
            except Exception:
                logger.exception("[%s] Failed to mark persisted row as failed.", session_id)
        await r.delete(lock_key)
        await _publish_stage_event(session_id, "indexing", "failed", 0, detail[:200])
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="indexing",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            message="Indexing stage failed",
            level=logging.ERROR,
            exc_info=exc,
        )


# ─── Metrics helpers (read result files when stages complete) ─────────────────

async def _read_asr_metrics(session_id: str) -> None:
    r = _get_redis()
    raw = await r.get(f"result:asr:{session_id}")
    if not raw:
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    conf = data.get("avg_confidence", 0)
    if conf:
        record_transcription_confidence(conf)

    record_transcription_rtf(
        float(data.get("transcription_time_s") or 0.0),
        float(data.get("audio_duration_s") or 0.0),
    )


async def _read_diarisation_metrics(session_id: str) -> None:
    r = _get_redis()
    raw = await r.get(f"result:diarisation:{session_id}")
    if not raw:
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    speaker_total = data.get("speaker_count", 0)
    if speaker_total:
        record_speaker_count(speaker_total)


async def _read_nlp_metrics(session_id: str) -> None:
    data = await _load_nlp_payload_for_metrics(session_id)
    if not isinstance(data, dict):
        return

    pain_points = data.get("pain_points", {})
    if isinstance(pain_points, dict):
        pain_point_items = pain_points.get("pain_points", [])
    elif isinstance(pain_points, list):
        pain_point_items = pain_points
    else:
        pain_point_items = []

    for pain_point in pain_point_items:
        severity = pain_point.get("severity", "low")
        record_pain_points_total(severity)
        record_pain_point_confidence(float(pain_point.get("confidence") or 0.0))

    action_items = data.get("action_items", {})
    if isinstance(action_items, dict):
        record_action_items_total(len(action_items.get("action_items", [])))

    sentiment = data.get("sentiment", {})
    if isinstance(sentiment, dict):
        for segment in sentiment.get("per_segment", []):
            scores = segment.get("scores", {})
            if not isinstance(scores, dict):
                continue
            for label, score in scores.items():
                record_sentiment_score(label, float(score))


# ─── SSE helpers ──────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


async def _progress_sse_generator(session_id: str) -> AsyncGenerator[str, None]:
    """
    Async generator for the progress SSE stream.

    On connect
    ──────────
    1. Send current session state (catch-up for page refreshes)
    2. Send all buffered transcript preview word batches accumulated so far
    3. If already complete, send pipeline_complete and return

    Live
    ────
    4. Subscribe to progress:{session_id}
    5. Forward every event (progress + transcript_preview) as SSE
    6. Send heartbeat every HEARTBEAT_INTERVAL seconds to keep connection alive
    7. On pipeline_complete (genai report done), close stream
    """
    r = _get_redis()

    # ── Catch-up: current stage state ────────────────────────────────────────
    current = await r.hgetall(f"session:{session_id}")
    if current:
        yield _sse({"type": "state_catchup", **current})

    # ── Catch-up: buffered transcript preview words ───────────────────────────
    preview_batches = await r.lrange(f"transcript:preview:{session_id}", 0, -1)
    if preview_batches:
        for batch_json in preview_batches:
            try:
                words = json.loads(batch_json)
                yield _sse({
                    "type":    "transcript_preview_catchup",
                    "words":   words,
                    "session_id": session_id,
                })
            except json.JSONDecodeError:
                pass

    # ── Check if pipeline is already complete ─────────────────────────────────
    # genai_report complete means everything is done
    indexing_result = await r.get(f"result:indexing:{session_id}")
    persisted = await _fetch_persisted_session(session_id)
    if indexing_result or (
        current.get("stage") == "indexing"
        and current.get("status") == "complete"
    ) or (
        persisted
        and persisted.get("stage") == "indexing"
        and persisted.get("status") == "complete"
    ):
        yield _sse({"type": "pipeline_complete", "session_id": session_id})
        return

    # ── Live subscription via asyncio.Queue ───────────────────────────────────
    # Using a queue lets us inject heartbeats alongside real events without
    # blocking the pub/sub listener on the SSE response loop.

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    pubsub = r.pubsub()
    await pubsub.subscribe(f"progress:{session_id}")

    async def _reader():
        """Reads from Redis pub/sub and puts raw JSON strings into the queue."""
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await queue.put(message["data"])
        except asyncio.CancelledError:
            pass
        finally:
            await queue.put(None)   # sentinel — signals the generator to stop

    reader_task = asyncio.create_task(_reader())

    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    queue.get(), timeout=float(HEARTBEAT_INTERVAL)
                )
            except asyncio.TimeoutError:
                yield _sse({"type": "heartbeat", "ts": int(time.time())})
                continue

            if raw is None:
                break   # pub/sub reader closed

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            yield _sse(event)

            # Close the stream when the full pipeline is done
            stage  = event.get("stage", "")
            status = event.get("status", "")
            if (stage == "indexing" and status in ("complete", "done")) or \
               event.get("type") == "pipeline_complete":
                yield _sse({"type": "pipeline_complete", "session_id": session_id})
                break

    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
        await pubsub.unsubscribe(f"progress:{session_id}")
        await _safe_close(pubsub)


# ─── Endpoints ────────────────────────────────────────────────────────────────

# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/ingest", status_code=202)
@app.post("/api/upload", status_code=202)   # alias — frontend Next.js posts here
async def ingest_video(
    request: Request,
    file: UploadFile = File(None),
    video: UploadFile = File(None),   # frontend VideoUpload uses field name "video"
):
    # Accept either field name
    upload = file or video
    if not upload:
        raise HTTPException(status_code=422, detail="No file provided. Use field name 'file' or 'video'.")
    file = upload  # normalise to 'file' for the rest of the function
    """
    Proxy video upload to the ingestion service.

    The gateway registers the session in Redis (for the library) and then
    returns the same response the ingestion service produced.
    The pipeline orchestrator will automatically trigger ASR once Stage 2
    publishes audio_extraction:complete.
    """
    ingestion_url = f"{INGESTION_URL}/ingest"

    # Stream the file to the ingestion service
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        content = await file.read()
        form.add_field(
            "file",
            content,
            filename=file.filename or "upload",
            content_type=file.content_type or "application/octet-stream",
        )

        try:
            async with session.post(
                ingestion_url, data=form, timeout=aiohttp.ClientTimeout(total=600)
            ) as resp:
                body = await resp.json()
                if resp.status not in (200, 202):
                    raise HTTPException(status_code=resp.status, detail=body)
        except aiohttp.ClientConnectorError:
            raise HTTPException(
                status_code=503,
                detail="Ingestion service unavailable. Ensure 'ingestion' container is running.",
            )

    session_id = body.get("session_id")
    if session_id:
        # Register session in library
        r = _get_redis()
        await r.hset(f"session:meta:{session_id}", mapping={
            "session_id":     session_id,
            "filename":       file.filename or "unknown",
            "created_at":     str(int(time.time())),
            "file_size_bytes": str(body.get("metadata", {}).get("file_size_bytes", 0)),
        })
        await r.expire(f"session:meta:{session_id}", 86400 * 30)   # 30 day TTL

        # Add to sorted set (score = timestamp for chronological ordering)
        await r.zadd("sessions:index", {session_id: int(time.time())})

        logger.info("[%s] Session registered. File: %s", session_id, file.filename)

    return JSONResponse(content=body, status_code=202)


# ── SSE Progress Stream ───────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/progress")
async def session_progress(session_id: str):
    """
    SSE stream for the processing pipeline progress.

    The browser connects here and keeps the connection open.
    Events arrive for every stage: ingestion → audio → ASR → diarisation
    → NLP → genai report.

    Event types:
      state_catchup              — current state snapshot on connect
      transcript_preview_catchup — buffered word batches on connect
      transcript_preview         — live word batches during ASR
      (progress event)           — stage/status/percent/detail
      heartbeat                  — every 15s to keep connection alive
      pipeline_complete          — all stages done, frontend can stop listening
    """
    return StreamingResponse(
        _progress_sse_generator(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── Session State ─────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Return current pipeline state for a session."""
    r    = _get_redis()
    data = await r.hgetall(f"session:{session_id}")
    meta = await r.hgetall(f"session:meta:{session_id}")
    if not data:
        persisted = await _fetch_persisted_session(session_id)
        if not persisted:
            raise HTTPException(status_code=404, detail="Session not found.")
        pipeline = _persisted_pipeline(persisted)
        return JSONResponse(content={
            "session_id": session_id,
            **pipeline,
            "meta": _persisted_summary(persisted),
        })

    if not meta:
        persisted = await _fetch_persisted_session(session_id)
        if persisted:
            meta = _persisted_summary(persisted)
    return JSONResponse(content={"session_id": session_id, **data, "meta": meta})


# ── Thumbnail ─────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/thumbnail")
async def get_thumbnail(session_id: str):
    """
    Proxy to ingestion service thumbnail endpoint.
    Returns the first-frame JPEG extracted from the uploaded video.
    """
    url = f"{INGESTION_URL}/thumbnail/{session_id}"
    async with aiohttp.ClientSession() as http:
        try:
            async with http.get(url) as resp:
                if resp.status == 404:
                    raise HTTPException(status_code=404, detail="Thumbnail not available.")
                if resp.status != 200:
                    raise HTTPException(status_code=resp.status, detail="Thumbnail error.")
                image_bytes = await resp.read()
        except aiohttp.ClientConnectorError:
            raise HTTPException(status_code=503, detail="Ingestion service unavailable.")

    from fastapi.responses import Response
    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=86400"},
    )


# ── Transcript ────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/transcript")
async def get_transcript(session_id: str):
    """
    Return the full word-level transcript once Stage 3 is complete.
    Used by the frontend to build the searchable transcript panel.
    """
    persisted = await _fetch_persisted_session(session_id)
    if persisted and persisted.get("transcript_json"):
        return JSONResponse(content=persisted["transcript_json"])

    path = STORAGE_DIR / session_id / "transcript_words.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Transcript not available yet. Ensure Stage 3 (ASR) has completed.",
        )
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/sessions/{session_id}/transcript/diarised")
async def get_diarised_transcript(session_id: str):
    """
    Return the speaker-labelled transcript once Stage 4 is complete.
    Used by the frontend for the colour-coded speaker transcript.
    """
    persisted = await _fetch_persisted_session(session_id)
    if persisted and persisted.get("diarised_transcript_json"):
        return JSONResponse(content=persisted["diarised_transcript_json"])

    path = STORAGE_DIR / session_id / "transcript_speaker.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Diarised transcript not available yet. Ensure Stage 4 has completed.",
        )
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/sessions/{session_id}/transcript/preview")
async def get_transcript_preview(session_id: str):
    """
    Return all word batches accumulated so far during ASR.
    Used for the live transcript preview while diarisation is still running.
    Merges all batches into a flat word list sorted by start time.
    """
    r       = _get_redis()
    batches = await r.lrange(f"transcript:preview:{session_id}", 0, -1)

    if not batches:
        return JSONResponse(content={"words": [], "batch_count": 0})

    words: list[dict] = []
    for batch_json in batches:
        try:
            words.extend(json.loads(batch_json))
        except json.JSONDecodeError:
            pass

    words.sort(key=lambda w: w.get("start", 0))
    return JSONResponse(content={"words": words, "batch_count": len(batches)})


@app.get("/api/sessions/{session_id}/search")
async def search_transcript(session_id: str, q: str, limit: int = 20):
    """
    Keyword search over indexed transcript words.
    Returns word-level hits with timestamps from Meilisearch when configured,
    otherwise falls back to PostgreSQL.
    """
    query = q.strip()
    if not query:
        return JSONResponse(content={"hits": [], "backend": SEARCH_BACKEND or "disabled"})

    started = time.perf_counter()
    hits, backend = await _search_transcript_index(session_id, query, limit=limit)
    latency_ms = int((time.perf_counter() - started) * 1000)
    increment_search_latency(latency_ms)
    log_event(
        logger,
        "search.query",
        session_id=session_id,
        stage="search",
        duration_ms=latency_ms,
        metadata={"backend": backend, "query": query, "hit_count": len(hits)},
        message="Transcript search complete",
    )
    return JSONResponse(content={"hits": hits, "backend": backend, "query": query})


# ── NLP Results ───────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/nlp")
async def get_nlp_results(session_id: str):
    """Return all NLP analysis results once Stage 5 is complete."""
    persisted = await _fetch_persisted_session(session_id)
    if persisted and persisted.get("nlp_json"):
        return JSONResponse(content=persisted["nlp_json"])

    path = STORAGE_DIR / session_id / "nlp_results.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="NLP results not available yet. Ensure Stage 5 has completed.",
        )
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


# ── GenAI Report ──────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/report/stream")
async def stream_report(session_id: str):
    """
    Proxy the GenAI report SSE stream from the genai service.

    The frontend connects here after receiving a 'genai_ready' progress event.
    Tokens stream token-by-token. Section headers fire immediately as the LLM
    produces them. The stream closes with a 'done' event.
    """
    genai_url = f"{GENAI_URL}/report/{session_id}/stream"

    async def _proxy_genai_stream() -> AsyncGenerator[str, None]:
        # Record when first token arrives (for genai_report_first_token_ms metric)
        first_token_recorded = False
        stream_start = time.time()
        progress_started = False
        stream_finished = False
        request_recorded = False
        last_progress_percent = 10
        r = _get_redis()

        def _record_genai_request_once(status: str) -> None:
            nonlocal request_recorded
            if request_recorded:
                return
            record_genai_request(status)
            request_recorded = True

        try:
            timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=180)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                running_event = {
                    "session_id": session_id,
                    "stage": "genai_report",
                    "status": "running",
                    "percent": 10,
                    "detail": "Generating AI report…",
                }
                await r.hset(f"session:{session_id}", mapping={
                    "stage": "genai_report",
                    "status": "running",
                    "percent": "10",
                    "detail": "Generating AI report…",
                })
                await r.expire(f"session:{session_id}", 86400)
                await r.publish(f"progress:{session_id}", json.dumps(running_event))
                progress_started = True

                async with http.get(genai_url) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        failed_event = {
                            "session_id": session_id,
                            "stage": "genai_report",
                            "status": "failed",
                            "percent": 0,
                            "detail": body[:200] or "GenAI service error",
                        }
                        await r.hset(f"session:{session_id}", mapping={
                            "stage": "genai_report",
                            "status": "failed",
                            "percent": "0",
                            "detail": body[:200] or "GenAI service error",
                        })
                        await r.expire(f"session:{session_id}", 86400)
                        await r.publish(f"progress:{session_id}", json.dumps(failed_event))
                        _record_genai_request_once("error")
                        yield f"data: {json.dumps({'type': 'error', 'message': body[:200]})}\n\n"
                        return

                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8")
                        if not line.strip():
                            yield "\n"   # pass through SSE blank lines
                            continue

                        yield line

                        if line.startswith("data: "):
                            try:
                                payload = json.loads(line[6:].strip())
                            except json.JSONDecodeError:
                                payload = {}

                            if payload.get("type") == "token":
                                last_progress_percent = max(last_progress_percent, 55)
                                running_update = {
                                    "session_id": session_id,
                                    "stage": "genai_report",
                                    "status": "running",
                                    "percent": last_progress_percent,
                                    "detail": "Streaming AI report…",
                                }
                                await r.hset(f"session:{session_id}", mapping={
                                    "stage": "genai_report",
                                    "status": "running",
                                    "percent": str(last_progress_percent),
                                    "detail": "Streaming AI report…",
                                })
                                await r.expire(f"session:{session_id}", 86400)
                                await r.publish(f"progress:{session_id}", json.dumps(running_update))

                            elif payload.get("type") == "section":
                                section_idx = int(payload.get("index", -1))
                                progress = max(
                                    last_progress_percent,
                                    min(95, 55 + int(((section_idx + 1) / 7) * 40)),
                                )
                                last_progress_percent = progress
                                title = payload.get("title", f"Section {section_idx + 1}")
                                running_update = {
                                    "session_id": session_id,
                                    "stage": "genai_report",
                                    "status": "running",
                                    "percent": progress,
                                    "detail": f"Drafting {title}…",
                                }
                                await r.hset(f"session:{session_id}", mapping={
                                    "stage": "genai_report",
                                    "status": "running",
                                    "percent": str(progress),
                                    "detail": f"Drafting {title}…",
                                })
                                await r.expire(f"session:{session_id}", 86400)
                                await r.publish(f"progress:{session_id}", json.dumps(running_update))

                            elif payload.get("type") == "error":
                                message = payload.get("message", "GenAI stream failed")
                                failed_event = {
                                    "session_id": session_id,
                                    "stage": "genai_report",
                                    "status": "failed",
                                    "percent": last_progress_percent if first_token_recorded else 10,
                                    "detail": message[:200],
                                }
                                await r.hset(f"session:{session_id}", mapping={
                                    "stage": "genai_report",
                                    "status": "failed",
                                    "percent": str(last_progress_percent if first_token_recorded else 10),
                                    "detail": message[:200],
                                })
                                await r.expire(f"session:{session_id}", 86400)
                                await r.publish(f"progress:{session_id}", json.dumps(failed_event))
                                _record_genai_request_once("error")

                            elif payload.get("type") == "done":
                                stream_finished = True
                                await r.set(
                                    f"result:genai:{session_id}",
                                    json.dumps({
                                        "status": "complete",
                                        "report_path": payload.get("report_path"),
                                        "json_path": payload.get("json_path"),
                                    }),
                                    ex=86400,
                                )
                                _record_genai_request_once("success")
                                await _publish_stage_event(
                                    session_id,
                                    "genai_report",
                                    "complete",
                                    100,
                                    "AI report generated",
                                )
                                asyncio.create_task(_run_indexing_stage(session_id))

                        # Record first-token latency
                        if not first_token_recorded and '"type": "token"' in line:
                            latency_ms = int((time.time() - stream_start) * 1000)
                            record_genai_first_token(latency_ms)
                            first_token_recorded = True

                    if progress_started and not stream_finished:
                        failed_event = {
                            "session_id": session_id,
                            "stage": "genai_report",
                            "status": "failed",
                            "percent": last_progress_percent if first_token_recorded else 10,
                            "detail": "GenAI stream ended before completion.",
                        }
                        await r.hset(f"session:{session_id}", mapping={
                            "stage": "genai_report",
                            "status": "failed",
                            "percent": str(last_progress_percent if first_token_recorded else 10),
                            "detail": "GenAI stream ended before completion.",
                        })
                        await r.expire(f"session:{session_id}", 86400)
                        await r.publish(f"progress:{session_id}", json.dumps(failed_event))
                        _record_genai_request_once("error")
                        yield f"data: {json.dumps({'type': 'error', 'message': 'GenAI stream ended before completion.'})}\n\n"
                        return

        except aiohttp.ClientConnectorError:
            failed_event = {
                "session_id": session_id,
                "stage": "genai_report",
                "status": "failed",
                "percent": 0,
                "detail": "GenAI service unavailable.",
            }
            if progress_started:
                await r.hset(f"session:{session_id}", mapping={
                    "stage": "genai_report",
                    "status": "failed",
                    "percent": "0",
                    "detail": "GenAI service unavailable.",
                })
                await r.expire(f"session:{session_id}", 86400)
                await r.publish(f"progress:{session_id}", json.dumps(failed_event))
            _record_genai_request_once("error")
            yield f"data: {json.dumps({'type': 'error', 'message': 'GenAI service unavailable.'})}\n\n"

    return StreamingResponse(
        _proxy_genai_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.get("/api/sessions/{session_id}/report")
async def get_report(session_id: str):
    """Return the completed GenAI report JSON."""
    persisted = await _fetch_persisted_session(session_id)
    if persisted and persisted.get("report_json"):
        return JSONResponse(content=persisted["report_json"])

    path = STORAGE_DIR / session_id / "genai_report.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not available yet. Stream /report/stream first.",
        )
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


# ── Session Library ───────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions(
    limit:  int = 20,
    offset: int = 0,
):
    """
    Return paginated list of all processed sessions, newest first.
    Used by the Meeting Library screen.
    """
    r = _get_redis()
    persisted_rows = await _list_persisted_sessions()
    persisted_map = {row["session_id"]: row for row in persisted_rows}

    session_ids = await r.zrevrange("sessions:index", 0, -1)
    merged: list[dict] = []
    seen: set[str] = set()

    for sid in session_ids:
        meta = await r.hgetall(f"session:meta:{sid}")
        state = await r.hgetall(f"session:{sid}")
        persisted = persisted_map.get(sid)
        entry = _persisted_summary(persisted) if persisted else {"session_id": sid}

        if meta:
            entry.update(meta)
        if persisted and "duration" not in entry:
            entry["duration"] = int(persisted.get("duration_seconds") or 0)

        entry["stage"] = state.get("stage", entry.get("stage", "unknown"))
        entry["status"] = state.get("status", entry.get("status", "unknown"))

        nlp_payload = None
        nlp_raw = await r.get(f"result:nlp:{sid}")
        if nlp_raw:
            try:
                nlp_payload = json.loads(nlp_raw)
            except json.JSONDecodeError:
                nlp_payload = None
        elif persisted:
            nlp_payload = persisted.get("nlp_json")
        _apply_nlp_counts(entry, nlp_payload)
        _apply_report_counts(entry, persisted.get("report_json") if persisted else None)

        merged.append(entry)
        seen.add(sid)

    for row in persisted_rows:
        sid = row["session_id"]
        if sid in seen:
            continue
        merged.append(_persisted_summary(row))

    merged.sort(key=lambda item: int(item.get("created_at") or 0), reverse=True)
    total = len(merged)
    paged = merged[offset: offset + limit]
    return JSONResponse(content={
        "sessions": paged,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """
    Return full detail for one session: metadata, pipeline state,
    NLP summary, and available result files.
    """
    r = _get_redis()
    persisted = await _fetch_persisted_session(session_id)

    meta = await r.hgetall(f"session:meta:{session_id}")
    state = await r.hgetall(f"session:{session_id}")
    nlp_raw = await r.get(f"result:nlp:{session_id}")
    asr_raw = await r.get(f"result:asr:{session_id}")
    dia_raw = await r.get(f"result:diarisation:{session_id}")

    if not meta and not persisted:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not meta and persisted:
        meta = _persisted_summary(persisted)

    pipeline = state or (_persisted_pipeline(persisted) if persisted else {})

    asr_result = json.loads(asr_raw) if asr_raw else None
    if not asr_result and persisted:
        asr_result = {
            "word_count": persisted.get("word_count", 0),
            "avg_confidence": persisted.get("avg_confidence", 0),
        }

    diar_result = json.loads(dia_raw) if dia_raw else None
    if not diar_result and persisted and persisted.get("diarised_transcript_json"):
        diar_result = {
            "speaker_count": persisted.get("speaker_count", 0),
            "segment_count": len(
                (persisted.get("diarised_transcript_json") or {}).get("segments", [])
            ),
        }

    if nlp_raw:
        nlp_result = json.loads(nlp_raw)
    elif persisted:
        nlp_result = persisted.get("nlp_json")
    else:
        nlp_result = None

    return JSONResponse(content={
        "session_id": session_id,
        "meta": meta,
        "pipeline": pipeline,
        "asr_result": asr_result,
        "diar_result": diar_result,
        "nlp_result": nlp_result,
    })


@app.delete("/api/sessions/clear")
async def clear_all_sessions():
    """
    Clear all stored meeting sessions from the database.
    This removes all meeting data permanently.
    """
    store = _get_session_store()
    try:
        store.clear_all_sessions()
        logger.info("All meeting sessions cleared from database.")
        return JSONResponse(content={"message": "All meeting sessions cleared successfully."})
    except Exception as exc:
        logger.error("Failed to clear sessions: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to clear sessions: {exc}")


# ── Retry a failed stage ──────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/retry/{stage}")
async def retry_stage(session_id: str, stage: str):
    """
    Retry a specific pipeline stage that has failed.
    The UI shows a Retry button when a stage status = 'failed'.
    """
    r = _get_redis()

    RETRYABLE = {
        "asr":            ("pipeline:triggered:{sid}:asr",
                           "asr_service.transcribe_audio", "asr"),
        "nlp":            ("pipeline:triggered:{sid}:nlp",
                           "nlp_service.analyse",          "nlp"),
    }

    if stage not in RETRYABLE:
        raise HTTPException(
            status_code=400,
            detail=f"Stage '{stage}' is not retryable via the gateway. "
                   f"Retryable stages: {list(RETRYABLE.keys())}",
        )

    lock_tpl, task_name, queue = RETRYABLE[stage]
    lock_key = lock_tpl.format(sid=session_id)

    # Delete the idempotency lock so the orchestrator (or direct dispatch) can re-run
    await r.delete(lock_key)

    try:
        _celery.send_task(
            task_name,
            kwargs={"session_id": session_id},
            queue=queue,
        )
        logger.info("[%s] Stage '%s' retried manually.", session_id, stage)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to dispatch retry: {exc}")

    return JSONResponse(content={"session_id": session_id, "stage": stage,
                                 "status": "retry_dispatched"})


# ── Prometheus Metrics ────────────────────────────────────────────────────────

@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus scrape endpoint.
    Add this URL to prometheus.yml as a scrape target.
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    await _refresh_runtime_metrics()
    return StreamingResponse(
        iter([generate_latest(REGISTRY)]),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/video")
async def stream_video(session_id: str, request: Request):
    """
    Stream the raw uploaded video file.
    Supports HTTP Range headers for video seeking in the browser <video> element.
    """
    video_dir = STORAGE_DIR / session_id
    # Find any raw video file (raw.mp4, raw.webm, etc.)
    candidates = list(video_dir.glob("raw.*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="Video file not found.")

    video_path = candidates[0]
    file_size  = video_path.stat().st_size

    # Determine MIME type from extension
    ext_mime = {
        ".mp4": "video/mp4", ".webm": "video/webm",
        ".mov": "video/quicktime", ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    mime = ext_mime.get(video_path.suffix.lower(), "video/mp4")

    range_header = request.headers.get("range")
    if range_header:
        # Parse Range: bytes=start-end
        try:
            range_val  = range_header.replace("bytes=", "")
            start_str, end_str = range_val.split("-")
            start = int(start_str)
            end   = int(end_str) if end_str else file_size - 1
            end   = min(end, file_size - 1)
            length = end - start + 1

            def _iter_range():
                with open(video_path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                _iter_range(),
                status_code=206,
                media_type=mime,
                headers={
                    "Content-Range":  f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges":  "bytes",
                    "Content-Length": str(length),
                },
            )
        except (ValueError, IndexError):
            pass  # fall through to full file

    # Full file
    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(video_path),
        media_type=mime,
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/sessions/{session_id}/audio")
async def stream_audio(session_id: str):
    """Stream the denoised WAV file for Wavesurfer.js waveform rendering."""
    audio_path = STORAGE_DIR / session_id / "audio_denoised.wav"
    if not audio_path.exists():
        audio_path = STORAGE_DIR / session_id / "audio_raw.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    from fastapi.responses import FileResponse
    return FileResponse(path=str(audio_path), media_type="audio/wav")


@app.get("/health")
async def health():
    r = _get_redis()
    store = _get_session_store()
    redis_ok = False
    postgres_ok = store is None
    search_ok = SEARCH_BACKEND in ("", "none", "disabled")

    try:
        await r.ping()
        redis_ok = True
    except Exception:
        pass

    if store is not None:
        postgres_ok = await asyncio.to_thread(store.ping)

    if SEARCH_BACKEND == "meilisearch":
        try:
            timeout = aiohttp.ClientTimeout(total=5, connect=3, sock_read=3)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.get(f"{SEARCH_URL}/health", headers=_meili_headers()) as resp:
                    search_ok = resp.status == 200
        except Exception:
            search_ok = False

    overall_ok = redis_ok and postgres_ok and search_ok

    return JSONResponse(content={
        "status":    "ok" if overall_ok else "degraded",
        "redis":     "ok" if redis_ok else "unavailable",
        "postgres":  "ok" if postgres_ok else "unavailable",
        "search":    "ok" if search_ok else ("disabled" if SEARCH_BACKEND in ("", "none", "disabled") else "unavailable"),
        "ingestion": INGESTION_URL,
        "genai":     GENAI_URL,
    })
