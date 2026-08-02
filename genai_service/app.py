"""
FastAPI application for Stage 6 GenAI meeting reports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from observability import configure_json_logging, log_event
from report_builder import ContextAssemblyError, ReportBuilder
from streamer import LLMBackend, ReportStreamer

configure_json_logging("genai_service")
logger = logging.getLogger(__name__)
REPORT_VERSION = "2026-04-09-v7"

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/tmp/meeting_insights"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

app = FastAPI(title="Meeting Insights - GenAI Report Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_active_generations: dict[str, asyncio.Event] = {}


def _session_dir(session_id: str) -> Path:
    directory = STORAGE_DIR / session_id
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return directory


def _get_streamer() -> ReportStreamer:
    backend = LLMBackend.OPENAI if LLM_BACKEND == "openai" else LLMBackend.OLLAMA
    return ReportStreamer(
        backend=backend,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        openai_api_key=OPENAI_KEY,
        openai_model=OPENAI_MODEL,
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _cache_is_current(json_path: Path) -> bool:
    if not json_path.exists():
        return False

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return payload.get("report_version") == REPORT_VERSION


async def _sse_generator(session_id: str, session_dir: Path):
    existing_md = session_dir / "genai_report.md"
    existing_json = session_dir / "genai_report.json"

    try:
        while True:
            if existing_md.exists() and existing_json.exists() and _cache_is_current(existing_json):
                log_event(
                    logger,
                    "report.cached",
                    session_id=session_id,
                    stage="genai_report",
                    metadata={
                        "report_path": str(existing_md),
                        "json_path": str(existing_json),
                        "backend": LLM_BACKEND,
                    },
                    message="Returning cached GenAI report",
                )
                yield _sse({
                    "type": "done",
                    "report_path": str(existing_md),
                    "json_path": str(existing_json),
                })
                return

            active = _active_generations.get(session_id)
            if active is None:
                break

            log_event(
                logger,
                "report.waiting_for_existing_generation",
                session_id=session_id,
                stage="genai_report",
                metadata={"backend": LLM_BACKEND},
                message="Waiting for an existing report generation to finish",
            )
            while not active.is_set():
                yield ": waiting for existing report generation\n\n"
                await asyncio.sleep(2)
                if existing_md.exists() and existing_json.exists():
                    yield _sse({
                        "type": "done",
                        "report_path": str(existing_md),
                        "json_path": str(existing_json),
                    })
                    return
            continue

        generation_done = asyncio.Event()
        _active_generations[session_id] = generation_done
        stage_started = time.perf_counter()
        log_event(
            logger,
            "stage.started",
            session_id=session_id,
            stage="genai_report",
            metadata={"backend": LLM_BACKEND},
            message="Starting GenAI report generation",
        )

        builder = ReportBuilder(session_dir)
        try:
            context = builder.assemble()
        except ContextAssemblyError as exc:
            log_event(
                logger,
                "stage.failed",
                session_id=session_id,
                stage="genai_report",
                duration_ms=(time.perf_counter() - stage_started) * 1000,
                metadata={"backend": LLM_BACKEND},
                message="Failed to assemble report context",
                level=logging.ERROR,
                exc_info=exc,
            )
            yield _sse({"type": "error", "message": str(exc)})
            return

        prompt = builder.build_prompt(context)
        streamer = _get_streamer()
        report_text = ""
        current_section_idx = -1

        async for event in streamer.stream(prompt):
            if event["type"] == "token":
                token = event["content"]
                report_text += token
                section_event = _detect_section(report_text, current_section_idx)
                if section_event:
                    current_section_idx = section_event["index"]
                    yield _sse({"type": "section", **section_event})
                yield _sse({"type": "token", "content": token})
                continue

            if event["type"] == "error":
                log_event(
                    logger,
                    "stage.failed",
                    session_id=session_id,
                    stage="genai_report",
                    duration_ms=(time.perf_counter() - stage_started) * 1000,
                    metadata={"backend": LLM_BACKEND},
                    message="GenAI backend returned an error",
                    level=logging.ERROR,
                )
                yield _sse({"type": "error", "message": event["message"]})
                return

        sections = builder.merge_sections(
            builder.parse_sections(report_text),
            builder.build_structured_sections(context),
        )
        final_report_text = builder.render_sections(sections)

        report_path = session_dir / "genai_report.md"
        report_path.write_text(final_report_text, encoding="utf-8")

        json_path = session_dir / "genai_report.json"
        json_path.write_text(
            json.dumps({
                "session_id": session_id,
                "report_version": REPORT_VERSION,
                "sections": sections,
            }, indent=2),
            encoding="utf-8",
        )

        log_event(
            logger,
            "stage.completed",
            session_id=session_id,
            stage="genai_report",
            duration_ms=(time.perf_counter() - stage_started) * 1000,
            metadata={
                "backend": LLM_BACKEND,
                "section_count": len(sections),
                "report_path": str(report_path),
                "json_path": str(json_path),
            },
            message="GenAI report generation complete",
        )
        yield _sse({
            "type": "done",
            "report_path": str(report_path),
            "json_path": str(json_path),
        })

    except Exception as exc:
        log_event(
            logger,
            "stage.failed",
            session_id=session_id,
            stage="genai_report",
            metadata={"backend": LLM_BACKEND},
            message="Unexpected GenAI report failure",
            level=logging.ERROR,
            exc_info=exc,
        )
        yield _sse({"type": "error", "message": str(exc)})
    finally:
        active = _active_generations.get(session_id)
        if active is not None:
            active.set()
            if _active_generations.get(session_id) is active:
                _active_generations.pop(session_id, None)


def _detect_section(buffer: str, current_idx: int) -> dict | None:
    heading_re = re.compile(r"(?m)^[ \t]{0,3}#{2,6}[ \t]+(.+?)\s*$")

    for match in heading_re.finditer(buffer):
        normalized = ReportBuilder._normalize_heading(match.group(1))
        if not normalized:
            continue

        for section in ReportBuilder.SECTION_DEFS:
            if section["index"] <= current_idx:
                continue
            if normalized in section["aliases"]:
                return {
                    "index": section["index"],
                    "title": section["title"],
                    "header": section["header"],
                }
    return None


@app.get("/report/{session_id}/stream")
async def stream_report(session_id: str):
    session_dir = _session_dir(session_id)
    return StreamingResponse(
        _sse_generator(session_id, session_dir),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/report/{session_id}/result")
async def get_report(session_id: str):
    session_dir = _session_dir(session_id)
    json_path = session_dir / "genai_report.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not yet generated. Call /stream first.",
        )
    return JSONResponse(content=json.loads(json_path.read_text()))


@app.get("/health")
async def health():
    return {"status": "ok", "backend": LLM_BACKEND}
