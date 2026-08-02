from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


def _count_report_lines(report: dict[str, Any], section_index: int) -> int:
    sections = report.get("sections", [])
    if not isinstance(sections, list):
        return 0
    target = next((section for section in sections if int(section.get("index", -1)) == section_index), None)
    if not isinstance(target, dict):
        return 0
    content = str(target.get("content") or "")
    if not content.strip():
        return 0
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    table_rows = [line for line in lines if "|" in line]
    if len(table_rows) >= 3:
        return max(0, len(table_rows) - 2)
    bullet_rows = [line for line in lines if line.startswith(("-", "*", "+"))]
    return len(bullet_rows)


class SessionArtifactsError(RuntimeError):
    """Raised when the session payload required for persistence is incomplete."""


@dataclass(slots=True)
class SessionArtifacts:
    session_id: str
    redis_meta: dict[str, Any]
    pipeline_state: dict[str, Any]
    metadata: dict[str, Any]
    transcript: dict[str, Any]
    diarised_transcript: dict[str, Any]
    nlp_results: dict[str, Any]
    report: dict[str, Any]
    report_markdown: str

    @property
    def report_sections(self) -> list[dict[str, Any]]:
        sections = self.report.get("sections", [])
        return sections if isinstance(sections, list) else []

    @property
    def filename(self) -> str:
        return (
            self.redis_meta.get("filename")
            or self.metadata.get("original_filename")
            or self.metadata.get("filename")
            or f"{self.session_id}.webm"
        )

    @property
    def created_at(self) -> int:
        raw = self.redis_meta.get("created_at") or self.metadata.get("created_at")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(time.time())

    @property
    def file_size_bytes(self) -> int:
        raw = self.redis_meta.get("file_size_bytes") or self.metadata.get("file_size_bytes") or 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @property
    def duration_seconds(self) -> float:
        raw = self.metadata.get("duration") or self.redis_meta.get("duration") or 0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    @property
    def speaker_count(self) -> int:
        raw = self.diarised_transcript.get("speaker_count", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @property
    def transcript_words(self) -> list[dict[str, Any]]:
        words = self.transcript.get("words", [])
        return words if isinstance(words, list) else []

    @property
    def word_count(self) -> int:
        return len(self.transcript_words)

    @property
    def avg_confidence(self) -> float:
        raw = self.transcript.get("avg_confidence", 0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    @property
    def pain_point_count(self) -> int:
        pain_points = self.nlp_results.get("pain_points", {})
        if isinstance(pain_points, dict):
            items = pain_points.get("pain_points", [])
            count = len(items) if isinstance(items, list) else 0
            return count or _count_report_lines(self.report, 2)
        if isinstance(pain_points, list):
            return len(pain_points)
        return _count_report_lines(self.report, 2)

    @property
    def action_item_count(self) -> int:
        action_items = self.nlp_results.get("action_items", {})
        if isinstance(action_items, dict):
            items = action_items.get("action_items", [])
            count = len(items) if isinstance(items, list) else 0
            return count or _count_report_lines(self.report, 3)
        if isinstance(action_items, list):
            return len(action_items)
        return _count_report_lines(self.report, 3)

    def to_meili_documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for idx, word in enumerate(self.transcript_words):
            token = (word.get("word") or "").strip()
            if not token:
                continue
            docs.append({
                "id": f"{self.session_id.replace('-', '_')}_{idx}",
                "session_id": self.session_id,
                "word_index": idx,
                "word": token,
                "start_time": float(word.get("start", 0) or 0),
                "end_time": float(word.get("end", 0) or 0),
                "confidence": float(word.get("confidence", 0) or 0),
                "speaker": word.get("speaker"),
                "segment_id": word.get("segment_id"),
                "chunk_idx": word.get("chunk_idx"),
            })
        return docs


def _load_json(path: Path, *, required: bool, default: Any = None) -> Any:
    if not path.exists():
        if required:
            raise SessionArtifactsError(f"Required artifact missing: {path.name}")
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionArtifactsError(f"Invalid JSON in {path.name}: {exc}") from exc


def load_session_artifacts(
    storage_dir: Path,
    session_id: str,
    redis_meta: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> SessionArtifacts:
    session_dir = storage_dir / session_id
    if not session_dir.exists():
        raise SessionArtifactsError(f"Session directory not found for {session_id}")

    metadata = _load_json(session_dir / "metadata.json", required=False, default={}) or {}
    transcript = _load_json(session_dir / "transcript_words.json", required=True)
    diarised = _load_json(session_dir / "transcript_speaker.json", required=True)
    nlp = _load_json(session_dir / "nlp_results.json", required=True)
    report = _load_json(session_dir / "genai_report.json", required=True)

    report_path = session_dir / "genai_report.md"
    if not report_path.exists():
        raise SessionArtifactsError("Required artifact missing: genai_report.md")

    return SessionArtifacts(
        session_id=session_id,
        redis_meta=redis_meta,
        pipeline_state=pipeline_state,
        metadata=metadata,
        transcript=transcript,
        diarised_transcript=diarised,
        nlp_results=nlp,
        report=report,
        report_markdown=report_path.read_text(encoding="utf-8"),
    )


class SessionStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS meeting_sessions (
                        session_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        created_at BIGINT NOT NULL,
                        file_size_bytes BIGINT NOT NULL DEFAULT 0,
                        duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
                        speaker_count INTEGER NOT NULL DEFAULT 0,
                        word_count INTEGER NOT NULL DEFAULT 0,
                        avg_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                        stage TEXT NOT NULL DEFAULT 'indexing',
                        status TEXT NOT NULL DEFAULT 'complete',
                        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        pipeline_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        transcript_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        diarised_transcript_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        nlp_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        report_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        report_markdown TEXT NOT NULL DEFAULT '',
                        pain_point_count INTEGER NOT NULL DEFAULT 0,
                        action_item_count INTEGER NOT NULL DEFAULT 0,
                        indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS meeting_report_sections (
                        session_id TEXT NOT NULL REFERENCES meeting_sessions(session_id) ON DELETE CASCADE,
                        section_index INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        header TEXT NOT NULL,
                        content TEXT NOT NULL,
                        PRIMARY KEY (session_id, section_index)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS meeting_transcript_words (
                        session_id TEXT NOT NULL REFERENCES meeting_sessions(session_id) ON DELETE CASCADE,
                        word_index INTEGER NOT NULL,
                        word TEXT NOT NULL,
                        start_time DOUBLE PRECISION NOT NULL DEFAULT 0,
                        end_time DOUBLE PRECISION NOT NULL DEFAULT 0,
                        confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                        speaker TEXT,
                        segment_id INTEGER,
                        chunk_idx INTEGER,
                        PRIMARY KEY (session_id, word_index)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS meeting_sessions_created_at_idx
                    ON meeting_sessions (created_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS meeting_transcript_words_session_start_idx
                    ON meeting_transcript_words (session_id, start_time)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS meeting_transcript_words_lookup_idx
                    ON meeting_transcript_words (session_id, lower(word))
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS meeting_report_sections_session_idx
                    ON meeting_report_sections (session_id, section_index)
                    """
                )
            conn.commit()

    def upsert_session_artifacts(self, artifacts: SessionArtifacts) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO meeting_sessions (
                        session_id,
                        filename,
                        created_at,
                        file_size_bytes,
                        duration_seconds,
                        speaker_count,
                        word_count,
                        avg_confidence,
                        stage,
                        status,
                        metadata_json,
                        pipeline_json,
                        transcript_json,
                        diarised_transcript_json,
                        nlp_json,
                        report_json,
                        report_markdown,
                        pain_point_count,
                        action_item_count,
                        indexed_at,
                        updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                    )
                    ON CONFLICT (session_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        created_at = EXCLUDED.created_at,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        duration_seconds = EXCLUDED.duration_seconds,
                        speaker_count = EXCLUDED.speaker_count,
                        word_count = EXCLUDED.word_count,
                        avg_confidence = EXCLUDED.avg_confidence,
                        stage = EXCLUDED.stage,
                        status = EXCLUDED.status,
                        metadata_json = EXCLUDED.metadata_json,
                        pipeline_json = EXCLUDED.pipeline_json,
                        transcript_json = EXCLUDED.transcript_json,
                        diarised_transcript_json = EXCLUDED.diarised_transcript_json,
                        nlp_json = EXCLUDED.nlp_json,
                        report_json = EXCLUDED.report_json,
                        report_markdown = EXCLUDED.report_markdown,
                        pain_point_count = EXCLUDED.pain_point_count,
                        action_item_count = EXCLUDED.action_item_count,
                        indexed_at = NOW(),
                        updated_at = NOW()
                    """,
                    (
                        artifacts.session_id,
                        artifacts.filename,
                        artifacts.created_at,
                        artifacts.file_size_bytes,
                        artifacts.duration_seconds,
                        artifacts.speaker_count,
                        artifacts.word_count,
                        artifacts.avg_confidence,
                        artifacts.pipeline_state.get("stage", "indexing"),
                        artifacts.pipeline_state.get("status", "complete"),
                        Jsonb(artifacts.metadata),
                        Jsonb(artifacts.pipeline_state),
                        Jsonb(artifacts.transcript),
                        Jsonb(artifacts.diarised_transcript),
                        Jsonb(artifacts.nlp_results),
                        Jsonb(artifacts.report),
                        artifacts.report_markdown,
                        artifacts.pain_point_count,
                        artifacts.action_item_count,
                    ),
                )

                cur.execute(
                    "DELETE FROM meeting_report_sections WHERE session_id = %s",
                    (artifacts.session_id,),
                )
                if artifacts.report_sections:
                    cur.executemany(
                        """
                        INSERT INTO meeting_report_sections (
                            session_id, section_index, title, header, content
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                artifacts.session_id,
                                int(section.get("index", idx)),
                                section.get("title") or f"Section {idx + 1}",
                                section.get("header") or "",
                                section.get("content") or "",
                            )
                            for idx, section in enumerate(artifacts.report_sections)
                        ],
                    )

                cur.execute(
                    "DELETE FROM meeting_transcript_words WHERE session_id = %s",
                    (artifacts.session_id,),
                )
                if artifacts.transcript_words:
                    cur.executemany(
                        """
                        INSERT INTO meeting_transcript_words (
                            session_id,
                            word_index,
                            word,
                            start_time,
                            end_time,
                            confidence,
                            speaker,
                            segment_id,
                            chunk_idx
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                artifacts.session_id,
                                idx,
                                (word.get("word") or "").strip(),
                                float(word.get("start", 0) or 0),
                                float(word.get("end", 0) or 0),
                                float(word.get("confidence", 0) or 0),
                                None if word.get("speaker") is None else str(word.get("speaker")),
                                None if word.get("segment_id") is None else int(word.get("segment_id")),
                                None if word.get("chunk_idx") is None else int(word.get("chunk_idx")),
                            )
                            for idx, word in enumerate(artifacts.transcript_words)
                            if (word.get("word") or "").strip()
                        ],
                    )
            conn.commit()

    def update_session_state(
        self,
        session_id: str,
        *,
        stage: str,
        status: str,
        percent: int,
        detail: str,
    ) -> None:
        pipeline_json = {
            "stage": stage,
            "status": status,
            "percent": percent,
            "detail": detail,
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meeting_sessions
                    SET
                        stage = %s,
                        status = %s,
                        pipeline_json = %s,
                        updated_at = NOW()
                    WHERE session_id = %s
                    """,
                    (
                        stage,
                        status,
                        Jsonb(pipeline_json),
                        session_id,
                    ),
                )
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        session_id,
                        filename,
                        created_at,
                        file_size_bytes,
                        duration_seconds,
                        speaker_count,
                        word_count,
                        avg_confidence,
                        stage,
                        status,
                        metadata_json,
                        pipeline_json,
                        transcript_json,
                        diarised_transcript_json,
                        nlp_json,
                        report_json,
                        report_markdown,
                        pain_point_count,
                        action_item_count,
                        indexed_at,
                        updated_at
                    FROM meeting_sessions
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                return cur.fetchone()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        session_id,
                        filename,
                        created_at,
                        file_size_bytes,
                        duration_seconds,
                        speaker_count,
                        word_count,
                        avg_confidence,
                        stage,
                        status,
                        report_json,
                        pain_point_count,
                        action_item_count
                    FROM meeting_sessions
                    ORDER BY created_at DESC
                    """
                )
                return list(cur.fetchall())

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception as exc:
            logger.warning("PostgreSQL ping failed: %s", exc)
            return False

    def search_transcript_words(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        session_id,
                        word_index,
                        word,
                        start_time,
                        end_time,
                        confidence,
                        speaker,
                        segment_id,
                        chunk_idx
                    FROM meeting_transcript_words
                    WHERE session_id = %s
                      AND lower(word) LIKE lower(%s)
                    ORDER BY start_time ASC
                    LIMIT %s
                    """,
                    (session_id, f"%{query.strip()}%", limit),
                )
                return list(cur.fetchall())

    def clear_all_sessions(self) -> None:
        """Delete all meeting sessions and related data from the database."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Due to CASCADE constraints, deleting from meeting_sessions will delete related records
                cur.execute("DELETE FROM meeting_sessions")
            conn.commit()
