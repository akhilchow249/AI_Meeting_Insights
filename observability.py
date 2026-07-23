from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": getattr(record, "service", self.service_name),
            "session_id": getattr(record, "session_id", None),
            "stage": getattr(record, "stage", None),
            "level": record.levelname.lower(),
            "event": getattr(record, "event", "log"),
            "duration_ms": getattr(record, "duration_ms", None),
            "metadata": getattr(record, "metadata", {}) or {},
            "message": message,
            "logger": record.name,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_json_logging(service_name: str, *, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter(service_name))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    session_id: str | None = None,
    stage: str | None = None,
    level: int = logging.INFO,
    duration_ms: int | float | None = None,
    metadata: dict[str, Any] | None = None,
    message: str | None = None,
    exc_info: bool | BaseException | tuple | None = None,
) -> None:
    logger.log(
        level,
        message or event,
        extra={
            "event": event,
            "session_id": session_id,
            "stage": stage,
            "duration_ms": None if duration_ms is None else int(duration_ms),
            "metadata": metadata or {},
        },
        exc_info=exc_info,
    )
