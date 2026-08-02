"""
Celery application for Stage 5 NLP analysis.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import redis
from celery import Celery, chord, group

from action_items import ActionItemExtractor
from entities import EntityExtractor
from observability import configure_json_logging, log_event
from pain_points import PainPointExtractor
from sentiment import SentimentAnalyser
from topics import TopicExtractor

configure_json_logging("nlp_service")
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/tmp/meeting_insights"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

NLP_STEP_LABELS = {
    "topics": "Key topic extraction (KeyBERT + LDA)",
    "entities": "Named entity recognition (spaCy en_core_web_trf)",
    "action_items": "Action item detection (regex + LLM)",
    "decisions": "Decision detection (sentence-level classification)",
    "pain_points": "Pain point extraction (classifier + LLM)",
    "sentiment": "Sentiment per speaker segment (RoBERTa)",
}
NLP_TOTAL_STEPS = len(NLP_STEP_LABELS)

celery_app = Celery("nlp_service", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="nlp",
    task_default_exchange="nlp",
    task_default_exchange_type="direct",
    task_default_routing_key="nlp",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,
)

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


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


def _load_transcript(session_id: str) -> dict:
    path = STORAGE_DIR / session_id / "transcript_speaker.json"
    if not path.exists():
        raise FileNotFoundError(
            f"transcript_speaker.json not found for session {session_id}. "
            "Ensure Stage 4 completed successfully."
        )
    return json.loads(path.read_text())


def _save(session_id: str, filename: str, data: dict | list) -> Path:
    path = STORAGE_DIR / session_id / filename
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def _nlp_state_key(session_id: str) -> str:
    return f"nlp:state:{session_id}"


def _nlp_timer_key(session_id: str) -> str:
    return f"nlp:timer:{session_id}"


def _reset_nlp_state(session_id: str) -> None:
    r = _get_redis()
    r.delete(_nlp_state_key(session_id))
    r.delete(_nlp_timer_key(session_id))


def _mark_nlp_start(session_id: str) -> None:
    r = _get_redis()
    r.set(_nlp_timer_key(session_id), str(time.time()), ex=86400)


def _nlp_duration_ms(session_id: str) -> int | None:
    raw = _get_redis().get(_nlp_timer_key(session_id))
    if not raw:
        return None
    return int((time.time() - float(raw)) * 1000)


def _mark_nlp_step(session_id: str, step: str, detail: str) -> None:
    r = _get_redis()
    key = _nlp_state_key(session_id)
    completed = r.hget(key, "completed_steps")
    completed_steps = set(filter(None, (completed or "").split(",")))
    completed_steps.add(step)
    completed_count = len(completed_steps)
    percent = min(95, 10 + int((completed_count / NLP_TOTAL_STEPS) * 85))
    r.hset(key, mapping={
        "completed_steps": ",".join(sorted(completed_steps)),
        "completed_count": str(completed_count),
    })
    r.expire(key, 86400)
    _publish(session_id, "nlp_analysis", "running", percent, detail)


def _log_task_started(session_id: str, task_name: str) -> float:
    started = time.perf_counter()
    log_event(
        logger,
        "nlp.task_started",
        session_id=session_id,
        stage="nlp_analysis",
        metadata={"task": task_name},
        message=f"Starting NLP task: {task_name}",
    )
    return started


def _log_task_completed(session_id: str, task_name: str, started: float, metadata: dict) -> None:
    log_event(
        logger,
        "nlp.task_completed",
        session_id=session_id,
        stage="nlp_analysis",
        duration_ms=(time.perf_counter() - started) * 1000,
        metadata={"task": task_name, **metadata},
        message=f"Completed NLP task: {task_name}",
    )


def _log_task_failed(session_id: str, task_name: str, started: float, exc: Exception) -> None:
    log_event(
        logger,
        "nlp.task_failed",
        session_id=session_id,
        stage="nlp_analysis",
        duration_ms=(time.perf_counter() - started) * 1000,
        metadata={"task": task_name},
        message=f"Failed NLP task: {task_name}",
        level=logging.ERROR,
        exc_info=exc,
    )


@celery_app.task(bind=True, name="nlp_service.run_topics", max_retries=2)
def run_topics(self, session_id: str) -> dict:
    task_started = _log_task_started(session_id, "topics")
    _publish(session_id, "nlp_analysis", "running", 12, NLP_STEP_LABELS["topics"])
    try:
        transcript = _load_transcript(session_id)
        extractor = TopicExtractor()
        result = extractor.extract(transcript["segments"])
        _save(session_id, "nlp_topics.json", result)
        _mark_nlp_step(
            session_id,
            "topics",
            f"{NLP_STEP_LABELS['topics']} complete: {len(result['keyphrases'])} keyphrases, {len(result['lda_topics'])} LDA topics",
        )
        _log_task_completed(
            session_id,
            "topics",
            task_started,
            {
                "keyphrase_count": len(result["keyphrases"]),
                "lda_topic_count": len(result["lda_topics"]),
            },
        )
        return {"task": "topics", "session_id": session_id, **result}
    except Exception as exc:
        _publish(session_id, "nlp_analysis", "failed", 0, str(exc))
        _log_task_failed(session_id, "topics", task_started, exc)
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="nlp_service.run_entities", max_retries=2)
def run_entities(self, session_id: str) -> dict:
    task_started = _log_task_started(session_id, "entities")
    _publish(session_id, "nlp_analysis", "running", 14, NLP_STEP_LABELS["entities"])
    try:
        transcript = _load_transcript(session_id)
        extractor = EntityExtractor()
        result = extractor.extract(transcript["segments"])
        _save(session_id, "nlp_entities.json", result)
        entity_count = sum(len(values) for values in result.values())
        _mark_nlp_step(
            session_id,
            "entities",
            f"{NLP_STEP_LABELS['entities']} complete: {entity_count} entities found",
        )
        _log_task_completed(
            session_id,
            "entities",
            task_started,
            {"entity_count": entity_count},
        )
        return {"task": "entities", "session_id": session_id, "entities": result}
    except Exception as exc:
        _publish(session_id, "nlp_analysis", "failed", 0, str(exc))
        _log_task_failed(session_id, "entities", task_started, exc)
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="nlp_service.run_action_items", max_retries=2)
def run_action_items(self, session_id: str) -> dict:
    task_started = _log_task_started(session_id, "action_items")
    _publish(session_id, "nlp_analysis", "running", 16, NLP_STEP_LABELS["action_items"])
    try:
        transcript = _load_transcript(session_id)
        extractor = ActionItemExtractor(
            ollama_url=OLLAMA_URL,
            model=LLM_MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
        action_items = extractor.extract_action_items(transcript["segments"])
        result = {"action_items": action_items}
        _save(session_id, "nlp_actions.json", result)
        _mark_nlp_step(
            session_id,
            "action_items",
            f"{NLP_STEP_LABELS['action_items']} complete: {len(action_items)} action items detected",
        )
        _log_task_completed(
            session_id,
            "action_items",
            task_started,
            {"action_item_count": len(action_items)},
        )
        return {"task": "action_items", "session_id": session_id, **result}
    except Exception as exc:
        _publish(session_id, "nlp_analysis", "failed", 0, str(exc))
        _log_task_failed(session_id, "action_items", task_started, exc)
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="nlp_service.run_decisions", max_retries=2)
def run_decisions(self, session_id: str) -> dict:
    task_started = _log_task_started(session_id, "decisions")
    _publish(session_id, "nlp_analysis", "running", 18, NLP_STEP_LABELS["decisions"])
    try:
        transcript = _load_transcript(session_id)
        extractor = ActionItemExtractor(
            ollama_url=OLLAMA_URL,
            model=LLM_MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
        decisions = extractor.extract_decisions(transcript["segments"])
        result = {"decisions": decisions}
        _save(session_id, "nlp_decisions.json", result)
        _mark_nlp_step(
            session_id,
            "decisions",
            f"{NLP_STEP_LABELS['decisions']} complete: {len(decisions)} decisions detected",
        )
        _log_task_completed(
            session_id,
            "decisions",
            task_started,
            {"decision_count": len(decisions)},
        )
        return {"task": "decisions", "session_id": session_id, **result}
    except Exception as exc:
        _publish(session_id, "nlp_analysis", "failed", 0, str(exc))
        _log_task_failed(session_id, "decisions", task_started, exc)
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="nlp_service.run_pain_points", max_retries=2)
def run_pain_points(self, session_id: str) -> dict:
    task_started = _log_task_started(session_id, "pain_points")
    _publish(session_id, "nlp_analysis", "running", 20, NLP_STEP_LABELS["pain_points"])
    try:
        transcript = _load_transcript(session_id)
        extractor = PainPointExtractor(
            threshold=0.34,
            classifier_threshold=0.62,
            ollama_url=OLLAMA_URL,
            model=LLM_MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
        result = extractor.extract(transcript["segments"])
        _save(session_id, "nlp_pain_points.json", result)
        _mark_nlp_step(
            session_id,
            "pain_points",
            f"{NLP_STEP_LABELS['pain_points']} complete: {len(result)} pain points detected",
        )
        _log_task_completed(
            session_id,
            "pain_points",
            task_started,
            {"pain_point_count": len(result)},
        )
        return {"task": "pain_points", "session_id": session_id, "pain_points": result}
    except Exception as exc:
        _publish(session_id, "nlp_analysis", "failed", 0, str(exc))
        _log_task_failed(session_id, "pain_points", task_started, exc)
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="nlp_service.run_sentiment", max_retries=2)
def run_sentiment(self, session_id: str) -> dict:
    task_started = _log_task_started(session_id, "sentiment")
    _publish(session_id, "nlp_analysis", "running", 22, NLP_STEP_LABELS["sentiment"])
    try:
        transcript = _load_transcript(session_id)
        analyser = SentimentAnalyser()
        result = analyser.analyse(transcript["segments"])
        _save(session_id, "nlp_sentiment.json", result)
        _mark_nlp_step(
            session_id,
            "sentiment",
            f"{NLP_STEP_LABELS['sentiment']} complete: {len(result['per_segment'])} segments analysed",
        )
        _log_task_completed(
            session_id,
            "sentiment",
            task_started,
            {"segment_count": len(result["per_segment"])},
        )
        return {"task": "sentiment", "session_id": session_id, **result}
    except Exception as exc:
        _publish(session_id, "nlp_analysis", "failed", 0, str(exc))
        _log_task_failed(session_id, "sentiment", task_started, exc)
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="nlp_service.merge_results")
def merge_results(self, task_results: list[dict], session_id: str) -> dict:
    merged: dict = {"session_id": session_id}
    for result in task_results:
        if isinstance(result, dict):
            merged[result.get("task", "unknown")] = {
                key: value
                for key, value in result.items()
                if key not in ("task", "session_id")
            }

    merged.setdefault("action_items", {"action_items": []})
    merged.setdefault("decisions", {"decisions": []})

    combined_actions = {
        "action_items": merged.get("action_items", {}).get("action_items", []),
        "decisions": merged.get("decisions", {}).get("decisions", []),
    }
    _save(session_id, "nlp_actions.json", combined_actions)
    _save(session_id, "nlp_decisions.json", merged.get("decisions", {"decisions": []}))
    _save(session_id, "nlp_results.json", merged)

    summary = {
        "status": "complete",
        "tasks": [task.get("task") for task in task_results if isinstance(task, dict)],
        "pain_point_count": len(merged.get("pain_points", {}).get("pain_points", [])),
        "action_item_count": len(merged.get("action_items", {}).get("action_items", [])),
        "decision_count": len(merged.get("decisions", {}).get("decisions", [])),
        "sentiment_segment_count": len(merged.get("sentiment", {}).get("per_segment", [])),
    }

    r = _get_redis()
    r.set(f"result:nlp:{session_id}", json.dumps(summary), ex=86400)
    _publish(
        session_id,
        "nlp_analysis",
        "complete",
        100,
        "NLP complete: topics, entities, action items, decisions, pain points, and sentiment",
    )
    log_event(
        logger,
        "stage.completed",
        session_id=session_id,
        stage="nlp_analysis",
        duration_ms=_nlp_duration_ms(session_id),
        metadata=summary,
        message="NLP analysis stage complete",
    )
    r.delete(_nlp_state_key(session_id))
    r.delete(_nlp_timer_key(session_id))
    return merged


@celery_app.task(bind=True, name="nlp_service.analyse", max_retries=1)
def analyse(self, session_id: str) -> str:
    _reset_nlp_state(session_id)
    _mark_nlp_start(session_id)
    _publish(
        session_id,
        "nlp_analysis",
        "running",
        0,
        "Launching NLP pipeline: topics, entities, action items, decisions, pain points, sentiment",
    )
    log_event(
        logger,
        "stage.started",
        session_id=session_id,
        stage="nlp_analysis",
        metadata={"task_count": NLP_TOTAL_STEPS},
        message="Starting NLP analysis stage",
    )

    job = chord(
        group(
            run_topics.s(session_id),
            run_entities.s(session_id),
            run_action_items.s(session_id),
            run_decisions.s(session_id),
            run_pain_points.s(session_id),
            run_sentiment.s(session_id),
        ),
        merge_results.s(session_id),
    )
    job.apply_async()

    log_event(
        logger,
        "nlp.chord_dispatched",
        session_id=session_id,
        stage="nlp_analysis",
        metadata={"task_count": NLP_TOTAL_STEPS},
        message="Dispatched NLP chord",
    )
    return session_id
