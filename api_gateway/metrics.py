"""
api-gateway/metrics.py
───────────────────────
Prometheus metrics definitions for all 10 required metrics from
spec section 7.1, plus a shared registry.

All metrics are recorded by the API gateway because it is the single
component that observes every stage transition via the Redis
progress:* pattern subscription.

Required metrics (spec §7.1)
─────────────────────────────
  audio_extraction_latency_ms    Histogram  alert: > 60 000
  transcription_latency_per_min  Gauge      alert: > 3× (real-time factor)
  transcription_confidence_score Histogram  alert: mean < 0.75
  diarisation_latency_ms         Histogram  alert: > 120 000
  speaker_count_detected         Gauge      informational
  pain_points_extracted_total    Counter    informational (labelled by severity)
  nlp_analysis_latency_ms        Histogram  alert: > 30 000
  genai_report_first_token_ms    Gauge      alert: > 5 000
  pipeline_stage_failure_rate    Counter    alert: > 5 %  (labelled by stage)
  search_query_latency_ms        Histogram  alert: > 500  (populated by search service)
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

# ─── Shared registry ──────────────────────────────────────────────────────────
# Using a custom registry instead of the default one avoids accidental
# collection of Go-style default process metrics that don't apply here.

REGISTRY = CollectorRegistry(auto_describe=True)

# ─── Metric definitions ───────────────────────────────────────────────────────

# 1. Audio extraction latency (Stage 2)
audio_extraction_latency = Histogram(
    "audio_extraction_latency_ms",
    "Time to extract + denoise + VAD audio from the uploaded video (milliseconds)",
    buckets=[1_000, 5_000, 10_000, 20_000, 30_000, 45_000, 60_000, 90_000, 120_000],
    registry=REGISTRY,
)

# 2. Transcription real-time factor (Stage 3)
# Value = transcription_time_seconds / audio_duration_seconds
# 1.0 = real-time, 3.0 = 3× slower than real-time (alert threshold)
transcription_rtf = Gauge(
    "transcription_latency_per_min",
    "Whisper processing time per minute of audio (real-time factor). Alert: > 3×",
    registry=REGISTRY,
)

# 3. Transcription confidence score (Stage 3)
transcription_confidence = Histogram(
    "transcription_confidence_score",
    "Average word-level confidence score from Whisper (0–1). Alert: mean < 0.75",
    buckets=[0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
    registry=REGISTRY,
)

# 4. Diarisation latency (Stage 4)
diarisation_latency = Histogram(
    "diarisation_latency_ms",
    "Time to complete pyannote speaker diarisation on the full audio (milliseconds)",
    buckets=[5_000, 10_000, 20_000, 30_000, 60_000, 90_000, 120_000, 180_000],
    registry=REGISTRY,
)

# 5. Speakers detected (Stage 4)
speaker_count = Gauge(
    "speaker_count_detected",
    "Number of distinct speakers identified by pyannote in the most recent meeting",
    registry=REGISTRY,
)

# 6. Pain points extracted (Stage 5)  — labelled by severity
pain_points_total = Counter(
    "pain_points_extracted_total",
    "Total pain points detected, labelled by severity",
    labelnames=["severity"],   # high | medium | low
    registry=REGISTRY,
)

# 7. NLP analysis latency (Stage 5)
nlp_latency = Histogram(
    "nlp_analysis_latency_ms",
    "Total time for all 5 NLP tasks to complete in parallel (milliseconds). Alert: > 30 000",
    buckets=[1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 45_000, 60_000],
    registry=REGISTRY,
)

# 8. GenAI first token latency (Stage 6)
genai_first_token = Gauge(
    "genai_report_first_token_ms",
    "Latency from NLP completion to first GenAI report token (milliseconds). Alert: > 5 000",
    registry=REGISTRY,
)

# 9. Pipeline stage failure rate — labelled by stage name
pipeline_failures = Counter(
    "pipeline_stage_failure_total",
    "Number of failures per pipeline stage. Alert: rate > 5 %",
    labelnames=["stage"],
    registry=REGISTRY,
)

pipeline_stage_events = Counter(
    "pipeline_stage_events_total",
    "Number of completed or failed pipeline stage terminal events",
    labelnames=["stage", "status"],
    registry=REGISTRY,
)

# 10. Search query latency — populated by the search service via a shared
# Prometheus push gateway or by this gateway when it proxies search calls
search_latency = Histogram(
    "search_query_latency_ms",
    "Time for transcript keyword search to return results (milliseconds). Alert: > 500",
    buckets=[10, 50, 100, 200, 300, 400, 500, 750, 1_000, 2_000],
    registry=REGISTRY,
)

active_processing_jobs = Gauge(
    "active_processing_jobs",
    "Number of meetings currently running or waiting for the next pipeline stage",
    registry=REGISTRY,
)

celery_queue_depth = Gauge(
    "celery_queue_depth",
    "Pending Celery tasks per Redis queue",
    labelnames=["queue"],
    registry=REGISTRY,
)

genai_requests_total = Counter(
    "genai_requests_total",
    "GenAI report generation requests observed by the gateway",
    labelnames=["status"],
    registry=REGISTRY,
)

genai_first_token_distribution = Histogram(
    "genai_report_first_token_distribution_ms",
    "Distribution of time-to-first-token for GenAI report streaming",
    buckets=[100, 250, 500, 1_000, 2_000, 3_000, 5_000, 8_000, 12_000],
    registry=REGISTRY,
)

pain_point_confidence = Histogram(
    "pain_point_confidence_score",
    "Pain point extraction confidence scores",
    buckets=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
    registry=REGISTRY,
)

action_items_total = Counter(
    "action_items_extracted_total",
    "Total action items extracted across meetings",
    registry=REGISTRY,
)

sentiment_score_distribution = Histogram(
    "sentiment_score_distribution",
    "Sentiment model score distribution labelled by class",
    labelnames=["label"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

# ─── Helper functions ─────────────────────────────────────────────────────────
# Called by main.py when it observes stage transitions in the orchestrator.

# Stage name → metric mapping
_STAGE_LATENCY_MAP = {
    "audio_extraction": audio_extraction_latency,
    "transcription":    None,      # transcription uses RTF, not raw ms
    "diarisation":      diarisation_latency,
    "nlp_analysis":     nlp_latency,
}


def record_stage_latency(stage: str, latency_ms: int) -> None:
    """
    Record stage completion latency.
    Called by the orchestrator when it observes a 'complete' event.
    """
    metric = _STAGE_LATENCY_MAP.get(stage)
    if metric is not None:
        metric.observe(latency_ms)


def record_stage_failure(stage: str) -> None:
    """Increment failure counter for a pipeline stage."""
    pipeline_failures.labels(stage=stage).inc()
    pipeline_stage_events.labels(stage=stage, status="failed").inc()


def record_stage_completion(stage: str) -> None:
    """Increment terminal completion counter for a pipeline stage."""
    pipeline_stage_events.labels(stage=stage, status="complete").inc()


def record_transcription_confidence(avg_confidence: float) -> None:
    """Record average Whisper word confidence for a completed transcription."""
    if 0.0 <= avg_confidence <= 1.0:
        transcription_confidence.observe(avg_confidence)


def record_transcription_rtf(
    transcription_time_s: float,
    audio_duration_s: float,
) -> None:
    """
    Record the real-time factor for a transcription job.
    rtf = transcription_time / audio_duration
    A value of 1.0 means Whisper ran at real-time speed.
    Values > 3.0 trigger the alert threshold.
    """
    if audio_duration_s > 0:
        rtf = transcription_time_s / audio_duration_s
        transcription_rtf.set(round(rtf, 4))


def record_speaker_count(n_speakers: int) -> None:
    """Record the number of speakers detected by pyannote."""
    speaker_count.set(n_speakers)


def record_pain_points_total(severity: str) -> None:
    """
    Increment the pain point counter.
    severity must be one of: high | medium | low
    """
    sev = severity.lower() if severity.lower() in ("high", "medium", "low") else "low"
    pain_points_total.labels(severity=sev).inc()


def record_genai_first_token(latency_ms: int) -> None:
    """Record GenAI time-to-first-token latency."""
    genai_first_token.set(latency_ms)
    genai_first_token_distribution.observe(latency_ms)


def increment_search_latency(latency_ms: int) -> None:
    """Record a search query latency — called when the gateway proxies a search."""
    search_latency.observe(latency_ms)


def set_active_processing_jobs(count: int) -> None:
    active_processing_jobs.set(max(0, count))


def set_queue_depth(queue: str, count: int) -> None:
    celery_queue_depth.labels(queue=queue).set(max(0, count))


def record_genai_request(status: str) -> None:
    value = status.lower().strip() or "unknown"
    genai_requests_total.labels(status=value).inc()


def record_pain_point_confidence(score: float) -> None:
    if 0.0 <= score <= 1.0:
        pain_point_confidence.observe(score)


def record_action_items_total(count: int) -> None:
    if count > 0:
        action_items_total.inc(count)


def record_sentiment_score(label: str, score: float) -> None:
    if 0.0 <= score <= 1.0:
        sentiment_score_distribution.labels(label=label.lower()).observe(score)
