"""
genai-service/test_genai.py
────────────────────────────
Standalone test script for Stage 6 — runs without Docker.

Tests:
  1. Context assembly from mock session files
  2. Prompt construction (prints to terminal)
  3. Streaming from Ollama (if running locally)
  4. Streaming from OpenAI (if OPENAI_API_KEY is set)
  5. Full end-to-end: prompt → stream → save report

Usage
─────
  # Test 1 + 2 only (no LLM needed):
  python test_genai.py context

  # Test with Ollama (must have Ollama running: ollama serve):
  python test_genai.py ollama

  # Test with OpenAI:
  set OPENAI_API_KEY=sk-...
  python test_genai.py openai

  # Full end-to-end with Ollama:
  python test_genai.py all
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# ── Create mock session directory ────────────────────────────────────────────

SESSION_DIR = Path(__file__).parent / "test_session"
SESSION_DIR.mkdir(exist_ok=True)


def create_mock_files():
    """Write all mock NLP output files that report_builder.py expects."""

    # transcript_speaker.json
    (SESSION_DIR / "transcript_speaker.json").write_text(json.dumps({
        "speaker_count": 3,
        "segments": [
            {"segment_id": 0, "speaker": "SPEAKER_00", "start": 0.5,  "end": 14.2,
             "text": "Hey everyone, let's get started. Today we are reviewing the Q3 pipeline and the API rate limit issue that has been causing failures.", "confidence": 0.94},
            {"segment_id": 1, "speaker": "SPEAKER_01", "start": 14.8, "end": 35.3,
             "text": "We are completely blocked on the authentication service. Every deploy breaks the token refresh flow and we have had three incidents in the past two weeks. We cannot ship the release until this is fixed.", "confidence": 0.89},
            {"segment_id": 2, "speaker": "SPEAKER_02", "start": 36.1, "end": 55.7,
             "text": "The batch job is not respecting the backoff headers. The rate limiter triggers every evening at 7pm causing downstream failures in the reporting pipeline.", "confidence": 0.91},
            {"segment_id": 3, "speaker": "SPEAKER_00", "start": 56.4, "end": 72.1,
             "text": "Alice, can you take ownership of the backoff fix by Friday? And Bob, please review the deployment pipeline before end of day tomorrow.", "confidence": 0.96},
            {"segment_id": 4, "speaker": "SPEAKER_01", "start": 73.0, "end": 88.5,
             "text": "Sure, I will handle the backoff fix by Friday. We have decided to migrate to the new API gateway next sprint. The team has agreed this is the right call.", "confidence": 0.92},
            {"segment_id": 5, "speaker": "SPEAKER_02", "start": 89.2, "end": 105.8,
             "text": "We are also understaffed on the infrastructure side. No dedicated DevOps engineer and the manual deployment process is slowing everything down. We need to hire someone.", "confidence": 0.88},
            {"segment_id": 6, "speaker": "SPEAKER_00", "start": 106.5, "end": 118.3,
             "text": "Agreed. Going forward we will prioritise the DevOps hire in Q4. Great work everyone despite these blockers.", "confidence": 0.95},
        ]
    }, indent=2))

    # nlp_topics.json
    (SESSION_DIR / "nlp_topics.json").write_text(json.dumps({
        "keyphrases": [
            {"phrase": "api rate limit",            "score": 0.821},
            {"phrase": "token refresh flow",        "score": 0.774},
            {"phrase": "deployment pipeline",       "score": 0.741},
            {"phrase": "authentication service",    "score": 0.698},
            {"phrase": "reporting pipeline",        "score": 0.672},
            {"phrase": "backoff headers",           "score": 0.654},
            {"phrase": "devops engineer",           "score": 0.621},
            {"phrase": "api gateway migration",     "score": 0.598},
            {"phrase": "batch job failure",         "score": 0.574},
            {"phrase": "release blocker",           "score": 0.541},
        ],
        "lda_topics": [
            {"topic_id": 0, "top_words": ["rate", "limit", "api", "failure", "pipeline"], "weight": 0.38},
            {"topic_id": 1, "top_words": ["auth", "token", "deploy", "fix", "service"],  "weight": 0.27},
            {"topic_id": 2, "top_words": ["devops", "hire", "manual", "process", "slow"],"weight": 0.18},
            {"topic_id": 3, "top_words": ["gateway", "migration", "sprint", "team", "agreed"], "weight": 0.10},
            {"topic_id": 4, "top_words": ["reporting", "downstream", "batch", "evening", "trigger"], "weight": 0.07},
        ]
    }, indent=2))

    # nlp_entities.json
    (SESSION_DIR / "nlp_entities.json").write_text(json.dumps({
        "PERSON":  [{"text": "Alice", "count": 2, "first_seen_at": 56.4, "speakers": ["SPEAKER_00"]},
                    {"text": "Bob",   "count": 1, "first_seen_at": 56.4, "speakers": ["SPEAKER_00"]}],
        "ORG":     [{"text": "Q3 Pipeline", "count": 1, "first_seen_at": 0.5,  "speakers": ["SPEAKER_00"]}],
        "PRODUCT": [{"text": "Api Gateway", "count": 2, "first_seen_at": 73.0, "speakers": ["SPEAKER_01"]}],
        "DATE":    [{"text": "Friday",      "count": 2, "first_seen_at": 56.4, "speakers": ["SPEAKER_00", "SPEAKER_01"]},
                    {"text": "Q4",          "count": 1, "first_seen_at": 106.5,"speakers": ["SPEAKER_00"]}],
        "GPE":     [], "EVENT": []
    }, indent=2))

    # nlp_actions.json
    (SESSION_DIR / "nlp_actions.json").write_text(json.dumps({
        "action_items": [
            {"action": "Fix backoff headers in batch job", "owner": "SPEAKER_01",
             "deadline": "Friday", "confidence": 0.91,
             "quote": "Alice, can you take ownership of the backoff fix by Friday?",
             "speaker": "SPEAKER_00", "timestamp": 56.4},
            {"action": "Review deployment pipeline", "owner": "SPEAKER_02",
             "deadline": "end of day tomorrow", "confidence": 0.87,
             "quote": "Bob, please review the deployment pipeline before end of day tomorrow.",
             "speaker": "SPEAKER_00", "timestamp": 68.1},
        ],
        "decisions": [
            {"decision": "Migrate to the new API gateway next sprint",
             "speaker": "SPEAKER_01", "timestamp": 78.3, "confidence": 0.93,
             "quote": "We have decided to migrate to the new API gateway next sprint."},
            {"decision": "Prioritise DevOps hire in Q4",
             "speaker": "SPEAKER_00", "timestamp": 108.2, "confidence": 0.89,
             "quote": "Going forward we will prioritise the DevOps hire in Q4."},
        ]
    }, indent=2))

    # nlp_pain_points.json
    (SESSION_DIR / "nlp_pain_points.json").write_text(json.dumps([
        {"pain_point": "Authentication service blocking release — token refresh breaks on every deploy",
         "speaker": "SPEAKER_01", "timestamp": 14.8,
         "severity": "high", "category": "technical_blocker",
         "quote": "Every deploy breaks the token refresh flow and we have had three incidents",
         "confidence": 0.94},
        {"pain_point": "API rate limiter triggering nightly — causing downstream reporting failures",
         "speaker": "SPEAKER_02", "timestamp": 36.1,
         "severity": "high", "category": "technical_blocker",
         "quote": "rate limiter triggers every evening causing downstream failures",
         "confidence": 0.91},
        {"pain_point": "No dedicated DevOps engineer — manual deployment process is a bottleneck",
         "speaker": "SPEAKER_02", "timestamp": 89.2,
         "severity": "medium", "category": "resource_constraint",
         "quote": "We are understaffed on the infrastructure side. No dedicated DevOps engineer.",
         "confidence": 0.83},
    ], indent=2))

    # nlp_sentiment.json
    (SESSION_DIR / "nlp_sentiment.json").write_text(json.dumps({
        "per_segment": [
            {"segment_id": 0, "speaker": "SPEAKER_00", "start": 0.5,   "end": 14.2,
             "sentiment": "neutral",  "scores": {"positive": 0.18, "neutral": 0.71, "negative": 0.11},
             "text": "Hey everyone, let's get started..."},
            {"segment_id": 1, "speaker": "SPEAKER_01", "start": 14.8,  "end": 35.3,
             "sentiment": "negative", "scores": {"positive": 0.04, "neutral": 0.12, "negative": 0.84},
             "text": "We are completely blocked on the authentication service..."},
            {"segment_id": 2, "speaker": "SPEAKER_02", "start": 36.1,  "end": 55.7,
             "sentiment": "negative", "scores": {"positive": 0.06, "neutral": 0.15, "negative": 0.79},
             "text": "The batch job is not respecting the backoff headers..."},
            {"segment_id": 3, "speaker": "SPEAKER_00", "start": 56.4,  "end": 72.1,
             "sentiment": "neutral",  "scores": {"positive": 0.24, "neutral": 0.62, "negative": 0.14},
             "text": "Alice, can you take ownership of the backoff fix by Friday?"},
            {"segment_id": 4, "speaker": "SPEAKER_01", "start": 73.0,  "end": 88.5,
             "sentiment": "positive", "scores": {"positive": 0.67, "neutral": 0.26, "negative": 0.07},
             "text": "Sure, I will handle the backoff fix by Friday..."},
            {"segment_id": 5, "speaker": "SPEAKER_02", "start": 89.2,  "end": 105.8,
             "sentiment": "negative", "scores": {"positive": 0.05, "neutral": 0.18, "negative": 0.77},
             "text": "We are also understaffed on the infrastructure side..."},
            {"segment_id": 6, "speaker": "SPEAKER_00", "start": 106.5, "end": 118.3,
             "sentiment": "positive", "scores": {"positive": 0.74, "neutral": 0.19, "negative": 0.07},
             "text": "Agreed. Going forward we will prioritise the DevOps hire in Q4..."},
        ],
        "per_speaker": {
            "SPEAKER_00": {"positive_pct": 0.33, "neutral_pct": 0.50, "negative_pct": 0.17,
                           "dominant": "neutral",  "trend": ["neutral","neutral","positive"], "segment_count": 3},
            "SPEAKER_01": {"positive_pct": 0.50, "neutral_pct": 0.00, "negative_pct": 0.50,
                           "dominant": "mixed",    "trend": ["negative","positive"],          "segment_count": 2},
            "SPEAKER_02": {"positive_pct": 0.00, "neutral_pct": 0.00, "negative_pct": 1.00,
                           "dominant": "negative", "trend": ["negative","negative"],          "segment_count": 2},
        },
        "overall": {"positive_pct": 0.29, "neutral_pct": 0.28, "negative_pct": 0.43}
    }, indent=2))

    # metadata.json
    (SESSION_DIR / "metadata.json").write_text(json.dumps({
        "duration": 118.3, "has_audio": True, "has_video": True,
        "resolution": "1280x720", "fps": 30.0,
        "audio_codec": "aac", "video_codec": "h264",
    }, indent=2))

    print(f"✅ Mock session files created in: {SESSION_DIR}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_context():
    """Test context assembly and prompt building — no LLM needed."""
    print("\n" + "="*60)
    print("TEST 1: Context Assembly + Prompt Building")
    print("="*60)

    from report_builder import ReportBuilder
    builder = ReportBuilder(SESSION_DIR)
    ctx     = builder.assemble()

    print(f"\n✅ Context assembled:")
    print(f"   Session ID    : {ctx.session_id}")
    print(f"   Duration      : {ctx.duration_secs}s")
    print(f"   Speakers      : {ctx.speaker_count}")
    print(f"   Keyphrases    : {len(ctx.keyphrases)}")
    print(f"   Action items  : {len(ctx.action_items)}")
    print(f"   Decisions     : {len(ctx.decisions)}")
    print(f"   Pain points   : {len(ctx.pain_points)}")

    prompt = builder.build_prompt(ctx)
    print(f"\n✅ Prompt built: {len(prompt)} characters")
    print(f"\n--- PROMPT PREVIEW (first 800 chars) ---")
    print(prompt[:800])
    print("...[truncated]...")

    # Save prompt for inspection
    prompt_path = SESSION_DIR / "debug_prompt.txt"
    prompt_path.write_text(prompt)
    print(f"\n💾 Full prompt saved to: {prompt_path}")
    return ctx, prompt


async def test_ollama_stream():
    """Test streaming from Ollama."""
    print("\n" + "="*60)
    print("TEST 2: Ollama Streaming")
    print("="*60)

    from report_builder import ReportBuilder
    from streamer import ReportStreamer, LLMBackend

    builder  = ReportBuilder(SESSION_DIR)
    ctx      = builder.assemble()
    prompt   = builder.build_prompt(ctx)
    streamer = ReportStreamer(
        backend=LLMBackend.OLLAMA,
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
        temperature=0.3,
        max_tokens=4096,
    )

    print("\n🚀 Streaming report from Ollama...\n")
    print("-" * 60)

    report_text = ""
    sections_seen = set()
    token_count  = 0

    from app import _detect_section

    async for event in streamer.stream(prompt):
        if event["type"] == "token":
            token = event["content"]
            report_text += token
            print(token, end="", flush=True)
            token_count += 1

            # Detect sections
            section = _detect_section(report_text, max(sections_seen, default=-1))
            if section and section["index"] not in sections_seen:
                sections_seen.add(section["index"])
                print(f"\n\n  ✅ [SECTION DETECTED: {section['title']}]\n")

        elif event["type"] == "error":
            print(f"\n❌ Error: {event['message']}")
            return

    print(f"\n\n{'='*60}")
    print(f"✅ Stream complete: {token_count} tokens")

    # Save report
    report_path = SESSION_DIR / "genai_report.md"
    report_path.write_text(report_text)
    print(f"💾 Report saved: {report_path}")

    # Parse sections
    from report_builder import ReportBuilder
    sections = builder.parse_sections(report_text)
    json_path = SESSION_DIR / "genai_report.json"
    json_path.write_text(json.dumps({"sections": sections}, indent=2))
    print(f"💾 JSON saved:   {json_path}")
    print(f"\nSections generated: {[s['title'] for s in sections if s['content']]}")


async def test_openai_stream():
    """Test streaming from OpenAI."""
    print("\n" + "="*60)
    print("TEST 3: OpenAI Streaming")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping.")
        return

    from report_builder import ReportBuilder
    from streamer import ReportStreamer, LLMBackend

    builder  = ReportBuilder(SESSION_DIR)
    ctx      = builder.assemble()
    prompt   = builder.build_prompt(ctx)
    streamer = ReportStreamer(
        backend=LLMBackend.OPENAI,
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.3,
        max_tokens=4096,
    )

    print("\n🚀 Streaming report from OpenAI gpt-4o...\n")
    print("-" * 60)

    report_text = ""
    token_count = 0

    async for event in streamer.stream(prompt):
        if event["type"] == "token":
            token = event["content"]
            report_text += token
            print(token, end="", flush=True)
            token_count += 1
        elif event["type"] == "error":
            print(f"\n❌ Error: {event['message']}")
            return

    print(f"\n\n✅ Stream complete: {token_count} tokens")
    (SESSION_DIR / "genai_report_openai.md").write_text(report_text)


async def run_all():
    create_mock_files()
    test_context()
    await test_ollama_stream()
    await test_openai_stream()


if __name__ == "__main__":
    create_mock_files()  # always create fresh mock data

    mode = sys.argv[1] if len(sys.argv) > 1 else "context"

    if mode == "context":
        test_context()
    elif mode == "ollama":
        asyncio.run(test_ollama_stream())
    elif mode == "openai":
        asyncio.run(test_openai_stream())
    elif mode == "all":
        asyncio.run(run_all())
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python test_genai.py [context|ollama|openai|all]")
        sys.exit(1)
