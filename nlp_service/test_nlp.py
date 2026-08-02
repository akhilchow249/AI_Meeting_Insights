"""
Run all 5 NLP modules individually against the mock transcript.
Usage: python test_nlp.py [topics|entities|actions|pain_points|sentiment|all]
"""

import json
import sys
from pathlib import Path

# ── Load mock transcript ──────────────────────────────────────────────────────
TRANSCRIPT_PATH = Path(__file__).parent / "test_data" / "transcript_speaker.json"
transcript = json.loads(TRANSCRIPT_PATH.read_text())
segments   = transcript["segments"]


def test_topics():
    print("\n" + "="*60)
    print("TEST: KeyBERT + LDA Topic Extraction")
    print("="*60)
    from topics import TopicExtractor
    result = TopicExtractor().extract(segments)

    print(f"\n📌 Top {len(result['keyphrases'])} Keyphrases:")
    for kp in result["keyphrases"]:
        print(f"   {kp['score']:.3f}  {kp['phrase']}")

    print(f"\n📚 {len(result['lda_topics'])} LDA Topics:")
    for t in result["lda_topics"]:
        print(f"   Topic {t['topic_id']} ({t['weight']:.1%}): {', '.join(t['top_words'])}")

    return result


def test_entities():
    print("\n" + "="*60)
    print("TEST: spaCy Named Entity Recognition")
    print("="*60)
    from entities import EntityExtractor
    result = EntityExtractor().extract(segments)

    for etype, entities in result.items():
        if entities:
            print(f"\n  [{etype}]")
            for ent in entities[:5]:
                print(f"    x{ent['count']:2d}  {ent['text']:30s}  "
                      f"speakers={ent['speakers']}  "
                      f"first_seen={ent['first_seen_at']}s")

    return result


def test_actions():
    print("\n" + "="*60)
    print("TEST: Action Item + Decision Detection (regex only — no Ollama needed)")
    print("="*60)

    # Monkey-patch LLM call to return fallback (no Ollama running locally)
    import action_items as ai_module
    ai_module._llm_call = lambda *args, **kwargs: None

    from action_items import ActionItemExtractor
    result = ActionItemExtractor(ollama_url="http://localhost:11434").extract(segments)

    print(f"\n✅ Action Items ({len(result['action_items'])}):")
    for item in result["action_items"]:
        print(f"   [{item['speaker']} @ {item['timestamp']}s]  "
              f"owner={item['owner']}  deadline={item['deadline']}")
        print(f"   → {item['action']}")
        print(f"   quote: \"{item['quote'][:80]}\"")
        print(f"   confidence: {item['confidence']}")

    print(f"\n🔨 Decisions ({len(result['decisions'])}):")
    for d in result["decisions"]:
        print(f"   [{d['speaker']} @ {d['timestamp']}s]  conf={d['confidence']}")
        print(f"   → {d['decision']}")

    return result


def test_pain_points():
    print("\n" + "="*60)
    print("TEST: Pain Point Extraction (heuristic + regex — no Ollama needed)")
    print("="*60)

    # Monkey-patch LLM to return structured response based on heuristic
    def mock_llm_extract(sentence, speaker, context, ollama_url, model):
        import re
        from pain_points import _heuristic_score, SEVERITY_VALUES, CATEGORY_VALUES
        score = _heuristic_score(sentence)
        if score < 0.3:
            return {"is_pain_point": False}
        # Simple severity heuristic
        severity = "high" if score > 0.6 else "medium" if score > 0.35 else "low"
        # Simple category guess
        category = "technical_blocker"
        if re.search(r"\bstaff|hire|resource\b", sentence, re.I):
            category = "resource_constraint"
        elif re.search(r"\bmanual|process|workflow\b", sentence, re.I):
            category = "process_inefficiency"
        return {
            "is_pain_point": True,
            "pain_point": sentence[:200],
            "severity": severity,
            "category": category,
            "quote": sentence[:100],
            "confidence": round(score, 3),
        }

    import pain_points as pp_module
    pp_module._llm_extract = mock_llm_extract

    from pain_points import PainPointExtractor
    result = PainPointExtractor(threshold=0.25).extract(segments)

    print(f"\n🚨 Pain Points ({len(result)}):")
    for pp in result:
        print(f"\n   [{pp['speaker']} @ {pp['timestamp']}s]")
        print(f"   severity  : {pp['severity'].upper()}")
        print(f"   category  : {pp['category']}")
        print(f"   confidence: {pp['confidence']}")
        print(f"   quote     : \"{pp['quote'][:80]}\"")
        print(f"   → {pp['pain_point'][:120]}")

    return result


def test_sentiment():
    print("\n" + "="*60)
    print("TEST: RoBERTa Sentiment Analysis")
    print("="*60)
    from sentiment import SentimentAnalyser
    result = SentimentAnalyser().analyse(segments)

    print(f"\n📊 Per-Segment Sentiment:")
    for seg in result["per_segment"]:
        bar = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
        icon = bar.get(seg["sentiment"], "⚪")
        print(f"   {icon} [{seg['speaker']}]  {seg['sentiment']:8s}  "
              f"+{seg['scores']['positive']:.2f} "
              f"~{seg['scores']['neutral']:.2f} "
              f"-{seg['scores']['negative']:.2f}  "
              f"\"{seg['text'][:60]}…\"")

    print(f"\n👥 Per-Speaker Summary:")
    for speaker, stats in result["per_speaker"].items():
        print(f"   {speaker}: "
              f"🟢{stats['positive_pct']:.0%}  "
              f"🟡{stats['neutral_pct']:.0%}  "
              f"🔴{stats['negative_pct']:.0%}  "
              f"(dominant: {stats['dominant']})")

    print(f"\n🌍 Overall:")
    o = result["overall"]
    print(f"   🟢 Positive {o['positive_pct']:.0%}  "
          f"🟡 Neutral {o['neutral_pct']:.0%}  "
          f"🔴 Negative {o['negative_pct']:.0%}")

    return result


def run_all():
    results = {}
    results["topics"]      = test_topics()
    results["entities"]    = test_entities()
    results["actions"]     = test_actions()
    results["pain_points"] = test_pain_points()
    results["sentiment"]   = test_sentiment()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE — saving combined output")
    print("="*60)

    import json
    out = Path(__file__).parent / "test_data" / "nlp_test_output.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"Output saved to: {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    tests = {
        "topics":      test_topics,
        "entities":    test_entities,
        "actions":     test_actions,
        "pain_points": test_pain_points,
        "sentiment":   test_sentiment,
        "all":         run_all,
    }
    fn = tests.get(mode)
    if fn is None:
        print(f"Unknown test: {mode}. Choose from: {list(tests.keys())}")
        sys.exit(1)
    fn()