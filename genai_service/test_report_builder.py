from pathlib import Path

from report_builder import ReportBuilder


def test_parse_sections_supports_exact_headers():
    builder = ReportBuilder(Path("."))
    report = """## 1. Executive Summary
Summary text.

## 2. Key Decisions Made
- Decision one

## 3. Pain Points & Blockers
Blocker text.

## 4. Action Items
Action text.

## 5. Meeting Sentiment Arc
Sentiment text.

## 6. Key Topics Discussed
Topic text.

## 7. Recommended Follow-ups
Follow-up text.
"""

    sections = builder.parse_sections(report)

    assert sections[0]["content"] == "Summary text."
    assert sections[1]["content"] == "- Decision one"
    assert sections[6]["content"] == "Follow-up text."


def test_parse_sections_supports_fallback_markdown_headings():
    builder = ReportBuilder(Path("."))
    report = """## Meeting Intelligence Report

### Executive Summary
Summary text.

### Key Decisions Made
- Decision one

### Pain Points & Blockers
Blocker text.

### Action Items
Action text.

### Meeting Sentiment Arc
Sentiment text.

### Key Topics Discussed
Topic text.

### Recommended Follow-ups
Follow-up text.
"""

    sections = builder.parse_sections(report)

    assert sections[0]["content"] == "Summary text."
    assert sections[2]["content"] == "Blocker text."
    assert sections[6]["content"] == "Follow-up text."
