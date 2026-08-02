"""
Assemble transcript and NLP outputs into a structured prompt for report generation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ContextAssemblyError(RuntimeError):
    """Raised when required session files are missing."""


@dataclass
class ReportContext:
    session_id: str
    duration_secs: float
    speaker_count: int
    transcript_text: str
    transcript_segments: list[dict]
    keyphrases: list[dict]
    lda_topics: list[dict]
    entities: dict[str, list[dict]]
    action_items: list[dict]
    decisions: list[dict]
    pain_points: list[dict]
    sentiment_per_speaker: dict[str, dict]
    sentiment_overall: dict[str, float]
    sentiment_per_segment: list[dict]


class ReportBuilder:
    """
    Load session files, assemble LLM context, build the prompt, and split the
    generated report back into the 7 required sections.
    """

    MAX_TRANSCRIPT_CHARS = 2_200
    MAX_TRANSCRIPT_LINES = 12
    MAX_TRANSCRIPT_LINE_CHARS = 160
    MAX_KEYPHRASES = 10
    MAX_ENTITY_VALUES = 2
    MAX_ACTION_ITEMS = 8
    MAX_DECISIONS = 6
    MAX_PAIN_POINTS = 6
    MAX_SENTIMENT_SPEAKERS = 4
    MAX_NEGATIVE_SPIKES = 2
    MAX_QUOTE_CHARS = 60

    SECTION_DEFS = [
        {
            "index": 0,
            "header": "## 1. Executive Summary",
            "title": "Executive Summary",
            "aliases": {"executive summary"},
        },
        {
            "index": 1,
            "header": "## 2. Key Decisions Made",
            "title": "Key Decisions Made",
            "aliases": {"key decisions made", "key decisions", "decisions made"},
        },
        {
            "index": 2,
            "header": "## 3. Pain Points & Blockers",
            "title": "Pain Points & Blockers",
            "aliases": {
                "pain points and blockers",
                "pain points blockers",
                "pain points",
                "blockers",
            },
        },
        {
            "index": 3,
            "header": "## 4. Action Items",
            "title": "Action Items",
            "aliases": {"action items", "next actions", "actions"},
        },
        {
            "index": 4,
            "header": "## 5. Meeting Sentiment Arc",
            "title": "Meeting Sentiment Arc",
            "aliases": {"meeting sentiment arc", "sentiment arc", "meeting sentiment"},
        },
        {
            "index": 5,
            "header": "## 6. Key Topics Discussed",
            "title": "Key Topics Discussed",
            "aliases": {"key topics discussed", "key topics", "topics discussed"},
        },
        {
            "index": 6,
            "header": "## 7. Recommended Follow-ups",
            "title": "Recommended Follow-ups",
            "aliases": {
                "recommended follow ups",
                "follow up recommendations",
                "followup recommendations",
                "recommended next steps",
            },
        },
    ]

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir

    def assemble(self) -> ReportContext:
        transcript_data = self._load_required("transcript_speaker.json")

        topics_data = self._load_optional("nlp_topics.json", {"keyphrases": [], "lda_topics": []})
        entities_data = self._load_optional("nlp_entities.json", {})
        actions_data = self._load_optional("nlp_actions.json", {"action_items": [], "decisions": []})
        decisions_data = self._load_optional("nlp_decisions.json", {"decisions": []})
        pain_data = self._load_optional("nlp_pain_points.json", [])
        sentiment_data = self._load_optional(
            "nlp_sentiment.json",
            {"per_speaker": {}, "overall": {}, "per_segment": []},
        )
        metadata = self._load_optional("metadata.json", {"duration": 0})

        pain_points: list[dict] = (
            pain_data if isinstance(pain_data, list) else pain_data.get("pain_points", [])
        )

        transcript_text = self._format_transcript(transcript_data.get("segments", []))

        return ReportContext(
            session_id=self.session_dir.name,
            duration_secs=float(metadata.get("duration", 0)),
            speaker_count=transcript_data.get("speaker_count", 1),
            transcript_text=transcript_text,
            transcript_segments=transcript_data.get("segments", []),
            keyphrases=topics_data.get("keyphrases", []),
            lda_topics=topics_data.get("lda_topics", []),
            entities=entities_data,
            action_items=actions_data.get("action_items", []),
            decisions=decisions_data.get("decisions", actions_data.get("decisions", [])),
            pain_points=pain_points,
            sentiment_per_speaker=sentiment_data.get("per_speaker", {}),
            sentiment_overall=sentiment_data.get("overall", {}),
            sentiment_per_segment=sentiment_data.get("per_segment", []),
        )

    def build_prompt(self, ctx: ReportContext) -> str:
        duration_str = self._format_duration(ctx.duration_secs)
        transcript_block = self._fmt_transcript_block(ctx)
        keyphrases_block = self._fmt_keyphrases(ctx)
        entities_block = self._fmt_entities(ctx)
        actions_block = self._fmt_actions(ctx)
        decisions_block = self._fmt_decisions(ctx)
        pain_points_block = self._fmt_pain_points(ctx)
        sentiment_block = self._fmt_sentiment(ctx)

        return f"""You are an expert meeting analyst. You have been given a diarised meeting transcript and structured NLP outputs. Write only the narrative sections of the final meeting report.

MEETING METADATA
Duration: {duration_str}
Speakers: {ctx.speaker_count} participants
Session ID: {ctx.session_id}

TRANSCRIPT
{transcript_block}

EXTRACTED KEYPHRASES (KeyBERT)
{keyphrases_block}

NAMED ENTITIES
{entities_block}

ACTION ITEMS DETECTED
{actions_block}

DECISIONS DETECTED
{decisions_block}

PAIN POINTS DETECTED
{pain_points_block}

SENTIMENT ANALYSIS
{sentiment_block}

INSTRUCTIONS
Write the report below and follow these rules EXACTLY:

1. Output EXACTLY these 4 section headers and nothing else:
   ## 1. Executive Summary
   ## 5. Meeting Sentiment Arc
   ## 6. Key Topics Discussed
   ## 7. Recommended Follow-ups

2. Do NOT add any extra headings, subheadings, "continued" headings, notes, preamble, or closing text.
3. Use only facts grounded in the transcript and NLP outputs. If something is missing, write "Not specified" instead of inventing it.
4. Prefer generic labels such as Speaker 0, Speaker 1, etc. Do not output raw labels like SPEAKER_00.
5. Keep the report compact, direct, and information-dense.
6. Put each sentence, bullet, and table row on its own line so the report streams cleanly line by line.
7. Start writing Section 1 immediately. Do not wait to plan the later sections before producing the executive summary.
8. If the pain-point detector missed issues, infer blockers from the transcript. Delays, risks, concerns, objections, unresolved asks, dependencies, and budget/scope pressure all count as pain points.
9. Decisions, pain points, and action items are rendered separately by the structured pipeline. Do not create sections 2, 3, or 4 yourself, and do not restate their full contents inside sections 5, 6, or 7.
10. Output all 4 requested sections even if a section has no items. In that case, write one explicit line saying nothing was detected.

SECTION REQUIREMENTS
- Section 1: Write 3 to 4 short sentences, each on its own line, covering meeting purpose, key outcome, most important decision, and most critical pain point.
- Section 5: Write 2 short paragraphs at most. Keep them compact. Describe how the tone evolved, which speaker drove the most positive energy, whether there were tension spikes, and any relevant timestamps.
- Section 6: Write a numbered list of up to 10 KeyBERT-driven topics. Each item must be the topic name followed by one short sentence explaining how it was discussed.
- Section 7: Write 3 to 5 bullet points with concrete recommendations for what should be done before the next meeting.

Begin immediately with ## 1. Executive Summary.

REPORT:"""

    def build_structured_sections(self, ctx: ReportContext) -> list[dict]:
        sections = {
            section["index"]: {
                "index": section["index"],
                "title": section["title"],
                "header": section["header"],
                "content": "",
            }
            for section in self.SECTION_DEFS
        }
        sections[1]["content"] = self._render_key_decisions(ctx)
        sections[2]["content"] = self._render_pain_points(ctx)
        sections[3]["content"] = self._render_action_items(ctx)
        return [sections[section["index"]] for section in self.SECTION_DEFS]

    def merge_sections(
        self,
        generated_sections: list[dict],
        structured_sections: list[dict],
    ) -> list[dict]:
        sections = {
            section["index"]: {
                "index": section["index"],
                "title": section["title"],
                "header": section["header"],
                "content": section.get("content", "").strip(),
            }
            for section in generated_sections
        }

        for section in structured_sections:
            if section["index"] in {1, 2, 3}:
                sections[section["index"]] = {
                    "index": section["index"],
                    "title": section["title"],
                    "header": section["header"],
                    "content": section.get("content", "").strip(),
                }

        for definition in self.SECTION_DEFS:
            sections.setdefault(
                definition["index"],
                {
                    "index": definition["index"],
                    "title": definition["title"],
                    "header": definition["header"],
                    "content": "",
                },
            )

        return [sections[section["index"]] for section in self.SECTION_DEFS]

    def render_sections(self, sections: list[dict]) -> str:
        chunks: list[str] = []
        for section in sections:
            header = section.get("header", "").strip()
            content = section.get("content", "").strip()
            if not header:
                continue
            chunks.append(header)
            if content:
                chunks.append(content)
        return "\n\n".join(chunks).strip() + "\n"

    def parse_sections(self, report_text: str) -> list[dict]:
        report_text = report_text.replace("\r\n", "\n")
        matches: list[dict] = []
        seen: set[int] = set()
        heading_re = re.compile(r"(?m)^[ \t]{0,3}(#{2,6})[ \t]+(.+?)\s*$")

        for match in heading_re.finditer(report_text):
            section = self._match_section(match.group(2))
            if section is None or section["index"] in seen:
                continue

            seen.add(section["index"])
            matches.append(
                {
                    "index": section["index"],
                    "start": match.start(),
                    "content_start": match.end(),
                }
            )

        matches.sort(key=lambda item: item["start"])
        sections_by_index = {
            section["index"]: {
                "index": section["index"],
                "title": section["title"],
                "header": section["header"],
                "content": "",
            }
            for section in self.SECTION_DEFS
        }

        for i, match in enumerate(matches):
            end = matches[i + 1]["start"] if i + 1 < len(matches) else len(report_text)
            sections_by_index[match["index"]]["content"] = report_text[
                match["content_start"] : end
            ].strip()

        return [sections_by_index[section["index"]] for section in self.SECTION_DEFS]

    def _load_required(self, filename: str) -> dict:
        path = self.session_dir / filename
        if not path.exists():
            raise ContextAssemblyError(
                f"Required file '{filename}' not found in session directory. "
                f"Ensure Stage 4 (diarisation) completed successfully."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_optional(self, filename: str, default: Any) -> Any:
        path = self.session_dir / filename
        if not path.exists():
            logger.warning("Optional file '%s' not found, using default.", filename)
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse '%s': %s, using default.", filename, exc)
            return default

    def _transcript_lines(self, segments: list[dict]) -> list[str]:
        lines = []
        for seg in segments:
            ts = self._fmt_ts(seg.get("start", 0))
            speaker = self._pretty_speaker(seg.get("speaker", "UNKNOWN"))
            text = re.sub(r"\s+", " ", seg.get("text", "").strip())
            if len(text) > self.MAX_TRANSCRIPT_LINE_CHARS:
                text = text[: self.MAX_TRANSCRIPT_LINE_CHARS - 3].rstrip() + "..."
            if text:
                lines.append(f"[{ts}] {speaker}: {text}")
        return lines

    def _format_transcript(self, segments: list[dict]) -> str:
        return "\n".join(self._transcript_lines(segments))

    def _fmt_transcript_block(self, ctx: ReportContext) -> str:
        lines = self._transcript_lines(ctx.transcript_segments)
        if not lines:
            return ctx.transcript_text

        full_text = "\n".join(lines)
        if len(full_text) <= self.MAX_TRANSCRIPT_CHARS:
            return full_text

        sampled = lines[:]
        max_lines = min(self.MAX_TRANSCRIPT_LINES, len(lines))

        while max_lines >= 6:
            indices = sorted({
                round(i * (len(lines) - 1) / max(max_lines - 1, 1))
                for i in range(max_lines)
            })
            sampled = [lines[i] for i in indices]
            text = "\n".join(sampled)
            if len(text) <= self.MAX_TRANSCRIPT_CHARS:
                return text
            max_lines -= 1

        text = "\n".join(sampled)
        if len(text) > self.MAX_TRANSCRIPT_CHARS:
            text = text[: self.MAX_TRANSCRIPT_CHARS - 60].rstrip()
            text += "\n[... transcript sampled across the meeting for speed ...]"
        return text

    def _fmt_keyphrases(self, ctx: ReportContext) -> str:
        if not ctx.keyphrases:
            return "No keyphrases extracted."
        return "\n".join(
            f"  {i + 1:2d}. {kp['phrase']:35s} (score: {kp['score']:.3f})"
            for i, kp in enumerate(ctx.keyphrases[: self.MAX_KEYPHRASES])
        )

    def _fmt_entities(self, ctx: ReportContext) -> str:
        if not ctx.entities:
            return "No entities extracted."
        lines = []
        for etype, ents in ctx.entities.items():
            if ents:
                names = ", ".join(e["text"] for e in ents[: self.MAX_ENTITY_VALUES])
                lines.append(f"  {etype:10s}: {names}")
        return "\n".join(lines) if lines else "No entities extracted."

    def _fmt_actions(self, ctx: ReportContext) -> str:
        if not ctx.action_items:
            return "No action items detected."
        return "\n".join(
            f"  - {self._clean_sentence(item.get('action', 'Not specified'))}."
            for item in ctx.action_items[: self.MAX_ACTION_ITEMS]
        )

    def _fmt_decisions(self, ctx: ReportContext) -> str:
        if not ctx.decisions:
            return "No decisions detected."
        return "\n".join(
            f"  - {self._pretty_speaker(d.get('speaker', '?'))}: {d.get('decision', '?')}"
            for d in ctx.decisions[: self.MAX_DECISIONS]
        )

    def _fmt_pain_points(self, ctx: ReportContext) -> str:
        if not ctx.pain_points:
            return "No pain points detected."
        lines = []
        for pp in ctx.pain_points[: self.MAX_PAIN_POINTS]:
            lines.append(
                f"  - {pp.get('severity', '?').upper()} / {pp.get('category', '?')} / "
                f"{self._pretty_speaker(pp.get('speaker', '?'))}: {pp.get('pain_point', '?')}"
            )
        return "\n".join(lines)

    def _fmt_sentiment(self, ctx: ReportContext) -> str:
        lines = []
        overall = ctx.sentiment_overall
        if overall:
            lines.append(
                "  Overall: "
                f"Positive {overall.get('positive_pct', 0):.0%}  "
                f"Neutral {overall.get('neutral_pct', 0):.0%}  "
                f"Negative {overall.get('negative_pct', 0):.0%}"
            )
        if ctx.sentiment_per_speaker:
            ranked_speakers = sorted(
                ctx.sentiment_per_speaker.items(),
                key=lambda item: float(item[1].get("positive_pct", 0)),
                reverse=True,
            )[: self.MAX_SENTIMENT_SPEAKERS]
            lines.append(
                "  Positive drivers: "
                + ", ".join(
                    f"{self._pretty_speaker(speaker)} {stats.get('positive_pct', 0):.0%}"
                    for speaker, stats in ranked_speakers
                )
            )

        negative_spikes = [
            s
            for s in ctx.sentiment_per_segment
            if s.get("sentiment") == "negative"
            and s.get("scores", {}).get("negative", 0) > 0.75
        ]
        if negative_spikes:
            lines.append("  High-negativity moments:")
            for spike in negative_spikes[: self.MAX_NEGATIVE_SPIKES]:
                lines.append(
                    f"    [{self._fmt_ts(spike.get('start', 0))}] "
                    f"{self._pretty_speaker(spike.get('speaker', '?'))}: "
                    f"\"{spike.get('text', '')[: self.MAX_QUOTE_CHARS]}...\""
                )
        return "\n".join(lines) if lines else "No sentiment data."

    def _render_key_decisions(self, ctx: ReportContext) -> str:
        if not ctx.decisions:
            return "- No confirmed decisions detected."

        lines = []
        for item in ctx.decisions[: self.MAX_DECISIONS]:
            decision = self._clean_sentence(item.get("decision", "Not specified"))
            speaker = self._pretty_speaker(item.get("speaker", "Not specified"))
            lines.append(f"- {decision}. Proposed by {speaker}. Dissent: None.")
        return "\n".join(lines)

    def _render_pain_points(self, ctx: ReportContext) -> str:
        if not ctx.pain_points:
            return "- No pain points detected."

        lines = []
        for item in ctx.pain_points[: self.MAX_PAIN_POINTS]:
            pain_point = self._clean_sentence(item.get("pain_point", "Not specified"))
            lines.append(f"- {pain_point}.")
        return "\n".join(lines)

    def _render_action_items(self, ctx: ReportContext) -> str:
        if not ctx.action_items:
            return "- No action items detected."

        lines = []
        for item in ctx.action_items[: self.MAX_ACTION_ITEMS]:
            action = self._clean_sentence(item.get("action", "Not specified"))
            lines.append(f"- {action}.")
        return "\n".join(lines)

    @staticmethod
    def _infer_action_priority(deadline: Any, confidence: Any) -> str:
        due = str(deadline or "").lower()
        if any(token in due for token in ("today", "asap", "urgent", "end of day")):
            return "High"
        if any(token in due for token in ("tomorrow", "this week", "next week", "next meeting", "before")):
            return "Medium"
        try:
            return "Medium" if float(confidence or 0) >= 0.75 else "Low"
        except (TypeError, ValueError):
            return "Low"

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _match_section(self, heading_text: str) -> dict | None:
        normalized = self._normalize_heading(heading_text)
        if not normalized:
            return None

        for section in self.SECTION_DEFS:
            if normalized in section["aliases"]:
                return section
        return None

    @staticmethod
    def _normalize_heading(heading_text: str) -> str:
        heading_text = heading_text.strip()
        heading_text = re.sub(r"^[0-9]+[\.)]\s*", "", heading_text)
        heading_text = heading_text.replace("&", " and ")
        heading_text = re.sub(r"[-_]+", " ", heading_text)
        heading_text = re.sub(r"[^a-zA-Z0-9 ]+", "", heading_text)
        heading_text = re.sub(r"\s+", " ", heading_text)
        return heading_text.lower().strip()

    @staticmethod
    def _pretty_speaker(value: Any) -> str:
        text = str(value or "").strip()
        match = re.match(r"^SPEAKER[_\s-]?0*(\d+)$", text, flags=re.IGNORECASE)
        if match:
            return f"Speaker {int(match.group(1))}"
        return text or "Not specified"

    @staticmethod
    def _clean_sentence(value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        return text.rstrip(".") or "Not specified"

    def _pain_point_next_step(self, item: dict[str, Any]) -> str:
        category = str(item.get("category", "") or "").lower()
        if category == "resource_constraint":
            return "Clarify scope, budget, or staffing needed"
        if category == "process_inefficiency":
            return "Define a clear process owner and next milestone"
        if category == "external_dependency":
            return "Follow up with the external dependency and confirm timing"
        if category == "unclear_requirements":
            return "Resolve the open requirement or decision before the next meeting"
        if category == "team_communication":
            return "Align stakeholders and address the concern directly"
        if category == "technical_blocker":
            return "Assign an owner to investigate and unblock it"
        return "Assign an owner and confirm the next step"

    @staticmethod
    def _escape_table_cell(value: Any) -> str:
        return str(value or "Not specified").replace("\n", " ").replace("|", "/").strip() or "Not specified"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if not seconds:
            return "unknown"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"
