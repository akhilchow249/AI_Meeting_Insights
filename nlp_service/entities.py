"""
nlp-service/entities.py
────────────────────────
Named Entity Recognition using spaCy en_core_web_trf (transformer-based).

Extracts and deduplicates entities of interest:
  PERSON   — meeting participants, names mentioned
  ORG      — companies, teams, departments
  PRODUCT  — software, tools, services mentioned
  DATE     — explicit dates and deadlines
  GPE      — geopolitical entities (countries, cities)
  EVENT    — named events (sprints, releases, conferences)

Output schema
─────────────
{
  "PERSON":  [{"text": "Alice", "count": 3, "first_seen_at": 12.4}, ...],
  "ORG":     [...],
  "PRODUCT": [...],
  "DATE":    [...],
  "GPE":     [...],
  "EVENT":   [...]
}
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

ENTITY_TYPES = {"PERSON", "ORG", "PRODUCT", "DATE", "GPE", "EVENT"}
ROLE_RE = re.compile(
    r"\b(?:i(?:'m| am)|this is)\s+([A-Z][a-z]+)\b.*?\b(head of|chief|director of|lead of|work in)\s+([A-Z][A-Za-z ]+)",
    re.IGNORECASE,
)
NAME_RE = re.compile(r"\b(?:i(?:'m| am)|this is)\s+([A-Z][a-z]+)\b", re.IGNORECASE)
DATE_HINT_RE = re.compile(
    r"\b(today|tomorrow|next meeting|next week|next sprint|fortnight|fortnight's time|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
EVENT_HINT_RE = re.compile(
    r"\b(next meeting|staff meeting|board presentation|presentation|release|sprint|meeting)\b",
    re.IGNORECASE,
)

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        model = "en_core_web_trf"
        try:
            _nlp = spacy.load(model)
        except OSError:
            logger.info("Downloading spaCy model %s…", model)
            from spacy.cli import download
            download(model)
            _nlp = spacy.load(model)
        # Disable unused pipeline components for speed
        _nlp.select_pipes(enable=["transformer", "ner"])
    return _nlp


# ─── EntityExtractor ──────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Runs spaCy en_core_web_trf NER over all transcript segments and
    returns a deduplicated, frequency-sorted entity dictionary.

    Design note: we process segments individually (not the full transcript
    as one string) so that each entity can be linked back to its speaker
    and approximate timestamp.
    """

    # spaCy trf has a 512 token limit — split long segments
    MAX_CHARS_PER_CHUNK = 2000

    def extract(self, segments: list[dict]) -> dict[str, list[dict]]:
        """
        Parameters
        ----------
        segments : list[dict]
            Diarised transcript segments.  Each must have at minimum:
            {"speaker": str, "start": float, "text": str}

        Returns
        -------
        dict[entity_type, list[entity_record]]
            Sorted by frequency descending.
        """
        nlp = _get_nlp()

        # entity_type → canonical_text → {count, first_seen_at, speakers}
        registry: dict[str, dict[str, dict]] = defaultdict(dict)

        for segment in segments:
            text      = segment.get("text", "").strip()
            speaker   = segment.get("speaker", "UNKNOWN")
            timestamp = segment.get("start", 0.0)

            if not text:
                continue

            # Process in chunks if segment is very long
            chunks = self._chunk(text)
            for chunk in chunks:
                doc = nlp(chunk)
                for ent in doc.ents:
                    if ent.label_ not in ENTITY_TYPES:
                        continue

                    canonical = self._canonicalise(ent.text)
                    if not canonical or len(canonical) < 2:
                        continue

                    etype = ent.label_
                    if canonical not in registry[etype]:
                        registry[etype][canonical] = {
                            "text":         canonical,
                            "count":        0,
                            "first_seen_at": round(timestamp, 2),
                            "speakers":     set(),
                        }
                    registry[etype][canonical]["count"] += 1
                    registry[etype][canonical]["speakers"].add(speaker)

            for etype, canonical in self._rule_based_entities(text):
                if canonical not in registry[etype]:
                    registry[etype][canonical] = {
                        "text": canonical,
                        "count": 0,
                        "first_seen_at": round(timestamp, 2),
                        "speakers": set(),
                    }
                registry[etype][canonical]["count"] += 1
                registry[etype][canonical]["speakers"].add(speaker)

        # Convert sets → sorted lists, sort by frequency
        result: dict[str, list[dict]] = {}
        for etype in ENTITY_TYPES:
            records = list(registry[etype].values())
            for r in records:
                r["speakers"] = sorted(r["speakers"])
            records.sort(key=lambda r: -r["count"])
            result[etype] = records

        total = sum(len(v) for v in result.values())
        logger.info("NER complete: %d entities across %d types.", total, len(ENTITY_TYPES))
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> list[str]:
        if len(text) <= self.MAX_CHARS_PER_CHUNK:
            return [text]
        # Split on sentence boundaries
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) > self.MAX_CHARS_PER_CHUNK:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks or [text]

    @staticmethod
    def _canonicalise(text: str) -> str:
        """Normalise entity text: title-case, strip extra whitespace."""
        return " ".join(text.split()).title()

    def _rule_based_entities(self, text: str) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []

        for match in ROLE_RE.finditer(text):
            person = self._canonicalise(match.group(1))
            dept = self._canonicalise(match.group(3))
            if person:
                found.append(("PERSON", person))
            if dept:
                found.append(("ORG", dept))

        for match in NAME_RE.finditer(text):
            person = self._canonicalise(match.group(1))
            if person:
                found.append(("PERSON", person))

        for match in DATE_HINT_RE.finditer(text):
            found.append(("DATE", self._canonicalise(match.group(1))))

        for match in EVENT_HINT_RE.finditer(text):
            found.append(("EVENT", self._canonicalise(match.group(1))))

        return found
