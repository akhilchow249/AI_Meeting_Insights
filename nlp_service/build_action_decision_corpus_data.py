"""
Build action-item and decision training data from AMI summary annotations.

This module uses three sources:
  1. Held-out gold seed JSONL files for evaluation.
  2. AMI abstractive and participant summary annotations for train rows.
  3. Deterministic synthetic templates to broaden wording coverage.

Outputs are written to nlp_service/training_data/action_item_data.jsonl and
nlp_service/training_data/decision_data.jsonl.
"""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AMI_ROOT = ROOT / "tmp" / "meeting_corpora" / "ami"
TRAINING_DIR = Path(__file__).resolve().parent / "training_data"

ACTION_SEED_FILE = TRAINING_DIR / "action_item_seed.jsonl"
DECISION_SEED_FILE = TRAINING_DIR / "decision_seed.jsonl"
ACTION_OUT_FILE = TRAINING_DIR / "action_item_data.jsonl"
DECISION_OUT_FILE = TRAINING_DIR / "decision_data.jsonl"

SEED = 42
MIN_CHARACTERS = 18
MIN_ALPHA_WORDS = 4
DROP_TEXTS = {"*na*", "n/a", "na"}
NITE_ID = "{http://nite.sourceforge.net/}id"

SUMMARY_DIRS = (
    ("ami_abstractive", AMI_ROOT / "abstractive"),
    ("ami_participant", AMI_ROOT / "participantSummaries"),
)

ACTION_POSITIVE_SECTIONS = {"actions", "participant_actions"}
ACTION_NEGATIVE_SECTIONS = {
    "abstract",
    "participant_abstract",
    "decisions",
    "participant_decisions",
    "problems",
    "participant_problems",
}
DECISION_POSITIVE_SECTIONS = {"decisions", "participant_decisions"}
DECISION_NEGATIVE_SECTIONS = {
    "abstract",
    "participant_abstract",
    "actions",
    "participant_actions",
    "problems",
    "participant_problems",
}

ACTION_TARGET_POSITIVE = 900
ACTION_TARGET_NEGATIVE = 1800
DECISION_TARGET_POSITIVE = 1000
DECISION_TARGET_NEGATIVE = 1400
ACTION_SYNTHETIC_PER_LABEL = 260
DECISION_SYNTHETIC_PER_LABEL = 140

DECISION_SIGNAL_PATTERNS = [
    re.compile(r"\bwe(?:'ve| have)? decided\b", re.IGNORECASE),
    re.compile(r"\bgoing forward\b", re.IGNORECASE),
    re.compile(r"\bthe team (?:agrees|has agreed)\b", re.IGNORECASE),
    re.compile(r"\bit has been decided\b", re.IGNORECASE),
    re.compile(r"\b(?:we|the team) (?:have|has) resolved\b", re.IGNORECASE),
    re.compile(r"\bleadership has approved\b", re.IGNORECASE),
    re.compile(r"\beffective immediately\b", re.IGNORECASE),
    re.compile(r"\bwe will not\b", re.IGNORECASE),
    re.compile(r"\bofficially\b", re.IGNORECASE),
]

ACTION_OWNER_PATTERN = re.compile(
    r"^(?P<owner>(?:[Tt]he )?[A-Za-z][A-Za-z /-]+?)\s+"
    r"(?:will|should|needs to|need to|must|can|could|has to|have to)\b"
)
ACTION_DEADLINE_PATTERN = re.compile(
    r"\b(?:by|before|after|on|this|next|tomorrow|today|tonight|"
    r"end of day|end of the sprint|end of sprint)\b[^.?!;]*",
    re.IGNORECASE,
)
ACTION_WEAK_POSITIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*i(?:'ll| will)\s+start\b",
        r"\bapologies? for absence\b",
        r"\bintroduce (?:myself|ourselves)\b",
        r"\bnext item\b",
        r"\bitem on the agenda\b",
        r"\bget back to the agenda\b",
        r"\bplease,\s*carry on\b",
        r"\bclient meeting\b",
        r"\blet me just have a look\b",
        r"\bwhat do you think\b",
    )
]
MANUAL_ACTION_POSITIVES = [
    "We need to send an email letting all the staff know there are only five spaces that belong to us.",
    "Jason, you need to park by the garages.",
    "And I'll coordinate letting the next priority member know if anyone is off sick.",
    "Oh, can you put a sign up on all the spaces, Jason?",
    "Oh, OK, why don't I get a list and then I'll circulate it so there's no confusion.",
    "Let's all come up with four and email them over in the next two days and I'll take it from there.",
    "I'll speak with Clive and let you know dates.",
    "And as far as those issues are concerned, you must make sure they're scheduled for the next meeting.",
]
MANUAL_ACTION_NEGATIVES = [
    "Let's start with apologies for absence.",
    "I'll start.",
    "Oh yeah, I'm Lucy Strokes, PA to Rita.",
    "Okay, on to the next item.",
    "Well, next...",
    "Please, carry on.",
    "Please, can we get back?",
    "Rita, Rita, client meeting.",
    "Yeah, can we please get back to the agenda item?",
    "What does everyone think about that?",
    "Okay, so what has this got to do with staff morale?",
    "Let me just have a look.",
]


def _text_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalise_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text


def _is_usable_text(text: str) -> bool:
    if len(text) < MIN_CHARACTERS:
        return False
    if _text_key(text).strip(". ") in DROP_TEXTS:
        return False
    words = re.findall(r"[A-Za-z']+", text)
    return len(words) >= MIN_ALPHA_WORDS


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_summary_records() -> list[dict]:
    records: list[dict] = []
    for source, directory in SUMMARY_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.xml")):
            meeting_id = path.stem.split(".")[0]
            root = ET.parse(path).getroot()
            for child in root:
                section = child.tag.split("}")[-1]
                sentence_nodes = list(child.findall("sentence")) + list(child.findall("sent"))
                for sentence in sentence_nodes:
                    parts = [part.strip() for part in sentence.itertext() if part and part.strip()]
                    text = _normalise_text(" ".join(parts))
                    if not _is_usable_text(text):
                        continue
                    records.append(
                        {
                            "text": text,
                            "meeting_id": meeting_id,
                            "section": section,
                            "source": source,
                            "sentence_id": sentence.attrib.get(NITE_ID, ""),
                        }
                    )
    return records


def _extract_action_owner(text: str) -> str | None:
    if re.match(r"^(?:can|could) you\b", text, re.IGNORECASE):
        return "implied"
    if re.match(r"^(?:please|let's|lets|we need)\b", text, re.IGNORECASE):
        return "implied"
    match = re.match(r"^(?P<owner>[A-Z][A-Za-z]+),", text)
    if match:
        return match.group("owner")
    match = ACTION_OWNER_PATTERN.match(text)
    if match:
        return match.group("owner").strip()
    return None


def _extract_deadline(text: str) -> str | None:
    match = ACTION_DEADLINE_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).strip(" ,.")


def _extract_decision_signal(text: str) -> str | None:
    for pattern in DECISION_SIGNAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).lower()
    return None


def _is_weak_action_positive(text: str) -> bool:
    return any(pattern.search(text) for pattern in ACTION_WEAK_POSITIVE_PATTERNS)


def _manual_action_records() -> tuple[list[dict], list[dict]]:
    positives = [
        {
            "text": text,
            "label": 1,
            "split": "train",
            "source": "manual_hard_positive",
            "label_source": "manual_action_item",
            "owner": _extract_action_owner(text),
            "deadline": _extract_deadline(text),
        }
        for text in MANUAL_ACTION_POSITIVES
    ]
    negatives = [
        {
            "text": text,
            "label": 0,
            "split": "train",
            "source": "manual_hard_negative",
            "label_source": "manual_action_item",
            "owner": None,
            "deadline": None,
        }
        for text in MANUAL_ACTION_NEGATIVES
    ]
    return positives, negatives


def _build_summary_records(summary_records: list[dict], positive_sections: set[str], negative_sections: set[str], task: str) -> tuple[list[dict], list[dict]]:
    positives: list[dict] = []
    negatives: list[dict] = []
    for record in summary_records:
        if record["section"] in positive_sections:
            label = 1
        elif record["section"] in negative_sections:
            label = 0
        else:
            continue

        base = {
            "text": record["text"],
            "label": label,
            "split": "train",
            "source": record["source"],
            "label_source": record["section"],
            "meeting_id": record["meeting_id"],
            "summary_section": record["section"],
            "summary_sentence_id": record["sentence_id"],
        }
        if task == "action_item":
            base["owner"] = _extract_action_owner(record["text"])
            base["deadline"] = _extract_deadline(record["text"])
            if label == 1 and _is_weak_action_positive(record["text"]):
                label = 0
                base["label"] = 0
                base["label_source"] = f"{record['section']}_relabelled_negative"
        else:
            base["speaker_signal"] = _extract_decision_signal(record["text"])

        if label == 1:
            positives.append(base)
        else:
            negatives.append(base)
    return positives, negatives


def _make_action_synthetic_records() -> tuple[list[dict], list[dict]]:
    owners = [
        "Alice",
        "Bob",
        "Priya",
        "the backend team",
        "the QA team",
        "the design team",
        "the platform team",
        "someone from product ops",
        "the incident lead",
        "marketing",
    ]
    tasks = [
        "finalize the API integration",
        "prepare the regression test plan",
        "send the updated spec to the vendor",
        "review the pull request and merge it",
        "document the new authentication flow",
        "update the roadmap deck",
        "set up the Grafana dashboard",
        "run the load tests against staging",
        "share the rollout checklist with stakeholders",
        "confirm the migration window with operations",
    ]
    deadlines = [
        "by Friday",
        "before next week",
        "by end of day",
        "before Thursday's release",
        "this week",
        "by Monday",
        "before the end of the sprint",
        "ahead of next Tuesday",
        "before sprint planning",
        "tomorrow morning",
    ]
    topics = [
        "the authentication flow",
        "the migration plan",
        "release timing",
        "the vendor contract",
        "roadmap priorities",
        "load testing coverage",
        "API versioning",
        "dashboard metrics",
        "customer onboarding",
        "staffing plans",
    ]
    decisions = [
        "the release should move to Q4",
        "Postgres is the right default database",
        "the pilot should start next month",
        "the vendor should be replaced",
        "the new design direction makes sense",
        "the feature freeze should happen earlier",
        "we should adopt blue-green deployment",
        "the API should remain backwards compatible",
        "the budget increase is justified",
        "the incident process needs tighter ownership",
    ]

    positive_templates = [
        "{owner}, please {task} {deadline}.",
        "We need {owner} to {task} {deadline}.",
        "Can {owner} {task} {deadline}?",
    ]
    negative_templates = [
        "We discussed {topic} in detail during the meeting.",
        "The team agreed that {decision}.",
        "We are still reviewing {topic}.",
        "The update on {topic} went well last week.",
    ]

    positives: list[dict] = []
    negatives: list[dict] = []

    positive_seen: set[str] = set()
    negative_seen: set[str] = set()

    for template in positive_templates:
        for owner_index, owner in enumerate(owners):
            for task_index, task in enumerate(tasks):
                deadline = deadlines[(owner_index + task_index) % len(deadlines)]
                text = template.format(owner=owner, task=task, deadline=deadline)
                key = _text_key(text)
                if key in positive_seen:
                    continue
                positive_seen.add(key)
                positives.append(
                    {
                        "text": text,
                        "label": 1,
                        "split": "train",
                        "source": "synthetic_template",
                        "label_source": "synthetic_action_item",
                        "owner": owner,
                        "deadline": deadline,
                    }
                )
                if len(positives) >= ACTION_SYNTHETIC_PER_LABEL:
                    break
            if len(positives) >= ACTION_SYNTHETIC_PER_LABEL:
                break
        if len(positives) >= ACTION_SYNTHETIC_PER_LABEL:
            break

    for template in negative_templates:
        for topic_index, topic in enumerate(topics):
            for decision_index, decision in enumerate(decisions):
                decision_text = decisions[(topic_index + decision_index) % len(decisions)]
                text = template.format(topic=topic, decision=decision_text)
                key = _text_key(text)
                if key in negative_seen:
                    continue
                negative_seen.add(key)
                negatives.append(
                    {
                        "text": text,
                        "label": 0,
                        "split": "train",
                        "source": "synthetic_template",
                        "label_source": "synthetic_action_item",
                        "owner": None,
                        "deadline": None,
                    }
                )
                if len(negatives) >= ACTION_SYNTHETIC_PER_LABEL:
                    break
            if len(negatives) >= ACTION_SYNTHETIC_PER_LABEL:
                break
        if len(negatives) >= ACTION_SYNTHETIC_PER_LABEL:
            break

    return positives, negatives


def _make_decision_synthetic_records() -> tuple[list[dict], list[dict]]:
    action_clauses = [
        "migrate the reporting service to Postgres next quarter",
        "retire the legacy auth gateway before the summer release",
        "move all deploys to blue-green rollout",
        "standardize on typed API contracts for new services",
        "freeze feature work two weeks before launch",
        "deprecate the v1 mobile endpoint this quarter",
        "staff the on-call rotation with two engineers every weekend",
        "ship the audit trail in the first phase of the rollout",
        "drop long-lived feature branches across the platform team",
        "require design review before new integrations are implemented",
    ]
    noun_clauses = [
        "a shift to quarterly roadmap checkpoints",
        "the extra budget for observability tooling",
        "the transition to a managed Kafka cluster",
        "the revised release calendar",
        "the hiring plan for two senior backend engineers",
        "the proposal to simplify the support workflow",
    ]
    open_clauses = [
        "migrate the reporting service to Postgres next quarter",
        "retire the legacy auth gateway before the summer release",
        "move all deploys to blue-green rollout",
        "standardize on typed API contracts for new services",
        "freeze feature work two weeks before launch",
        "deprecate the v1 mobile endpoint this quarter",
        "shift the onboarding experiment to mobile first",
        "replace the vendor with an internal build",
    ]
    open_options = [
        "a managed Kafka cluster",
        "a lighter approval process",
        "a new mobile onboarding flow",
        "a different incident severity model",
        "a quarterly release cadence",
        "an internal vendor replacement",
        "a smaller beta program",
        "a revised staffing plan",
    ]

    positive_templates = [
        "We have decided to {clause}.",
        "Going forward we will {clause}.",
        "The team agrees that we will {clause}.",
        "It has been decided that we will {clause}.",
        "Effective immediately we will {clause}.",
    ]
    approval_templates = [
        "Leadership has approved {clause}.",
        "We are officially moving ahead with {clause}.",
    ]
    negative_templates = [
        "We are still evaluating whether to {clause}.",
        "The team is leaning toward {option}, but no final call has been made.",
        "Someone suggested we might {clause} if the budget allows.",
        "We need more information before deciding whether to {clause}.",
    ]

    positives: list[dict] = []
    negatives: list[dict] = []

    for template in positive_templates:
        for clause in action_clauses:
            text = template.format(clause=clause)
            positives.append(
                {
                    "text": text,
                    "label": 1,
                    "split": "train",
                    "source": "synthetic_template",
                    "label_source": "synthetic_decision",
                    "speaker_signal": _extract_decision_signal(text),
                }
            )
            if len(positives) >= DECISION_SYNTHETIC_PER_LABEL - len(approval_templates):
                break
        if len(positives) >= DECISION_SYNTHETIC_PER_LABEL - len(approval_templates):
            break

    for template in approval_templates:
        for clause in noun_clauses:
            text = template.format(clause=clause)
            positives.append(
                {
                    "text": text,
                    "label": 1,
                    "split": "train",
                    "source": "synthetic_template",
                    "label_source": "synthetic_decision",
                    "speaker_signal": _extract_decision_signal(text),
                }
            )
            if len(positives) >= DECISION_SYNTHETIC_PER_LABEL:
                break
        if len(positives) >= DECISION_SYNTHETIC_PER_LABEL:
            break

    for template in negative_templates:
        if "{option}" in template:
            source_items = open_options
            field_name = "option"
        else:
            source_items = open_clauses
            field_name = "clause"

        for item in source_items:
            text = template.format(**{field_name: item})
            negatives.append(
                {
                    "text": text,
                    "label": 0,
                    "split": "train",
                    "source": "synthetic_template",
                    "label_source": "synthetic_decision",
                    "speaker_signal": None,
                }
            )
            if len(negatives) >= DECISION_SYNTHETIC_PER_LABEL:
                break
        if len(negatives) >= DECISION_SYNTHETIC_PER_LABEL:
            break

    return positives, negatives


def _dedupe_records(records: list[dict], blocked_keys: set[str] | None = None) -> list[dict]:
    blocked_keys = blocked_keys or set()
    deduped: list[dict] = []
    seen: set[str] = set()
    for record in records:
        key = _text_key(record["text"])
        if key in blocked_keys or key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _drop_ambiguous_records(positives: list[dict], negatives: list[dict]) -> tuple[list[dict], list[dict]]:
    positive_keys = {_text_key(record["text"]) for record in positives}
    negative_keys = {_text_key(record["text"]) for record in negatives}
    ambiguous = positive_keys & negative_keys
    if not ambiguous:
        return positives, negatives
    positives = [record for record in positives if _text_key(record["text"]) not in ambiguous]
    negatives = [record for record in negatives if _text_key(record["text"]) not in ambiguous]
    return positives, negatives


def _select_train_records(records: list[dict], target_count: int, prefer_source: str | None = None) -> list[dict]:
    rng = random.Random(SEED)
    records = list(records)
    preferred = [record for record in records if record.get("source") == prefer_source] if prefer_source else []
    remainder = [record for record in records if record.get("source") != prefer_source] if prefer_source else records
    rng.shuffle(preferred)
    rng.shuffle(remainder)
    selected = preferred[:target_count]
    remaining = target_count - len(selected)
    if remaining > 0:
        selected.extend(remainder[:remaining])
    if len(selected) < target_count:
        raise RuntimeError(
            f"Needed {target_count} train rows but only found {len(selected)} after filtering."
        )
    rng.shuffle(selected)
    return selected


def _assemble_dataset(
    *,
    task_name: str,
    seed_path: Path,
    summary_positive: list[dict],
    summary_negative: list[dict],
    synthetic_positive: list[dict],
    synthetic_negative: list[dict],
    positive_target_count: int,
    negative_target_count: int,
) -> list[dict]:
    seed_records = _load_jsonl(seed_path)
    eval_keys = {_text_key(record["text"]) for record in seed_records}

    positives = _dedupe_records(summary_positive + synthetic_positive, blocked_keys=eval_keys)
    negatives = _dedupe_records(summary_negative + synthetic_negative, blocked_keys=eval_keys)
    positives, negatives = _drop_ambiguous_records(positives, negatives)

    if not positives or not negatives:
        raise RuntimeError(f"{task_name} dataset build found no usable train rows.")

    train_positive = _select_train_records(positives, positive_target_count, prefer_source="manual_hard_positive")
    train_negative = _select_train_records(negatives, negative_target_count, prefer_source="manual_hard_negative")

    all_records = train_positive + train_negative + seed_records
    rng = random.Random(SEED)
    rng.shuffle(all_records)
    return all_records


def build_action_item_dataset(
    positive_target_count: int = ACTION_TARGET_POSITIVE,
    negative_target_count: int = ACTION_TARGET_NEGATIVE,
) -> list[dict]:
    summary_records = _iter_summary_records()
    summary_positive, summary_negative = _build_summary_records(
        summary_records,
        ACTION_POSITIVE_SECTIONS,
        ACTION_NEGATIVE_SECTIONS,
        task="action_item",
    )
    synthetic_positive, synthetic_negative = _make_action_synthetic_records()
    manual_positive, manual_negative = _manual_action_records()
    return _assemble_dataset(
        task_name="action_item",
        seed_path=ACTION_SEED_FILE,
        summary_positive=summary_positive + manual_positive,
        summary_negative=summary_negative + manual_negative,
        synthetic_positive=synthetic_positive,
        synthetic_negative=synthetic_negative,
        positive_target_count=positive_target_count,
        negative_target_count=negative_target_count,
    )


def build_decision_dataset(
    positive_target_count: int = DECISION_TARGET_POSITIVE,
    negative_target_count: int = DECISION_TARGET_NEGATIVE,
) -> list[dict]:
    summary_records = _iter_summary_records()
    summary_positive, summary_negative = _build_summary_records(
        summary_records,
        DECISION_POSITIVE_SECTIONS,
        DECISION_NEGATIVE_SECTIONS,
        task="decision",
    )
    synthetic_positive, synthetic_negative = _make_decision_synthetic_records()
    return _assemble_dataset(
        task_name="decision",
        seed_path=DECISION_SEED_FILE,
        summary_positive=summary_positive,
        summary_negative=summary_negative,
        synthetic_positive=synthetic_positive,
        synthetic_negative=synthetic_negative,
        positive_target_count=positive_target_count,
        negative_target_count=negative_target_count,
    )


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarise_records(name: str, records: list[dict]) -> None:
    labels = Counter(record["label"] for record in records)
    splits = Counter(record.get("split", "") for record in records)
    sources = Counter(record.get("source", "") for record in records)
    print(f"{name}: total={len(records)} labels={dict(labels)} splits={dict(splits)}")
    print(f"{name}: top sources={dict(sources.most_common(5))}")


def main() -> None:
    if not AMI_ROOT.exists():
        raise FileNotFoundError(
            f"AMI corpus not found at {AMI_ROOT}. Extract the corpus under tmp/meeting_corpora/ami first."
        )

    action_records = build_action_item_dataset()
    decision_records = build_decision_dataset()

    save_jsonl(action_records, ACTION_OUT_FILE)
    save_jsonl(decision_records, DECISION_OUT_FILE)

    summarise_records("action_item", action_records)
    summarise_records("decision", decision_records)
    print(f"Wrote {ACTION_OUT_FILE}")
    print(f"Wrote {DECISION_OUT_FILE}")


if __name__ == "__main__":
    main()
