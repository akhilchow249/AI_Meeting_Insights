"""
Build labelled pain-point training data from the public AMI and ICSI
meeting corpora after they have been unpacked under tmp/meeting_corpora/.

Label policy:
  - Positive (label=1): a dialogue act linked to at least one abstractive
    summary sentence in the "problems" section.
  - Negative (label=0): a dialogue act linked only to non-problem sections
    such as abstract, actions, decisions, or progress.

Outputs:
  - training_data/pain_point_corpus_data.jsonl
  - training_data/pain_point_combined_data.jsonl
  - training_data/pain_points_train.jsonl
  - training_data/pain_points_eval.jsonl
"""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

NITE_ID = "{http://nite.sourceforge.net/}id"
NITE_NS = {"nite": "http://nite.sourceforge.net/"}
REF_RE = re.compile(r"([^#]+)#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")
NO_SPACE_BEFORE = {
    ".",
    ",",
    "?",
    "!",
    ":",
    ";",
    "%",
    ")",
    "]",
    "}",
    "-",
    "'s",
    "'re",
    "'ve",
    "'ll",
    "'d",
    "'m",
    "n't",
}
FILLER_ONLY = {"uh", "um", "mm", "mmm", "hmm", "yeah", "okay", "right", "oh"}

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / "tmp" / "meeting_corpora"
TRAINING_DIR = Path(__file__).resolve().parent / "training_data"

AMI_ROOT = TMP_DIR / "ami"
ICSI_ROOT = TMP_DIR / "icsi" / "ICSIplus"

CORPUS_FILE = TRAINING_DIR / "pain_point_corpus_data.jsonl"
COMBINED_FILE = TRAINING_DIR / "pain_point_combined_data.jsonl"
TRAIN_FILE = TRAINING_DIR / "pain_points_train.jsonl"
EVAL_FILE = TRAINING_DIR / "pain_points_eval.jsonl"
EXISTING_FILE = TRAINING_DIR / "pain_point_data.jsonl"

SEED = 42
EVAL_FRACTION = 0.15
NEGATIVE_RATIO = 2.0
MIN_CHARACTERS = 25
MIN_ALPHA_WORDS = 5
MANUAL_TRAIN_POSITIVES = [
    "There's been an actual problem with car parking ever since Orchard and Shipman moved into the new offices at the back.",
    "Sue has identified a growing problem with staff morale.",
    "There is a concern in some departments that there is low staff morale.",
    "This is indicated by the recent low sales figures and also increased sickness absence in some of the departments.",
    "There seems to be a significant problem with staff morale in the sales team and also sickness absence company-wide.",
    "There is definitely low morale in the company in certain departments.",
    "The recent launch of Comet software has caused problems because people have not had enough training and are making mistakes.",
    "There are issues like lack of training and lack of effective appraisals.",
    "Some of the issues are to do with the recent restructuring, the job losses, poor management in some departments and lack of training on the new software.",
]
MANUAL_TRAIN_NEGATIVES = [
    "Thanks for coming to today's monthly meeting.",
    "Let's start with apologies for absence.",
    "I'll start.",
    "Oh yeah, I'm Lucy Strokes, PA to Rita.",
    "Okay, on to the next item.",
    "Please, carry on.",
    "Jason, you need to park by the garages.",
    "Let's all come up with four and email them over in the next two days and I'll take it from there.",
    "If no one's listening I'll send it.",
    "What does everyone think about that?",
    "Rita, Rita, client meeting.",
]


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    root: Path
    abstractive_dir: Path
    summlink_dir: Path
    dialogue_dir: Path
    words_dir: Path


AMI = CorpusConfig(
    name="ami",
    root=AMI_ROOT,
    abstractive_dir=AMI_ROOT / "abstractive",
    summlink_dir=AMI_ROOT / "extractive",
    dialogue_dir=AMI_ROOT / "dialogueActs",
    words_dir=AMI_ROOT / "words",
)

ICSI = CorpusConfig(
    name="icsi",
    root=ICSI_ROOT,
    abstractive_dir=ICSI_ROOT / "Contributions" / "Summarization" / "abstractive",
    summlink_dir=ICSI_ROOT / "Contributions" / "Summarization" / "extractive",
    dialogue_dir=ICSI_ROOT / "DialogueActs",
    words_dir=ICSI_ROOT / "Words",
)


def parse_href(href: str) -> tuple[str, str, str] | None:
    match = REF_RE.search(href)
    if not match:
        return None
    filename = Path(match.group(1)).name
    start_id = match.group(2)
    end_id = match.group(3) or start_id
    return filename, start_id, end_id


def join_tokens(tokens: list[tuple[str, bool]]) -> str:
    text = ""
    for token, is_punct in tokens:
        if not token:
            continue
        if not text:
            text = token
        elif is_punct:
            text += token
        else:
            text += f" {token}"
    return re.sub(r"\s+", " ", text).strip()


def normalise_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -")
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def alpha_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def is_high_quality(text: str) -> bool:
    if len(text) < MIN_CHARACTERS:
        return False
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < MIN_ALPHA_WORDS:
        return False
    if all(word.lower() in FILLER_ONLY for word in words):
        return False
    return True


@lru_cache(maxsize=None)
def load_xml_order(path_str: str) -> tuple[list[str], dict[str, int], dict[str, ET.Element]]:
    root = ET.parse(path_str).getroot()
    items = [elem for elem in root if elem.attrib.get(NITE_ID)]
    order = [elem.attrib[NITE_ID] for elem in items]
    index = {item_id: i for i, item_id in enumerate(order)}
    element_map = {elem.attrib[NITE_ID]: elem for elem in items}
    return order, index, element_map


@lru_cache(maxsize=None)
def load_dialogue_map(path_str: str) -> dict[str, ET.Element]:
    root = ET.parse(path_str).getroot()
    return {elem.attrib[NITE_ID]: elem for elem in root if elem.attrib.get(NITE_ID)}


def extract_dialogue_text(config: CorpusConfig, dialogue_file: str, dialogue_id: str) -> str:
    dialogue_path = config.dialogue_dir / dialogue_file
    dialogue_map = load_dialogue_map(str(dialogue_path))
    dialogue = dialogue_map.get(dialogue_id)
    if dialogue is None:
        return ""

    tokens: list[tuple[str, bool]] = []
    for child in dialogue.findall("nite:child", NITE_NS):
        parsed = parse_href(child.attrib.get("href", ""))
        if not parsed:
            continue
        words_file, start_id, end_id = parsed
        words_path = config.words_dir / words_file
        order, index, element_map = load_xml_order(str(words_path))
        if start_id not in index or end_id not in index:
            continue

        for item_id in order[index[start_id] : index[end_id] + 1]:
            node = element_map[item_id]
            if node.tag.split("}")[-1] != "w":
                continue
            token = (node.text or "").strip()
            if not token:
                continue
            token_type = node.attrib.get("c", "")
            is_punct = (
                node.attrib.get("punc") == "true"
                or token in NO_SPACE_BEFORE
                or token_type in {".", "CM", "QM", "EX", "COLON", "SCOLON", "HYPH"}
            )
            tokens.append((token, is_punct))

    return normalise_text(join_tokens(tokens))


def section_lookup(summary_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    root = ET.parse(summary_path).getroot()
    for section in root:
        section_name = section.tag.split("}")[-1]
        for sentence in section.findall("sentence"):
            sentence_id = sentence.attrib.get(NITE_ID)
            if sentence_id:
                lookup[sentence_id] = section_name
    return lookup


def iter_summlinks(summlink_path: Path) -> list[tuple[str, str, str]]:
    links: list[tuple[str, str, str]] = []
    root = ET.parse(summlink_path).getroot()
    for link in root:
        extractive_file = ""
        extractive_id = ""
        abstractive_id = ""
        for pointer in link.findall("nite:pointer", NITE_NS):
            parsed = parse_href(pointer.attrib.get("href", ""))
            if not parsed:
                continue
            filename, start_id, _ = parsed
            if pointer.attrib.get("role") == "extractive":
                extractive_file = filename
                extractive_id = start_id
            elif pointer.attrib.get("role") == "abstractive":
                abstractive_id = start_id
        if extractive_file and extractive_id and abstractive_id:
            links.append((extractive_file, extractive_id, abstractive_id))
    return links


def extract_corpus_records(config: CorpusConfig) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"sections": set(), "summary_sentence_ids": set()}
    )

    for summary_path in sorted(config.abstractive_dir.glob("*.xml")):
        meeting_id = summary_path.stem.split(".")[0]
        sentence_to_section = section_lookup(summary_path)
        summlink_path = config.summlink_dir / summary_path.name.replace("abssumm", "summlink")
        if not summlink_path.exists():
            continue

        for extractive_file, extractive_id, abstractive_id in iter_summlinks(summlink_path):
            section = sentence_to_section.get(abstractive_id)
            if not section:
                continue
            key = (meeting_id, extractive_file, extractive_id)
            grouped[key]["sections"].add(section)
            grouped[key]["summary_sentence_ids"].add(abstractive_id)

    records: list[dict] = []
    for (meeting_id, dialogue_file, dialogue_id), meta in grouped.items():
        text = extract_dialogue_text(config, dialogue_file, dialogue_id)
        if not is_high_quality(text):
            continue
        speaker = dialogue_file.split(".")[1] if "." in dialogue_file else ""
        sections = sorted(meta["sections"])
        label = 1 if "problems" in meta["sections"] else 0
        records.append(
            {
                "text": text,
                "label": label,
                "split": "",
                "source": config.name,
                "label_source": "summlink_problems" if label == 1 else "summlink_nonproblem",
                "meeting_id": meeting_id,
                "speaker": speaker,
                "dialogue_file": dialogue_file,
                "dialogue_act_id": dialogue_id,
                "summary_sections": sections,
                "summary_sentence_ids": sorted(meta["summary_sentence_ids"]),
            }
        )

    return records


def assign_meeting_splits(records: list[dict]) -> None:
    rng = random.Random(SEED)
    for source in sorted({record["source"] for record in records}):
        meetings = sorted({record["meeting_id"] for record in records if record["source"] == source})
        rng.shuffle(meetings)
        eval_count = max(1, round(len(meetings) * EVAL_FRACTION))
        eval_meetings = set(meetings[:eval_count])
        for record in records:
            if record["source"] == source:
                record["split"] = "eval" if record["meeting_id"] in eval_meetings else "train"


def balanced_records(records: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    chosen: list[dict] = []
    for split in ("train", "eval"):
        for source in ("ami", "icsi"):
            subset = [r for r in records if r["split"] == split and r["source"] == source]
            positives = [r for r in subset if r["label"] == 1]
            negatives = [r for r in subset if r["label"] == 0]
            target_negatives = min(len(negatives), int(len(positives) * NEGATIVE_RATIO))
            if target_negatives < len(negatives):
                negatives = rng.sample(negatives, target_negatives)
            chosen.extend(positives)
            chosen.extend(negatives)
    rng.shuffle(chosen)
    return chosen


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_labels(records: list[dict]) -> dict[str, dict[int, int]]:
    summary: dict[str, dict[int, int]] = {}
    for split in ("train", "eval"):
        split_records = [record for record in records if record.get("split") == split]
        summary[split] = dict(Counter(record["label"] for record in split_records))
    return summary


def manual_records() -> list[dict]:
    positives = [
        {
            "text": text,
            "label": 1,
            "split": "train",
            "source": "manual_hard_positive",
            "label_source": "manual_positive",
            "meeting_id": "manual",
        }
        for text in MANUAL_TRAIN_POSITIVES
    ]
    negatives = [
        {
            "text": text,
            "label": 0,
            "split": "train",
            "source": "manual_hard_negative",
            "label_source": "manual_negative",
            "meeting_id": "manual",
        }
        for text in MANUAL_TRAIN_NEGATIVES
    ]
    return positives + negatives


def main() -> None:
    missing = [path for path in (AMI_ROOT, ICSI_ROOT) if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Expected unpacked corpora under tmp/meeting_corpora/, but these paths are missing: "
            f"{missing_paths}"
        )

    corpus_records = extract_corpus_records(AMI) + extract_corpus_records(ICSI)
    assign_meeting_splits(corpus_records)
    corpus_records = balanced_records(corpus_records)

    existing_records = load_jsonl(EXISTING_FILE)
    combined_records = existing_records + corpus_records + manual_records()

    train_records = [
        {"text": record["text"], "label": record["label"]}
        for record in combined_records
        if record.get("split") == "train"
    ]
    eval_records = [
        {"text": record["text"], "label": record["label"]}
        for record in combined_records
        if record.get("split") == "eval"
    ]

    save_jsonl(CORPUS_FILE, corpus_records)
    save_jsonl(COMBINED_FILE, combined_records)
    save_jsonl(TRAIN_FILE, train_records)
    save_jsonl(EVAL_FILE, eval_records)

    print("Saved corpus-only data to", CORPUS_FILE)
    print("Saved combined data to", COMBINED_FILE)
    print("Saved train split to", TRAIN_FILE)
    print("Saved eval split to", EVAL_FILE)
    print("Corpus label counts by split:", count_labels(corpus_records))
    print("Combined label counts by split:", count_labels(combined_records))
    print("Combined train/eval sizes:", len(train_records), len(eval_records))


if __name__ == "__main__":
    main()
