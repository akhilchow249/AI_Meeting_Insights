"""
train_classifiers.py
────────────────────
Fine-tunes three BERT-based classifiers on the datasets produced by
generate_training_data.py, following the spec in Project-4.pdf page 9.

  Classifier 1 — Pain Point       (distilbert-base-uncased  base)
  Classifier 2 — Action Item      (distilbert-base-uncased  base)
  Classifier 3 — Decision         (distilbert-base-uncased  base)

After training, each model is saved to ./models/<name>/ and evaluated
against the gold-label eval split. A confusion matrix + classification
report is printed and saved as PNG.

Usage:
  pip install transformers datasets torch scikit-learn matplotlib seaborn
  python train_classifiers.py
"""

import importlib.metadata as importlib_metadata
import inspect
import json, os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from datasets import Dataset, DatasetDict, ClassLabel
from packaging.version import Version
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "training_data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BASE_MODEL = "distilbert-base-uncased"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
PAIN_POINT_DATA_FILE = DATA_DIR / "pain_point_combined_data.jsonl"
if not PAIN_POINT_DATA_FILE.exists():
    PAIN_POINT_DATA_FILE = DATA_DIR / "pain_point_data.jsonl"

CLASSIFIERS = [
    {
        "name":      "pain_point",
        "data_file": PAIN_POINT_DATA_FILE,
        "labels":    ["not_pain_point", "pain_point"],
        "threshold": 0.75,   # minimum confidence to surface result in UI
    },
    {
        "name":      "action_item",
        "data_file": DATA_DIR / "action_item_data.jsonl",
        "labels":    ["not_action_item", "action_item"],
        "threshold": 0.70,
    },
    {
        "name":      "decision",
        "data_file": DATA_DIR / "decision_data.jsonl",
        "labels":    ["not_decision", "decision"],
        "threshold": 0.70,
    },
]

TRAINING_ARGS = dict(
    num_train_epochs         = 5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate            = 2e-5,
    weight_decay             = 0.01,
    warmup_ratio             = 0.1,
    eval_strategy            = "epoch",
    save_strategy            = "epoch",
    load_best_model_at_end   = True,
    metric_for_best_model    = "f1",
    greater_is_better        = True,
    logging_steps            = 20,
    report_to                = "none",
    fp16                     = torch.cuda.is_available(),
    seed                     = 42,
)


def build_training_arguments(output_dir: Path) -> TrainingArguments:
    kwargs = dict(TRAINING_ARGS)
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(output_dir=str(output_dir), **kwargs)


def validate_training_runtime(script_name: str) -> None:
    """Fail fast with a clear message when the HF training stack is mismatched."""
    transformers_version = importlib_metadata.version("transformers")
    try:
        accelerate_version = importlib_metadata.version("accelerate")
    except importlib_metadata.PackageNotFoundError:
        accelerate_version = None

    if Version(transformers_version) < Version("5.0.0"):
        return

    if accelerate_version is None or Version(accelerate_version) < Version("1.1.0"):
        installed = accelerate_version or "not installed"
        raise RuntimeError(
            f"{script_name} detected transformers {transformers_version} with accelerate {installed}. "
            "Trainer support in transformers>=5.0 requires accelerate>=1.1.0. "
            "Use the project .venv or install matching versions before retraining."
        )


def resolve_base_model() -> str:
    cache_root = HF_CACHE_DIR / "models--distilbert-base-uncased" / "snapshots"
    if cache_root.exists():
        snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return BASE_MODEL


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(records: list[dict], dataset_name: str) -> DatasetDict:
    """Build train/eval datasets from explicit splits or a deterministic fallback."""
    train = [r for r in records if r.get("split") == "train"]
    eval_ = [r for r in records if r.get("split") == "eval"]
    has_explicit_splits = any("split" in r for r in records)

    if has_explicit_splits:
        if not train or not eval_:
            split_counts = Counter(r.get("split", "<missing>") for r in records)
            raise ValueError(
                f"{dataset_name} has invalid split metadata: {dict(split_counts)}. "
                "Expected both 'train' and 'eval' rows. Regenerate this dataset before training."
            )
    else:
        labels = [r["label"] for r in records]
        stratify = labels if len(set(labels)) > 1 else None
        train, eval_ = train_test_split(
            records,
            test_size=0.1,
            random_state=42,
            stratify=stratify,
        )

    def to_hf(rows):
        return Dataset.from_dict({
            "text":  [r["text"]  for r in rows],
            "label": [r["label"] for r in rows],
        })

    return DatasetDict({"train": to_hf(train), "test": to_hf(eval_)})


# ─────────────────────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────────────────────

def tokenise_dataset(ds: DatasetDict, tokenizer) -> DatasetDict:
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )
    return ds.map(tok, batched=True, remove_columns=["text"])


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1":        f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, average="binary", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names: list[str], save_path: Path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {save_path.stem}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


# ─────────────────────────────────────────────────────────────
# TRAIN ONE CLASSIFIER
# ─────────────────────────────────────────────────────────────

def train_classifier(cfg: dict):
    name      = cfg["name"]
    labels    = cfg["labels"]
    data_file = cfg["data_file"]
    out_dir   = MODEL_DIR / name
    base_model = resolve_base_model()

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    # Load data
    records = load_jsonl(data_file)
    label_dist = Counter(r["label"] for r in records)
    print(f"  Label distribution: {dict(label_dist)}")

    ds        = build_dataset(records, name)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=Path(base_model).exists())
    tok_ds    = tokenise_dataset(ds, tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
        local_files_only=Path(base_model).exists(),
    )

    # Training args
    args = build_training_arguments(out_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save best model
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))
    print(f"  Model saved -> {out_dir / 'best'}")

    # Evaluate
    print(f"\n  Evaluation on gold eval split ({len(tok_ds['test'])} examples):")
    preds_output = trainer.predict(tok_ds["test"])
    preds  = np.argmax(preds_output.predictions, axis=-1)
    gold   = preds_output.label_ids

    report = classification_report(
        gold,
        preds,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    print(report)

    # Save report
    report_path = out_dir / "eval_report.txt"
    report_path.write_text(report)

    # Confusion matrix
    plot_confusion_matrix(
        gold, preds,
        class_names=labels,
        save_path=out_dir / "confusion_matrix.png",
    )

    # Threshold calibration note
    print(f"  UI confidence threshold: >= {cfg['threshold']}")
    probs = torch.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()
    above = (probs[:, 1] >= cfg["threshold"]).sum()
    print(f"  Examples above threshold: {above}/{len(preds)} "
          f"({100*above/len(preds):.1f}%)")

    return {"name": name, "report": report}


# ─────────────────────────────────────────────────────────────
# INFERENCE HELPER  (import this in your nlp-service)
# ─────────────────────────────────────────────────────────────

class MeetingClassifier:
    """
    Thin wrapper around a saved classifier.

    Usage:
        clf = MeetingClassifier(str(MODEL_DIR / "pain_point" / "best"), threshold=0.75)
        result = clf.predict("We are blocked on the auth service.")
        # {'label': 'pain_point', 'score': 0.92, 'above_threshold': True}
    """

    def __init__(self, model_path: str, threshold: float = 0.75):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.threshold = threshold

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs     = torch.softmax(logits, dim=-1)[0].tolist()
        label_idx = int(torch.argmax(logits))
        label_str = self.model.config.id2label[label_idx]
        score     = probs[label_idx]
        return {
            "label":           label_str,
            "score":           round(score, 4),
            "above_threshold": score >= self.threshold,
            "all_probs":       {
                self.model.config.id2label[i]: round(p, 4)
                for i, p in enumerate(probs)
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validate_training_runtime("train_classifiers.py")
    results = []
    for cfg in CLASSIFIERS:
        if not cfg["data_file"].exists():
            print(f"  [SKIP] {cfg['name']} — data file not found: {cfg['data_file']}")
            print(f"         Run generate_training_data.py first.")
            continue
        results.append(train_classifier(cfg))

    print("\n" + "="*60)
    print("  Training complete. Summary:")
    for r in results:
        print(f"    {r['name']}: model saved to models/{r['name']}/best/")
    print("="*60)
