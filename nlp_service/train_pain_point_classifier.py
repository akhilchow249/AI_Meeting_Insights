"""
nlp-service/models/train_pain_point_classifier.py
───────────────────────────────────────────────────
Fine-tunes distilbert-base-uncased as a binary pain point classifier.

Usage
─────
1. Prepare training data in data/pain_points_train.jsonl:
   {"text": "We are blocked on the auth service.", "label": 1}
   {"text": "I think we should reconsider the approach.", "label": 0}

2. Prepare evaluation data in data/pain_points_eval.jsonl (never used for training):
   At least 200 manually labelled gold examples.

3. Run:
   python models/train_pain_point_classifier.py

4. Model saved to models/pain_point_classifier/
   pain_points.py loads it automatically on next container start.

Training data targets
─────────────────────
  2,000+ labelled pairs total
  Sources:
    - AMI Meeting Corpus annotations
    - ICSI Meeting Corpus
    - Synthetic GPT-4o / Claude generated examples (silver labels)
    - 200 manually labelled gold evaluation examples (held out)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-uncased"
MODEL_OUT  = Path(__file__).parent / "pain_point_classifier"
DATA_DIR   = Path(__file__).parent.parent / "training_data"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding=False,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="binary")
    report = classification_report(labels, predictions,
                                   target_names=["not_pain_point", "pain_point"],
                                   output_dict=True)
    return {
        "f1": f1,
        "precision_pain": report["pain_point"]["precision"],
        "recall_pain":    report["pain_point"]["recall"],
    }


def train():
    logger.info("Loading training data…")
    train_data = load_jsonl(DATA_DIR / "pain_points_train.jsonl")
    eval_data  = load_jsonl(DATA_DIR / "pain_points_eval.jsonl")

    train_ds = Dataset.from_list(train_data)
    eval_ds  = Dataset.from_list(eval_data)

    logger.info("Train: %d samples, Eval: %d samples", len(train_ds), len(eval_ds))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2,
        id2label={0: "not_pain_point", 1: "pain_point"},
        label2id={"not_pain_point": 0, "pain_point": 1},
    )

    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    eval_ds  = eval_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(MODEL_OUT / "logs"),
        logging_steps=50,
        fp16=True,           # use float16 on GPU — safe on RTX 3050
        report_to="none",    # disable wandb
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training…")
    trainer.train()

    logger.info("Saving model to %s", MODEL_OUT)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    logger.info("Training complete.")


if __name__ == "__main__":
    train()
