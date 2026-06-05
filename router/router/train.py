"""Fine-tune DistilBERT for intent classification.

Run with:

    python -m router.train

Device is detected automatically (CUDA → MPS → CPU). Training time depends
on hardware; measure it on your machine. Output: model + tokenizer saved to
`router/model/`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .dataset import load_dataset_splits
from .intents import ID2LABEL, INTENTS, LABEL2ID

ROUTER_PKG = Path(__file__).resolve().parent
MODEL_DIR = ROUTER_PKG.parent / "model"
CHECKPOINT_DIR = ROUTER_PKG.parent / "checkpoints"

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 64


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _compute_metrics_fn():
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1_score(labels, preds, average="macro")
        per_class = f1_score(labels, preds, average=None, labels=list(range(len(INTENTS))))
        out = {"accuracy": acc, "f1": f1_macro}
        for cls_idx, score in enumerate(per_class):
            out[f"f1_{ID2LABEL[cls_idx]}"] = float(score)
        return out

    return compute_metrics


def train(
    output_dir: Path = MODEL_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    num_train_epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
) -> Path:
    device = _detect_device()
    print(f"[train] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(INTENTS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    splits = load_dataset_splits()

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        enc["labels"] = [LABEL2ID[lbl] for lbl in batch["label"]]
        return enc

    tokenized = splits.map(tokenize, batched=True, remove_columns=["text", "label"])

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
        save_total_limit=2,
        fp16=(device == "cuda"),
    )

    # transformers 5.x renamed Trainer's `tokenizer` kwarg to `processing_class`.
    # We pass it both ways to stay compatible with the 4.x pin as well.
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics_fn(),
    )
    import inspect as _inspect

    _trainer_params = _inspect.signature(Trainer.__init__).parameters
    if "processing_class" in _trainer_params:
        trainer_kwargs["processing_class"] = tokenizer  # transformers >= 5
    else:
        trainer_kwargs["tokenizer"] = tokenizer  # transformers < 5
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # Final classification report on the held-out test split.
    preds = trainer.predict(tokenized["test"])
    y_pred = preds.predictions.argmax(axis=-1)
    y_true = preds.label_ids
    print("\n=== Final classification report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=INTENTS,
            digits=4,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n[train] saved model + tokenizer to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT intent router")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    # Be friendly when run inside CI / containers.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
