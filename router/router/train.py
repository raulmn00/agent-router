"""Fine-tune DistilBERT for intent classification.

Run with:

    python -m router.train

Device is detected automatically (CUDA → MPS → CPU). Training time depends
on hardware; measure it on your machine. Output:

  - `router/model/`        — trained weights + tokenizer (gitignored)
  - `router/results/`      — evaluation evidence (committed to the repo)
      ├── confusion_matrix.png        rendered with sklearn ConfusionMatrixDisplay
      ├── confusion_matrix.csv        same matrix in text form
      ├── classification_report.json  per-class precision/recall/F1 + macro/weighted
      ├── classification_report.txt   human-readable version of the same
      └── training_meta.json          timestamp, device, sizes, hyperparams, base model

The artifacts under `results/` are deliberately version-controlled so a reader
can verify the training actually happened with the stated config.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import evaluate
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
RESULTS_DIR = ROUTER_PKG.parent / "results"

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


# --------------------------------------------------------------------------- #
# Evaluation-evidence helpers — pure functions, tested with mocked predictions #
# --------------------------------------------------------------------------- #


def save_evaluation_artifacts(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    results_dir: Path = RESULTS_DIR,
    intent_names: list[str] = INTENTS,
) -> dict[str, Path]:
    """Write the confusion matrix (PNG + CSV) and classification report (JSON + TXT).

    Returns a dict of artifact name → path on disk. Pure: no training, no
    network. Inputs are the raw predictions from `Trainer.predict`.
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    label_ids = list(range(len(intent_names)))

    results_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # --- confusion matrix (numeric) ---
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=label_ids)

    # CSV: rows = true intent, cols = predicted intent
    csv_path = results_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + intent_names)
        for i, row in enumerate(cm):
            writer.writerow([intent_names[i]] + [int(v) for v in row])
    paths["confusion_matrix_csv"] = csv_path

    # PNG rendered with sklearn's ConfusionMatrixDisplay (lazy matplotlib import).
    png_path = results_dir / "confusion_matrix.png"
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless — works in CI / Colab / Cloud Run
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=intent_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title("Confusion matrix — held-out test split")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths["confusion_matrix_png"] = png_path
    except ImportError:
        # matplotlib not installed — CSV still gets written; only the image
        # is skipped. Loud warning in the terminal so a CI run notices.
        print(
            "[train] WARNING: matplotlib is not installed; "
            f"skipped writing {png_path}. Install matplotlib to enable."
        )

    # --- classification report (dict + string) ---
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=label_ids,
        target_names=intent_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=label_ids,
        target_names=intent_names,
        digits=4,
        zero_division=0,
    )

    json_path = results_dir / "classification_report.json"
    json_path.write_text(json.dumps(report_dict, indent=2, sort_keys=True))
    paths["classification_report_json"] = json_path

    txt_path = results_dir / "classification_report.txt"
    txt_path.write_text(report_str)
    paths["classification_report_txt"] = txt_path

    return paths


def save_training_metadata(
    metadata: dict[str, Any],
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Persist the conditions of this training run for auditability.

    Caller passes a fully-built dict — we just enforce JSON-friendly serialization
    and the file location.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "training_meta.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    return path


# --------------------------------------------------------------------------- #
# Training entrypoint                                                          #
# --------------------------------------------------------------------------- #


def train(
    output_dir: Path = MODEL_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    num_train_epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    results_dir: Path = RESULTS_DIR,
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

    # Evidence artifacts: confusion matrix + classification report + metadata.
    # These get committed to the repo so the README's evaluation table is
    # backed by a regeneratable, auditable record of the actual training run.
    artifact_paths = save_evaluation_artifacts(
        y_true=y_true, y_pred=y_pred, results_dir=results_dir, intent_names=INTENTS,
    )

    # Pull final metrics straight from the predictions instead of re-computing.
    final_metrics = preds.metrics if hasattr(preds, "metrics") else {}
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "base_model": BASE_MODEL,
        "num_train_examples": len(tokenized["train"]),
        "num_test_examples": len(tokenized["test"]),
        "num_labels": len(INTENTS),
        "intents": list(INTENTS),
        "max_length": MAX_LENGTH,
        "hyperparameters": {
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "fp16": device == "cuda",
        },
        "final_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                          for k, v in final_metrics.items()},
        "framework_versions": {
            "torch": torch.__version__,
            "transformers": _safe_pkg_version("transformers"),
            "sklearn": _safe_pkg_version("sklearn"),
        },
    }
    meta_path = save_training_metadata(metadata, results_dir=results_dir)
    artifact_paths["training_meta"] = meta_path

    # Summary so the operator sees exactly where each piece of evidence landed.
    print("\n[train] evaluation evidence written to {}/".format(results_dir))
    for label, path in artifact_paths.items():
        print(f"        {label:30s} -> {path.relative_to(ROUTER_PKG.parent)}")

    return output_dir


def _safe_pkg_version(name: str) -> str:
    """Best-effort version lookup; metadata file should never crash the run."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # pragma: no cover — defensive
        return "unknown"


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
