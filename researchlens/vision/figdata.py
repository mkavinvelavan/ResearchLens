"""Shared data + metrics for the three figure-type classifiers (image, caption, image+caption).

All three read the same manifest.csv, so they train and test on exactly the same figures.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from researchlens.vision.prepare_spiqa_figures import CLASSES


def read_manifest(data_dir: Path) -> dict[str, list[dict]]:
    """{split: [{path, rel_path, label(int), caption, paper_id}]} in a fixed order."""
    data_dir = Path(data_dir)
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    with open(data_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            splits[r["split"]].append({
                "path": str(data_dir / r["path"]), "rel_path": r["path"], "label": CLASSES.index(r["label"]),
                "caption": (r.get("caption") or "").strip(), "paper_id": r.get("paper_id", ""),
            })
    print({k: len(v) for k, v in splits.items()})
    return splits


def build_transforms(processor, train: bool):
    """Pad figures to a white square (no distortion), resize; light augmentation, never flips."""
    from PIL import Image, ImageOps
    from torchvision import transforms as T

    def _size(key):
        sz = processor.size  # dict in transformers v4, SizeDict object in v5
        try:
            return sz.get(key) if isinstance(sz, dict) else getattr(sz, key, None)
        except Exception:
            return None

    size = _size("height") or _size("shortest_edge") or 224
    mean, std = processor.image_mean, processor.image_std

    def pad_to_square(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        s = max(w, h)
        return ImageOps.expand(img, ((s - w) // 2, (s - h) // 2, s - w - (s - w) // 2, s - h - (s - h) // 2), fill="white")

    ops = [T.Lambda(pad_to_square), T.Resize((size, size))]
    if train:
        ops += [T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)), T.ColorJitter(0.2, 0.2, 0.1)]
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def class_weights(train_items: list[dict]):
    import torch

    counts = np.bincount([it["label"] for it in train_items], minlength=len(CLASSES)).astype(float)
    return torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float)


def train_eval_subset(train_items: list[dict], n: int = 2000, seed: int = 0) -> list[dict]:
    """Fixed random slice of the training set, evaluated every epoch to measure over-fitting cheaply."""
    import random

    items = list(train_items)
    random.Random(seed).shuffle(items)
    return items[:n]


def epoch_metrics(y_true, y_pred) -> dict:
    """Metrics logged every epoch (train subset and validation)."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def classification_metrics(y_true, probs: np.ndarray, prefix: str = "test") -> dict:
    """Every classification metric for one split, keys prefixed (train_/val_/test_)."""
    from sklearn.metrics import (balanced_accuracy_score, classification_report, cohen_kappa_score, log_loss,
                                 roc_auc_score, top_k_accuracy_score)

    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(-1, keepdims=True)  # exact sum-to-1 (sklearn roc_auc is strict)
    y_pred = probs.argmax(-1)
    labels = list(range(len(CLASSES)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = None
    out = {
        "accuracy": rep["accuracy"],
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "top2_accuracy": top_k_accuracy_score(y_true, probs, k=2, labels=labels),
        "f1_macro": rep["macro avg"]["f1-score"],
        "precision_macro": rep["macro avg"]["precision"],
        "recall_macro": rep["macro avg"]["recall"],
        "f1_weighted": rep["weighted avg"]["f1-score"],
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc_macro_ovr": auc,
        "log_loss": log_loss(y_true, np.clip(probs, 1e-7, 1), labels=labels),
    }
    out = {f"{prefix}_{k}": v for k, v in out.items()}
    if prefix == "test":
        out["per_class"] = {c: {m: rep[c][m] for m in ("precision", "recall", "f1-score", "support")} for c in CLASSES}
        out["classes"] = CLASSES
    return out


SPLIT_KEYS = ["f1_macro", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "top2_accuracy",
              "cohen_kappa"]


def split_view(metrics: dict) -> dict:
    """{split: {metric: value}} for print_split_table."""
    return {s: {k: metrics.get(f"{s}_{k}") or 0.0 for k in SPLIT_KEYS} for s in ("train", "val", "test")
            if f"{s}_accuracy" in metrics}
