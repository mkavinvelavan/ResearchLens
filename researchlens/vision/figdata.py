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


def classification_metrics(y_true, probs: np.ndarray) -> dict:
    """The same test metrics for every figure classifier (so the report compares like with like)."""
    from sklearn.metrics import classification_report, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = probs.argmax(-1)
    rep = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = None
    return {
        "test_accuracy": rep["accuracy"],
        "test_f1_macro": rep["macro avg"]["f1-score"],
        "test_precision_macro": rep["macro avg"]["precision"],
        "test_recall_macro": rep["macro avg"]["recall"],
        "test_f1_weighted": rep["weighted avg"]["f1-score"],
        "test_roc_auc_macro_ovr": auc,
        "per_class": {c: {m: rep[c][m] for m in ("precision", "recall", "f1-score", "support")} for c in CLASSES},
        "classes": CLASSES,
    }
