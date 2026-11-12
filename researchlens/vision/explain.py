"""Qualitative figures for the vision report (run in the same Colab session after training):

  attention_rollout.png  - where the ViT looks (attention rollout, Abnar & Zuidema 2020), correct + wrong cases
  misclassified.png      - grid of test figures the model got wrong

Example:
    python -m researchlens.vision.explain --results_dir /content/drive/MyDrive/ResearchLens/results/v2_figcls/vit_base/seed_42 \
        --model_dir /content/drive/MyDrive/ResearchLens/models/v2_figcls/vit_base/seed_42 --data_dir /content/data/spiqa_figs
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def attention_rollout(attentions) -> np.ndarray:
    """attentions: tuple(layers) of [1, heads, tokens, tokens] -> CLS-to-patch map [grid, grid]."""
    import torch

    result = None
    for a in attentions:
        a = a[0].mean(0)  # average heads
        a = a + torch.eye(a.size(-1), device=a.device)
        a = a / a.sum(-1, keepdim=True)
        result = a if result is None else a @ result
    cls = result[0, 1:]
    g = int(round(cls.numel() ** 0.5))
    m = cls[: g * g].reshape(g, g).float().cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from researchlens.evaluation.style import apply_style, save_fig
    from researchlens.vision.train_figcls import build_transforms

    apply_style()
    res, data = Path(args.results_dir), Path(args.data_dir)
    figdir = res / "figures"
    preds = [json.loads(l) for l in open(res / "predictions.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    wrong = [p for p in preds if p["gold"] != p["pred"]]
    right = [p for p in preds if p["gold"] == p["pred"]]

    # ---- misclassified grid ----
    sample = rng.sample(wrong, min(len(wrong), 12))
    if sample:
        cols = 4
        rows = int(np.ceil(len(sample) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, p in zip(np.atleast_1d(axes).ravel(), sample):
            ax.imshow(Image.open(data / p["path"]).convert("RGB"))
            ax.set_title(f"true: {p['gold']}\npred: {p['pred']}", fontsize=9)
        fig.suptitle("Misclassified test figures")
        save_fig(fig, figdir / "misclassified")

    # ---- attention rollout (ViT models only) ----
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir, attn_implementation="eager").eval()
    if model.config.model_type not in ("vit", "deit", "beit", "dinov2"):
        print(f"Attention rollout skipped for model_type={model.config.model_type}")
        return
    tf = build_transforms(processor, train=False)
    picks = rng.sample(right, min(len(right), args.n // 2)) + rng.sample(wrong, min(len(wrong), args.n // 2))
    fig, axes = plt.subplots(2, len(picks), figsize=(len(picks) * 2.6, 5.6))
    axes = np.array(axes).reshape(2, -1)
    for j, p in enumerate(picks):
        img = Image.open(data / p["path"])
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            out = model(pixel_values=x, output_attentions=True)
        heat = attention_rollout(out.attentions)
        shown = x[0].permute(1, 2, 0).numpy()
        shown = (shown - shown.min()) / (shown.max() - shown.min() + 1e-8)
        axes[0, j].imshow(shown)
        axes[0, j].set_title(f"true: {p['gold']}", fontsize=9)
        axes[1, j].imshow(shown)
        axes[1, j].imshow(np.array(Image.fromarray((heat * 255).astype("uint8")).resize(shown.shape[:2][::-1], Image.BILINEAR)),
                          cmap="inferno", alpha=0.55)
        axes[1, j].set_title(f"pred: {p['pred']}", fontsize=9)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle("ViT attention rollout (top: input, bottom: attention)")
    save_fig(fig, figdir / "attention_rollout")
    print(f"Saved to {figdir}")


if __name__ == "__main__":
    main()
