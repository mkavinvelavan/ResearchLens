"""Generic plotting + aggregation used by every model's report.

All functions read the standard result files (metrics.json, history.csv, predictions.jsonl),
so graphs can be redrawn at any time without retraining.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .style import INK_2, PALETTE, apply_style, save_fig, seq_cmap


# ----------------------------------------------------------------------------- loading
def load_run(seed_dir: Path) -> dict:
    seed_dir = Path(seed_dir)
    run = {"dir": seed_dir, "metrics": {}, "history": pd.DataFrame(), "predictions": []}
    if (seed_dir / "metrics.json").exists():
        run["metrics"] = json.loads((seed_dir / "metrics.json").read_text())
    if (seed_dir / "history.csv").exists():
        run["history"] = pd.read_csv(seed_dir / "history.csv")
    if (seed_dir / "predictions.jsonl").exists():
        with open(seed_dir / "predictions.jsonl", encoding="utf-8") as f:
            run["predictions"] = [json.loads(l) for l in f]
    if (seed_dir / "embeddings.npy").exists():
        run["embeddings"] = np.load(seed_dir / "embeddings.npy")
    return run


def find_runs(results_root: Path, model_tag: str) -> dict[str, list[dict]]:
    """{run_name: [run(seed) ...]} for every finished run (one with metrics.json)."""
    out: dict[str, list[dict]] = {}
    base = Path(results_root) / model_tag
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "report"):
        seeds = [load_run(s) for s in sorted(run_dir.glob("seed_*")) if (s / "metrics.json").exists()]
        if seeds:
            out[run_dir.name] = seeds
    return out


def summary_table(runs: dict[str, list[dict]], keys: list[str]) -> pd.DataFrame:
    """Mean ± std across seeds for each run and metric."""
    rows = []
    for name, seeds in runs.items():
        row = {"run": name, "n_seeds": len(seeds)}
        for k in keys:
            vals = [s["metrics"].get(k) for s in seeds if s["metrics"].get(k) is not None]
            if vals:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        for k in ("total_params", "trainable_params", "train_seconds", "test_samples_per_second", "gpu", "peak_gpu_mem_gb"):
            row[k] = seeds[0]["metrics"].get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def formatted_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Report-ready table with 'mean ± std' strings (as %)."""
    out = pd.DataFrame({"Run": df["run"], "Seeds": df["n_seeds"]})
    for k in keys:
        if f"{k}_mean" in df:
            out[k] = [f"{m*100:.2f} ± {s*100:.2f}" for m, s in zip(df[f"{k}_mean"], df[f"{k}_std"])]
    return out


# ----------------------------------------------------------------------------- training curves
def _train_eval(h: pd.DataFrame):
    train = h[h["loss"].notna()] if "loss" in h else h.iloc[0:0]
    ev = h[h["eval_loss"].notna()] if "eval_loss" in h else h.iloc[0:0]
    return train, ev


def plot_loss_curves(seeds: list[dict], title: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, s in enumerate(seeds):
        train, ev = _train_eval(s["history"])
        lab = i == 0
        ax.plot(train["epoch"], train["loss"], color=PALETTE[0], alpha=0.9 if lab else 0.35, lw=1.6,
                label="Training loss" if lab else None)
        ax.plot(ev["epoch"], ev["eval_loss"], color=PALETTE[1], alpha=0.9 if lab else 0.35, marker="o",
                label="Validation loss" if lab else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title + (f"  ({len(seeds)} seeds)" if len(seeds) > 1 else ""))
    ax.legend()
    save_fig(fig, out)


def plot_val_metric(seeds: list[dict], metric: str, label: str, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    col = f"eval_{metric}"
    for i, s in enumerate(seeds):
        _, ev = _train_eval(s["history"])
        if col not in ev:
            continue
        seed = s["metrics"].get("seed", i)
        ax.plot(ev["epoch"], ev[col] * 100, marker="o", color=PALETTE[i % len(PALETTE)], label=f"seed {seed}")
        b = ev[col].idxmax()
        ax.scatter([ev.loc[b, "epoch"]], [ev.loc[b, col] * 100], s=110, facecolors="none",
                   edgecolors=PALETTE[i % len(PALETTE)], linewidths=2, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Validation {label} (%)")
    ax.set_title(title + "  (circle = best checkpoint)")
    if len(seeds) > 1:
        ax.legend()
    save_fig(fig, out)


def plot_lr(seed: dict, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    h = seed["history"]
    if "learning_rate" not in h:
        return
    h = h[h["learning_rate"].notna()]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h["step"], h["learning_rate"], color=PALETTE[0])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(title)
    save_fig(fig, out)


# ----------------------------------------------------------------------------- classification plots
def plot_confusion(y_true, y_pred, labels: list[str], title: str, out: Path, normalize: bool = True) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    apply_style()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.5, 0.9 * n + 2))
    im = ax.imshow(disp, cmap=seq_cmap(), vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    ax.set_xticks(range(n), labels, rotation=40, ha="right")
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            v = disp[i, j]
            txt = f"{v:.2f}" if normalize else str(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (v > 0.55 if normalize else v > disp.max() * 0.55) else "#0b0b0b")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, out)


def plot_per_class(seeds: list[dict], classes: list[str], title: str, out: Path) -> pd.DataFrame:
    """Grouped bars of precision / recall / F1 per class (mean over seeds, std as error bar)."""
    import matplotlib.pyplot as plt

    apply_style()
    rows = []
    for c in classes:
        for m in ("precision", "recall", "f1-score"):
            vals = [s["metrics"]["per_class"][c][m] for s in seeds if c in s["metrics"].get("per_class", {})]
            if vals:
                rows.append({"class": c, "metric": m, "mean": np.mean(vals), "std": np.std(vals, ddof=1) if len(vals) > 1 else 0})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(classes)), 3.8))
    w = 0.26
    x = np.arange(len(classes))
    for k, (m, name) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1")]):
        d = df[df.metric == m].set_index("class").reindex(classes)
        ax.bar(x + (k - 1) * w, d["mean"] * 100, w * 0.92, yerr=d["std"] * 100, color=PALETTE[k],
               label=name, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK_2})
    ax.set_xticks(x, classes, rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title, pad=30)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    save_fig(fig, out)
    return df


def plot_run_comparison(table: pd.DataFrame, keys: list[str], names: dict[str, str], title: str, out: Path) -> None:
    """Baseline vs fine-tuned: bars per run, one colour per metric, std error bars."""
    import matplotlib.pyplot as plt

    apply_style()
    runs = list(table["run"])
    x = np.arange(len(runs))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.6 * len(runs)), 3.8))
    for k, key in enumerate(keys):
        if f"{key}_mean" not in table:
            continue
        m, s = table[f"{key}_mean"] * 100, table[f"{key}_std"] * 100
        bars = ax.bar(x + (k - (len(keys) - 1) / 2) * w, m, w * 0.92, yerr=s, color=PALETTE[k], label=names.get(key, key),
                      capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK_2})
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(x, runs)
    ax.set_ylabel("Test score (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title, pad=30)
    ax.legend(ncol=len(keys), loc="lower left", bbox_to_anchor=(0, 1.0))
    save_fig(fig, out)


def plot_roc_pr(y_true_idx: np.ndarray, probs: np.ndarray, classes: list[str], title: str, out: Path) -> None:
    """One-vs-rest ROC and precision-recall curves, side by side."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, c in enumerate(classes):
        y = (y_true_idx == i).astype(int)
        if y.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, probs[:, i])
        a1.plot(fpr, tpr, color=PALETTE[i], label=f"{c} (AUC {auc(fpr, tpr):.2f})")
        pr, rc, _ = precision_recall_curve(y, probs[:, i])
        a2.plot(rc, pr, color=PALETTE[i], label=f"{c} (AP {average_precision_score(y, probs[:, i]):.2f})")
    a1.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (one-vs-rest)")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (one-vs-rest)")
    for a in (a1, a2):
        a.grid(True, axis="both")
        a.legend(fontsize=8)
    fig.suptitle(title)
    save_fig(fig, out)


def plot_embedding_2d(emb: np.ndarray, labels: list[str], classes: list[str], title: str, out: Path, seed: int = 0) -> None:
    """t-SNE of learned embeddings coloured by true class."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    apply_style()
    n = min(len(emb), 3000)
    idx = np.random.default_rng(seed).choice(len(emb), n, replace=False)
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(emb[idx])
    lab = np.array(labels)[idx]
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(classes):
        m = lab == c
        ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.75, color=PALETTE[i], label=c, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title)
    ax.legend(markerscale=2.5, fontsize=8, loc="best")
    save_fig(fig, out)


def plot_score_by_bucket(df: pd.DataFrame, bucket_col: str, score_col: str, xlabel: str, title: str, out: Path) -> None:
    """Score (%) per bucket, e.g. F1 vs entity length or accuracy vs formula length."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(df[bucket_col].astype(str), df[score_col] * 100, color=PALETTE[0], width=0.6)
    if "n" in df:
        ax.bar_label(bars, labels=[f"n={v}" for v in df["n"]], fontsize=8, color=INK_2, padding=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    save_fig(fig, out)
