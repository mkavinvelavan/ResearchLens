"""One place for every model's results: the summary table, over-fitting curves and the special tables
(cross-domain before/after, relation types, evidence ranking vs word overlap, formula reading).

Reads the standard result folders results/<tag>/<run>/seed_42/{metrics.json, history.csv}, e.g. after
    huggingface_hub.snapshot_download("<user>/researchlens-results", repo_type="dataset")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .plots import eval_frame, load_run, plot_loss_curves, plot_split_gap, plot_train_vs_val

# (tag, run) -> (side, readable name, used in app?)
KNOWN = {
    ("n1_ner", "scibert"): ("NLP", "SciBERT NER (AI only, SciERC)", "baseline"),
    ("n1_ner", "bert_base"): ("NLP", "BERT-base NER (SciERC)", "baseline"),
    ("n1_ner", "scibert_noreg"): ("NLP", "SciBERT NER, no regularisation", "baseline"),
    ("n1_ner", "scibert_multidomain"): ("NLP", "SciBERT NER, multi-domain (SciERC + ScienceIE)", "app"),
    ("n2_relation", "scibert_relation"): ("NLP", "SciBERT relation extraction (SciERC)", "app"),
    ("n3_evidence", "scibert_evidence"): ("NLP", "SciBERT evidence finder (QASPER)", "app"),
    ("v2_figcls", "scibert_caption"): ("NLP", "SciBERT caption -> figure type (SPIQA)", "app"),
    ("v2_figcls", "vit_base"): ("Vision", "ViT figure type (SPIQA, CS only)", "baseline"),
    ("v2_figcls", "resnet50"): ("Vision", "ResNet-50 figure type (SPIQA)", "baseline"),
    ("v2_figcls", "vit_multidomain"): ("Vision", "ViT figure type, multi-domain (SPIQA + BIOMEDICA)", "app"),
    ("v2_figcls", "fusion_vit_scibert"): ("Multimodal", "Gated fusion ViT + SciBERT (SPIQA)", "baseline"),
    ("v3_formula", "trocr_small"): ("Vision", "TrOCR formula -> LaTeX (im2latex-100k)", "app"),
}
MAIN = [  # (history metric, metrics.json key, label) - first one found is the model's main score
    ("f1", "f1_micro", "F1"), ("f1", "f1", "F1"), ("macro_f1", "f1_macro", "macro-F1"),
    ("exact_match", "bleu4", "BLEU-4"), ("exact_match", "exact_match", "exact match"),
]


def discover(root: Path) -> list[dict]:
    root = Path(root)
    runs = []
    for seed_dir in sorted(root.glob("*/*/seed_*")):
        if not (seed_dir / "metrics.json").exists():
            continue
        tag, run = seed_dir.parts[-3], seed_dir.parts[-2]
        r = load_run(seed_dir)
        r.update(tag=tag, run=run)
        side, name, use = KNOWN.get((tag, run), ("?", f"{tag}/{run}", "-"))
        r.update(side=side, name=name, use=use)
        runs.append(r)
    return runs


def main_metric(r: dict):
    m = r["metrics"]
    for hist, key, label in MAIN:
        if f"test_{key}" in m:
            return hist, key, label
    return None, None, None


def summary(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        if r["tag"].endswith("crossdomain"):
            continue
        hist, key, label = main_metric(r)
        m = r["metrics"]
        if key is None:
            continue
        row = {"side": r["side"], "model": r["name"], "role": r["use"], "metric": label,
               "train": m.get(f"train_{key}"), "val": m.get(f"val_{key}"), "test": m.get(f"test_{key}")}
        if row["train"] is not None and row["test"] is not None:
            row["gap (train-test)"] = row["train"] - row["test"]
        row.update({"best epoch": m.get("best_epoch"), "epochs run": m.get("epochs_trained"),
                    "params (M)": round((m.get("total_params") or 0) / 1e6, 1) or None,
                    "train time (min)": round((m.get("train_seconds") or 0) / 60, 1) or None})
        rows.append(row)
    df = pd.DataFrame(rows)
    order = {"NLP": 0, "Vision": 1, "Multimodal": 2}
    return df.sort_values(["side", "role"], key=lambda s: s.map(order) if s.name == "side" else s).reset_index(drop=True)


def pct(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("train", "val", "test", "gap (train-test)"):
        if c in out:
            out[c] = out[c].map(lambda v: None if v is None or pd.isna(v) else round(v * 100, 1))
    return out


def curves(runs: list[dict], out_dir: Path) -> list[Path]:
    """Loss curves + training-vs-validation curves + train/val/test bars for every run."""
    out_dir = Path(out_dir)
    made = []
    for r in runs:
        if r["history"].empty:
            continue
        hist, key, label = main_metric(r)
        stem = f"{r['tag']}__{r['run']}"
        plot_loss_curves([r], f"{r['name']} - loss", out_dir / f"{stem}__loss.png")
        made.append(out_dir / f"{stem}__loss.png")
        if hist and plot_train_vs_val([r], hist, label, f"{r['name']} - over-fitting check", out_dir / f"{stem}__trainval.png"):
            made.append(out_dir / f"{stem}__trainval.png")
        if key and plot_split_gap([r], {key: label}, f"{r['name']} - train / val / test", out_dir / f"{stem}__splits.png") is not None:
            made.append(out_dir / f"{stem}__splits.png")
    return made


def crossdomain(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        if not r["tag"].endswith("crossdomain"):
            continue
        m = r["metrics"]
        for k, v in m.items():
            if k.startswith("test_") and (k.endswith("_f1_micro") or k.endswith("_f1_macro") or k.endswith("_accuracy")) \
                    and isinstance(v, (int, float)):
                dom, met = k[5:].split("_", 1)
                rows.append({"task": r["tag"].replace("_crossdomain", ""), "model": r["run"], "domain": dom,
                             "metric": met, "score": round(v * 100, 1)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot_table(index=["task", "domain", "metric"], columns="model", values="score").reset_index()


def relation_types(runs: list[dict]) -> pd.DataFrame:
    r = next((x for x in runs if x["tag"] == "n2_relation"), None)
    if not r:
        return pd.DataFrame()
    pt = r["metrics"].get("test_per_type", {})
    return pd.DataFrame([{"relation": k, "precision": round(v["precision"] * 100, 1), "recall": round(v["recall"] * 100, 1),
                          "F1": round(v["f1-score"] * 100, 1), "test pairs": int(v["support"])} for k, v in pt.items()])


def evidence_ranking(runs: list[dict]) -> pd.DataFrame:
    r = next((x for x in runs if x["tag"] == "n3_evidence"), None)
    if not r:
        return pd.DataFrame()
    m = r["metrics"]
    rows = []
    for k, name in (("recall_at_1", "Recall@1"), ("recall_at_3", "Recall@3"), ("recall_at_5", "Recall@5"),
                    ("mrr", "MRR"), ("evidence_f1", "Evidence F1")):
        rows.append({"metric": name, "SciBERT evidence finder": round(m.get(f"rank_{k}", np.nan) * 100, 1),
                     "word overlap (BM25-like)": round(m.get(f"overlap_{k}", np.nan) * 100, 1)})
    return pd.DataFrame(rows)


def plot_bars(df: pd.DataFrame, index: str, cols: list[str], title: str, out: Path, ylabel: str = "Score (%)") -> Path:
    import matplotlib.pyplot as plt

    from .style import INK_2, PALETTE, apply_style, save_fig

    apply_style()
    fig, ax = plt.subplots(figsize=(max(5.5, 1.2 * len(df)), 3.8))
    x = np.arange(len(df))
    w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        bars = ax.bar(x + (k - (len(cols) - 1) / 2) * w, df[c], w * 0.92, color=PALETTE[k % len(PALETTE)], label=c)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(x, df[index], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 108)
    ax.set_title(title, pad=24)
    ax.legend(ncol=len(cols), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    save_fig(fig, out)
    return out
