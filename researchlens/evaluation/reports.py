"""One function per model that writes *every* graph and table for the report.

Output: <results_root>/<model_tag>/report/*.png|*.pdf + summary.csv + summary.tex + extras.
Usage (Results notebook):
    from researchlens.evaluation.reports import ner_report, figcls_report
    ner_report("/content/drive/MyDrive/ResearchLens/results")
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots as P


def _write_tables(table: pd.DataFrame, keys: list[str], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "summary_raw.csv", index=False)
    pretty = P.formatted_table(table, keys)
    pretty.to_csv(out_dir / "summary.csv", index=False)
    try:
        (out_dir / "summary.tex").write_text(pretty.to_latex(index=False), encoding="utf-8")
    except Exception:
        pass
    return pretty


def _common_curves(seeds, metric, metric_label, nice, out):
    P.plot_loss_curves(seeds, f"{nice}: loss", out / "loss_curves")
    P.plot_val_metric(seeds, metric, metric_label, f"{nice}: validation {metric_label}", out / "val_metric")
    P.plot_lr(seeds[0], f"{nice}: learning-rate schedule", out / "lr_schedule")


# ============================================================================ N1  NER
def bio_entities(tags: list[str]) -> set[tuple[int, int, str]]:
    """Strict IOB2 spans (start, end_inclusive, type)."""
    ents, start, typ = set(), None, None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and t[2:] != typ):
            if start is not None:
                ents.add((start, i - 1, typ))
            start, typ = (i, t[2:]) if t.startswith("B-") else (None, None)
    return ents


def ner_report(results_root, model_tag="n1_ner", main_run="scibert", names=None) -> pd.DataFrame:
    root = Path(results_root)
    runs = P.find_runs(root, model_tag)
    if main_run not in runs:
        raise FileNotFoundError(f"No finished '{main_run}' runs under {root / model_tag}. Found: {list(runs)}")
    out = root / model_tag / "report"
    seeds = runs[main_run]
    nice = "SciBERT NER (SciERC)"

    _common_curves(seeds, "f1", "F1", nice, out)

    keys = ["test_f1_micro", "test_f1_macro", "test_precision_micro", "test_recall_micro"]
    table = P.summary_table(runs, keys)
    pretty = _write_tables(table, keys, out)
    P.plot_run_comparison(table, ["test_f1_micro", "test_f1_macro"],
                          {"test_f1_micro": "Micro-F1", "test_f1_macro": "Macro-F1"},
                          "NER: fine-tuned SciBERT vs baselines", out / "baseline_comparison")

    classes = [c for c in seeds[0]["metrics"]["per_class"]]
    P.plot_per_class(seeds, classes, "NER: per-entity-type scores", out / "per_class")

    # token-level entity-type confusion (B-/I- collapsed to the type)
    preds = seeds[0]["predictions"]
    g = [t.split("-", 1)[-1] for p in preds for t in p["gold"]]
    pr = [t.split("-", 1)[-1] for p in preds for t in p["pred"]]
    P.plot_confusion(g, pr, classes + ["O"], "NER: token-level entity-type confusion", out / "confusion_matrix")

    # recall by gold entity length + error examples
    hit, tot, errors = Counter(), Counter(), []
    for p in preds:
        gs, ps = bio_entities(p["gold"]), bio_entities(p["pred"])
        for s, e, t in gs:
            L = min(e - s + 1, 5)
            tot[L] += 1
            hit[L] += (s, e, t) in ps
        if gs != ps:
            toks = p["tokens"]
            errors.append({
                "id": p["id"], "sentence": " ".join(toks),
                "missed": "; ".join(f"{' '.join(toks[s:e+1])} [{t}]" for s, e, t in sorted(gs - ps)),
                "spurious": "; ".join(f"{' '.join(toks[s:e+1])} [{t}]" for s, e, t in sorted(ps - gs)),
            })
    bucket = pd.DataFrame({"length": [str(k) if k < 5 else "5+" for k in sorted(tot)],
                           "recall": [hit[k] / tot[k] for k in sorted(tot)], "n": [tot[k] for k in sorted(tot)]})
    P.plot_score_by_bucket(bucket, "length", "recall", "Entity length (tokens)", "NER: recall vs entity length",
                           out / "recall_by_length")
    pd.DataFrame(errors).to_csv(out / "error_examples.csv", index=False)
    print(f"NER report written to {out}")
    return pretty


# ============================================================================ V2  figure classifier
def figcls_report(results_root, model_tag="v2_figcls", main_run="vit_base") -> pd.DataFrame:
    root = Path(results_root)
    runs = P.find_runs(root, model_tag)
    if main_run not in runs:
        raise FileNotFoundError(f"No finished '{main_run}' runs under {root / model_tag}. Found: {list(runs)}")
    out = root / model_tag / "report"
    seeds = runs[main_run]
    nice = "ViT figure classifier (SPIQA)"
    classes = seeds[0]["metrics"]["classes"]

    _common_curves(seeds, "macro_f1", "macro-F1", nice, out)

    keys = ["test_accuracy", "test_f1_macro", "test_roc_auc_macro_ovr"]
    table = P.summary_table(runs, keys)
    pretty = _write_tables(table, keys, out)
    P.plot_run_comparison(table, ["test_accuracy", "test_f1_macro"],
                          {"test_accuracy": "Accuracy", "test_f1_macro": "Macro-F1"},
                          "Figure classification: ViT vs baselines", out / "baseline_comparison")
    P.plot_per_class(seeds, classes, "Figure classification: per-class scores", out / "per_class")

    preds = seeds[0]["predictions"]
    y = [p["gold"] for p in preds]
    yhat = [p["pred"] for p in preds]
    P.plot_confusion(y, yhat, classes, "Figure classification: confusion matrix", out / "confusion_matrix")
    P.plot_confusion(y, yhat, classes, "Figure classification: confusion matrix (counts)",
                     out / "confusion_matrix_counts", normalize=False)

    probs = np.array([p["probs"] for p in preds])
    y_idx = np.array([classes.index(c) for c in y])
    P.plot_roc_pr(y_idx, probs, classes, nice, out / "roc_pr_curves")

    if "embeddings" in seeds[0]:
        P.plot_embedding_2d(seeds[0]["embeddings"], y, classes, "t-SNE of fine-tuned ViT embeddings (test set)",
                            out / "tsne_embeddings")
    # confidence: correct vs wrong
    conf = probs.max(1)
    correct = np.array(y) == np.array(yhat)
    import matplotlib.pyplot as plt
    from .style import PALETTE, apply_style, save_fig

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf[correct], bins=bins, color=PALETTE[0], alpha=0.75, label="Correct", edgecolor="#fcfcfb")
    ax.hist(conf[~correct], bins=bins, color=PALETTE[1], alpha=0.75, label="Wrong", edgecolor="#fcfcfb")
    ax.set(xlabel="Max softmax confidence", ylabel="Test figures", title="Confidence of correct vs wrong predictions")
    ax.legend()
    save_fig(fig, out / "confidence_hist")
    print(f"Figure-classifier report written to {out}")
    return pretty
