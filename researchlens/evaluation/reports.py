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
    P.plot_train_vs_val(seeds, metric, metric_label, f"{nice}: train vs validation {metric_label}",
                        out / f"train_vs_val_{metric}")
    P.plot_train_vs_val(seeds, "accuracy", "accuracy", f"{nice}: train vs validation accuracy",
                        out / "train_vs_val_accuracy")
    P.plot_lr(seeds[0], f"{nice}: learning-rate schedule", out / "lr_schedule")


def _export_all(runs, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    P.all_metrics_table(runs).to_csv(out / "all_metrics.csv", index=False)


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


def ner_cis(seed_run: dict) -> dict:
    """Bootstrap over test sentences: entity micro-F1 and token accuracy."""
    preds = seed_run["predictions"]
    tp, fp, fn, ok, tot = [], [], [], [], []
    for p in preds:
        g, pr = bio_entities(p["gold"]), bio_entities(p["pred"])
        tp.append(len(g & pr)); fp.append(len(pr - g)); fn.append(len(g - pr))
        ok.append(sum(a == b for a, b in zip(p["gold"], p["pred"]))); tot.append(len(p["gold"]))
    tp, fp, fn, ok, tot = map(np.array, (tp, fp, fn, ok, tot))

    def f1(i):
        t, f_p, f_n = tp[i].sum(), fp[i].sum(), fn[i].sum()
        return 2 * t / max(2 * t + f_p + f_n, 1)

    return {"test_f1_micro": P.bootstrap(f1, len(preds)),
            "test_accuracy": P.bootstrap(lambda i: ok[i].sum() / max(tot[i].sum(), 1), len(preds))}


def ner_report(results_root, model_tag="n1_ner", main_run="scibert", names=None) -> pd.DataFrame:
    root = Path(results_root)
    runs = P.find_runs(root, model_tag)
    if main_run not in runs:
        raise FileNotFoundError(f"No finished '{main_run}' runs under {root / model_tag}. Found: {list(runs)}")
    out = root / model_tag / "report"
    # older runs (before accuracy was logged): compute token accuracy from the saved predictions
    for rs in runs.values():
        for sd in rs:
            m, preds = sd["metrics"], sd["predictions"]
            if "test_accuracy" not in m and preds:
                g = np.array([t for p in preds for t in p["gold"]])
                pr = np.array([t for p in preds for t in p["pred"]])
                m["test_accuracy"] = float((g == pr).mean())
                m["test_entity_token_accuracy"] = float((g[g != "O"] == pr[g != "O"]).mean())
    seeds = runs[main_run]
    nice = "SciBERT NER (SciERC)"

    _common_curves(seeds, "f1", "F1", nice, out)
    for name, rs in runs.items():           # same curves for every other run (baseline, no-regularisation ...)
        if name != main_run and not rs[0]["history"].empty:
            _common_curves(rs, "f1", "F1", f"NER, {name}", out / "runs" / name)
    tbl = P.summary_table(runs, ["train_f1_micro", "val_f1_micro", "test_f1_micro"])
    if "train_f1_micro_mean" in tbl:
        P.plot_run_comparison(tbl, ["train_f1_micro", "val_f1_micro", "test_f1_micro"],
                              {"train_f1_micro": "Train F1", "val_f1_micro": "Validation F1", "test_f1_micro": "Test F1"},
                              "NER over-fitting check: train vs validation vs test F1 per run", out / "overfitting_comparison")
    P.plot_split_gap(seeds, {"f1_micro": "Entity F1 (micro)", "f1_macro": "Entity F1 (macro)",
                             "accuracy": "Token accuracy", "entity_token_accuracy": "Entity-token accuracy"},
                     "NER: train vs validation vs test (best checkpoint)", out / "generalisation_gap")
    _export_all(runs, out)

    keys = ["test_f1_micro", "test_f1_macro", "test_f1_weighted", "test_precision_micro", "test_recall_micro",
            "test_accuracy", "test_entity_token_accuracy", "test_sentence_exact_match",
            "train_f1_micro", "val_f1_micro"]
    table = P.add_cis(P.summary_table(runs, keys), runs, ner_cis)
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
RUN_LABELS = {
    "scibert_caption": "Caption only\n(SciBERT)",
    "resnet50": "Image only\n(ResNet-50)",
    "vit_linear_probe": "Image only\n(ViT probe)",
    "vit_base": "Image only\n(ViT)",
    "late_fusion_avg": "Image+caption\n(avg probs)",
    "fusion_vit_scibert": "Image+caption\n(gated fusion)",
}


def build_late_fusion(results_root, model_tag="v2_figcls", image_run="vit_base", text_run="scibert_caption",
                      out_run="late_fusion_avg") -> None:
    """Zero-training baseline: average the probabilities of the image-only and caption-only models
    (same seed, same test figures). Saved as a normal run folder so every plot can use it."""
    import json

    from researchlens.common.utils import write_json

    base = Path(results_root) / model_tag
    for img_dir in sorted((base / image_run).glob("seed_*")):
        txt_dir = base / text_run / img_dir.name
        if not (img_dir / "predictions.jsonl").exists() or not (txt_dir / "predictions.jsonl").exists():
            continue
        img = {r["path"]: r for r in map(json.loads, open(img_dir / "predictions.jsonl", encoding="utf-8"))}
        txt = {r["path"]: r for r in map(json.loads, open(txt_dir / "predictions.jsonl", encoding="utf-8"))}
        common = [k for k in img if k in txt]
        classes = json.loads((img_dir / "metrics.json").read_text())["classes"]
        probs = np.array([(np.array(img[k]["probs"]) + np.array(txt[k]["probs"])) / 2 for k in common])
        probs = probs / probs.sum(1, keepdims=True)  # saved probs are rounded -> renormalise to sum exactly to 1
        y =np.array([classes.index(img[k]["gold"]) for k in common])
        from sklearn.metrics import classification_report, roc_auc_score

        rep = classification_report(y, probs.argmax(1), labels=list(range(len(classes))), target_names=classes,
                                    output_dict=True, zero_division=0)
        out = base / out_run / img_dir.name
        out.mkdir(parents=True, exist_ok=True)
        write_json({
            "run_name": out_run, "seed": int(img_dir.name.split("_")[1]), "modality": "image + caption (prob. average)",
            "test_accuracy": rep["accuracy"], "test_f1_macro": rep["macro avg"]["f1-score"],
            "test_precision_macro": rep["macro avg"]["precision"], "test_recall_macro": rep["macro avg"]["recall"],
            "test_roc_auc_macro_ovr": roc_auc_score(y, probs, multi_class="ovr", average="macro"),
            "per_class": {c: {m: rep[c][m] for m in ("precision", "recall", "f1-score", "support")} for c in classes},
            "classes": classes, "train_seconds": 0, "total_params": None,
        }, out / "metrics.json")
        with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
            for k, pr in zip(common, probs):
                f.write(json.dumps({"path": k, "gold": img[k]["gold"], "pred": classes[int(pr.argmax())],
                                    "probs": [round(float(x), 5) for x in pr]}) + "\n")


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact McNemar test on paired predictions: is model B significantly different from model A?"""
    from scipy.stats import binomtest

    b01 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    b10 = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n = b01 + b10
    p = binomtest(b10, n, 0.5).pvalue if n else 1.0
    return {"only_A_correct": b01, "only_B_correct": b10, "p_value": p}


def _correct_by_path(run: dict) -> dict:
    return {p["path"]: p["gold"] == p["pred"] for p in run["predictions"]}


def figcls_report(results_root, model_tag="v2_figcls", main_run="vit_base") -> pd.DataFrame:
    import matplotlib.pyplot as plt

    from .style import INK_2, PALETTE, apply_style, save_fig

    root = Path(results_root)
    build_late_fusion(root, model_tag)
    runs = P.find_runs(root, model_tag)
    if main_run not in runs:
        raise FileNotFoundError(f"No finished '{main_run}' runs under {root / model_tag}. Found: {list(runs)}")
    order = [r for r in RUN_LABELS if r in runs] + [r for r in runs if r not in RUN_LABELS]
    runs = {r: runs[r] for r in order}
    out = root / model_tag / "report"
    classes = runs[main_run][0]["metrics"]["classes"]

    # ---- training curves for every trained run ----
    for name, seeds in runs.items():
        if seeds[0]["history"].empty:
            continue
        nice = RUN_LABELS.get(name, name).replace("\n", " ")
        sub = out if name == main_run else out / "runs" / name
        _common_curves(seeds, "macro_f1", "macro-F1", f"Figure type, {nice}", sub)

    # ---- summary table + modality comparison ----
    _export_all(runs, out)
    P.plot_split_gap(runs[main_run], {"f1_macro": "Macro-F1", "accuracy": "Accuracy",
                                      "balanced_accuracy": "Balanced accuracy"},
                     "Image-only ViT: train vs validation vs test", out / "generalisation_gap")
    if "fusion_vit_scibert" in runs:
        P.plot_split_gap(runs["fusion_vit_scibert"], {"f1_macro": "Macro-F1", "accuracy": "Accuracy",
                                                      "balanced_accuracy": "Balanced accuracy"},
                         "Gated fusion: train vs validation vs test", out / "runs" / "fusion_vit_scibert" / "generalisation_gap")
    keys = ["test_accuracy", "test_f1_macro", "test_balanced_accuracy", "test_precision_macro", "test_recall_macro",
            "test_top2_accuracy", "test_cohen_kappa", "test_roc_auc_macro_ovr", "train_f1_macro", "val_f1_macro"]
    table = P.add_cis(P.summary_table(runs, keys), runs,
                      lambda sd: P.classification_cis(sd["predictions"], classes))
    table["run"] = [RUN_LABELS.get(r, r).replace("\n", " ") for r in table["run"]]
    pretty = _write_tables(table, keys, out)
    table_plot = table.copy()
    table_plot["run"] = [RUN_LABELS.get(r, r) for r in runs]
    P.plot_run_comparison(table_plot, ["test_accuracy", "test_f1_macro"],
                          {"test_accuracy": "Accuracy", "test_f1_macro": "Macro-F1"},
                          "Figure-type classification: text vs vision vs fusion", out / "modality_comparison")

    # per-class F1 for each modality (key multimodal graph)
    show = [r for r in ("scibert_caption", "vit_base", "late_fusion_avg", "fusion_vit_scibert") if r in runs]
    apply_style()
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(classes)), 3.9))
    w = 0.8 / len(show)
    x = np.arange(len(classes))
    for k, r in enumerate(show):
        vals = np.array([[s["metrics"]["per_class"][c]["f1-score"] for c in classes] for s in runs[r]])
        ax.bar(x + (k - (len(show) - 1) / 2) * w, vals.mean(0) * 100, w * 0.92,
               yerr=vals.std(0, ddof=1) * 100 if len(vals) > 1 else None, color=PALETTE[k],
               label=RUN_LABELS[r].replace("\n", " "), capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK_2})
    ax.set_xticks(x, classes)
    ax.set_ylabel("Test F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-class F1 by modality", pad=30)
    ax.legend(ncol=len(show), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    save_fig(fig, out / "per_class_by_modality")

    # ---- significance: does fusion beat each single modality? (McNemar, seed-matched) ----
    sig_rows = []
    best = "fusion_vit_scibert" if "fusion_vit_scibert" in runs else ("late_fusion_avg" if "late_fusion_avg" in runs else None)
    if best:
        for other in [r for r in runs if r != best]:
            for sb in runs[best]:
                sa = next((s for s in runs[other] if s["metrics"].get("seed") == sb["metrics"].get("seed")), None)
                if not sa:
                    continue
                ca, cb = _correct_by_path(sa), _correct_by_path(sb)
                common = [k for k in cb if k in ca]
                res = mcnemar(np.array([ca[k] for k in common]), np.array([cb[k] for k in common]))
                sig_rows.append({"model_A": other, "model_B": best, "seed": sb["metrics"].get("seed"), "n": len(common), **res})
        pd.DataFrame(sig_rows).to_csv(out / "significance_mcnemar.csv", index=False)

    # ---- detailed plots for the main image model and the fusion model ----
    for name in [main_run] + [r for r in ("fusion_vit_scibert", "scibert_caption") if r in runs]:
        seeds = runs[name]
        nice = RUN_LABELS.get(name, name).replace("\n", " ")
        sub = out if name == main_run else out / "runs" / name
        P.plot_per_class(seeds, classes, f"{nice}: per-class scores", sub / "per_class")
        preds = seeds[0]["predictions"]
        y = [p["gold"] for p in preds]
        yhat = [p["pred"] for p in preds]
        P.plot_confusion(y, yhat, classes, f"{nice}: confusion matrix", sub / "confusion_matrix")
        P.plot_confusion(y, yhat, classes, f"{nice}: confusion matrix (counts)", sub / "confusion_matrix_counts",
                         normalize=False)
        probs = np.array([p["probs"] for p in preds])
        y_idx = np.array([classes.index(c) for c in y])
        P.plot_roc_pr(y_idx, probs, classes, nice, sub / "roc_pr_curves")
        if "embeddings" in seeds[0]:
            P.plot_embedding_2d(seeds[0]["embeddings"], y, classes, f"t-SNE of {nice} embeddings (test set)",
                                sub / "tsne_embeddings")
        conf, correct = probs.max(1), np.array(y) == np.array(yhat)
        apply_style()
        fig, ax = plt.subplots(figsize=(6, 3.4))
        bins = np.linspace(0, 1, 21)
        ax.hist(conf[correct], bins=bins, color=PALETTE[0], alpha=0.75, label="Correct", edgecolor="#fcfcfb")
        ax.hist(conf[~correct], bins=bins, color=PALETTE[1], alpha=0.75, label="Wrong", edgecolor="#fcfcfb")
        ax.set(xlabel="Max softmax confidence", ylabel="Test figures", title=f"{nice}: confidence, correct vs wrong")
        ax.legend()
        save_fig(fig, sub / "confidence_hist")

    # ---- fusion gate: how much each class relies on image vs caption ----
    if "fusion_vit_scibert" in runs:
        preds = [p for s in runs["fusion_vit_scibert"] for p in s["predictions"] if "gate_image" in p]
        g = pd.DataFrame(preds).groupby("gold")["gate_image"].agg(["mean", "std"]).reindex(classes)
        apply_style()
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        yy = np.arange(len(classes))
        ax.barh(yy, g["mean"] * 100, color=PALETTE[0], label="Image (ViT)", height=0.6)
        ax.barh(yy, (1 - g["mean"]) * 100, left=g["mean"] * 100, color=PALETTE[1], label="Caption (SciBERT)",
                height=0.6)
        for i, v in enumerate(g["mean"]):
            ax.text(v * 100 - 2, i, f"{v*100:.0f}%", va="center", ha="right", fontsize=8, color="white")
        ax.axvline(50, color=INK_2, lw=0.8, ls="--")
        ax.set_yticks(yy, classes)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Average fusion gate weight (%)")
        ax.grid(False)
        ax.set_title("Which modality the fusion model trusts, per true class", pad=30)
        ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
        save_fig(fig, out / "fusion_gate_by_class")

    print(f"Figure-classifier report written to {out}")
    return pretty
