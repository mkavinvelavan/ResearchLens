"""N3 - Evidence finder with SciBERT on QASPER: does this paragraph (or figure/table caption) help answer the question?

ResearchLens uses it to RE-RANK what Qdrant retrieves, so the Evidence Trace points at the right paragraph.

Input  : [CLS] question [SEP] paragraph-or-caption [SEP]
Output : P(evidence)
Data   : QASPER (Dasigi et al., 2021) - NLP papers, questions written by NLP practitioners from title + abstract,
         answers + evidence marked by other experts in the full paper. Evidence can be paragraphs OR figure/table
         captions ("FLOAT SELECTED: Table 2: ..."), so the candidates include captions too.
Train  : for every question with evidence: all evidence passages (positives) + negatives from the same paper
         (n_hard most word-overlapping non-evidence passages + n_random random ones).
Test   : RANK EVERY passage of the paper for every test question (the real use), report
         Recall@1/3/5, MRR, QASPER-style evidence F1, and the same numbers for a word-overlap (BM25-like)
         baseline -> shows what fine-tuning adds. Pair-level P/R/F1/accuracy/ROC-AUC on the sampled pairs too.

    python -m researchlens.nlp.train_evidence --config configs/n3_evidence.yaml --push_to_hub
Writes results/n3_evidence/<run>/seed_42/{metrics.json, history.csv, predictions.jsonl, config.json}
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from researchlens.common.results_io import HistoryCallback, print_split_table, save_config, save_metrics, save_predictions
from researchlens.common.utils import (
    LOCAL_DATA, Timer, add_regularisation_args, add_storage_args, apply_freezing, backup_to_drive, cleanup_checkpoints,
    count_parameters, dropout_overrides, hardware_info, latest_checkpoint, make_training_args, parse_args_with_config,
    run_dirs, set_seed, should_keep_model, verify_model_saved,
)

QASPER_URLS = {
    "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}
FILES = {"train": "qasper-train-v0.3.json", "validation": "qasper-dev-v0.3.json", "test": "qasper-test-v0.3.json"}
FLOAT = "FLOAT SELECTED:"


# ------------------------------------------------------------------ data
def download_qasper(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f).exists() for f in FILES.values()):
        return data_dir
    for name, url in QASPER_URLS.items():
        print(f"Downloading QASPER {name} from {url} ...", flush=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            buf = io.BytesIO(r.read())
        with tarfile.open(fileobj=buf) as tf:
            for m in tf.getmembers():
                base = Path(m.name).name
                if base in FILES.values():
                    (data_dir / base).write_bytes(tf.extractfile(m).read())
    missing = [f for f in FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"QASPER files missing after download: {missing}")
    return data_dir


def norm(s: str) -> str:
    s = s.strip()
    if s.startswith(FLOAT):
        s = s[len(FLOAT):].strip()
    return re.sub(r"\s+", " ", s).lower()


def passages(paper: dict) -> list[dict]:
    """Every paragraph (with its section) and every figure/table caption of a paper."""
    out = []
    for sec in paper.get("full_text") or []:
        for para in sec.get("paragraphs") or []:
            if para and para.strip():
                out.append({"text": para.strip(), "section": sec.get("section_name") or "", "kind": "paragraph"})
    for ft in paper.get("figures_and_tables") or []:
        cap = (ft.get("caption") or "").strip()
        if cap:
            out.append({"text": cap, "section": "", "kind": "table" if "table" in (ft.get("file") or "").lower() or
                        cap.lower().startswith("table") else "figure"})
    return out


def questions(paper: dict, cands: list[dict]) -> list[dict]:
    """Questions that have evidence among the paper's passages; gold = union over annotators,
    plus each annotator's own set (QASPER's evidence F1 takes the best-matching annotator)."""
    key = {norm(c["text"]): i for i, c in enumerate(cands)}
    out = []
    for qa in paper.get("qas") or []:
        per_annot = []
        for a in qa.get("answers") or []:
            ans = a.get("answer", a)
            idx = sorted({key[norm(e)] for e in (ans.get("evidence") or []) if norm(e) in key})
            if idx:
                per_annot.append(idx)
        if per_annot:
            out.append({"question": qa["question"].strip(), "qid": qa.get("question_id", ""),
                        "gold": sorted({i for s in per_annot for i in s}), "annotators": per_annot})
    return out


TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on for to and or is are was were be by with what which how do does did their they "
           "this that these those it its from as at than then has have use used using".split())


def words(s: str) -> list[str]:
    return [w for w in TOK.findall(s.lower()) if w not in STOP]


def overlap_scores(q: str, cands: list[dict]) -> np.ndarray:
    """BM25-style word overlap (the classic retrieval baseline), IDF computed within the paper."""
    docs = [words(c["text"]) for c in cands]
    N = len(docs)
    df = Counter(w for d in docs for w in set(d))
    avg = sum(map(len, docs)) / max(N, 1)
    qs = set(words(q))
    sc = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in qs:
            if tf[w]:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / max(avg, 1)))
        sc.append(s)
    return np.asarray(sc)


def load_qasper(data_dir: Path) -> dict[str, list[dict]]:
    """{split: [paper {id, title, passages, questions}]}"""
    folder = download_qasper(data_dir)
    out = {}
    for split, fname in FILES.items():
        raw = json.loads((folder / fname).read_text(encoding="utf-8"))
        papers = []
        for pid, paper in raw.items():
            cands = passages(paper)
            qs = questions(paper, cands)
            if cands and qs:
                papers.append({"id": pid, "title": paper.get("title", ""), "passages": cands, "questions": qs})
        nq = sum(len(p["questions"]) for p in papers)
        npass = np.mean([len(p["passages"]) for p in papers])
        nev = np.mean([len(q["gold"]) for p in papers for q in p["questions"]])
        print(f"{split:<10} {len(papers):>4} papers  {nq:>5} questions with evidence  "
              f"{npass:.0f} passages/paper  {nev:.1f} evidence passages/question", flush=True)
        out[split] = papers
    return out


def sample_pairs(papers: list[dict], n_hard: int, n_random: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    for p in papers:
        for q in p["questions"]:
            gold = set(q["gold"])
            for i in gold:
                pairs.append({"q": q["question"], "text": p["passages"][i]["text"], "label": 1})
            neg = [i for i in range(len(p["passages"])) if i not in gold]
            sc = overlap_scores(q["question"], p["passages"])
            hard = sorted(neg, key=lambda i: -sc[i])[:n_hard]
            rest = [i for i in neg if i not in hard]
            for i in hard + rng.sample(rest, min(n_random, len(rest))):
                pairs.append({"q": q["question"], "text": p["passages"][i]["text"], "label": 0})
    rng.shuffle(pairs)
    return pairs


# ------------------------------------------------------------------ ranking metrics
def ranking_metrics(scored: list[tuple[np.ndarray, dict]], prefix: str) -> dict:
    """scored: [(scores over the paper's passages, question)]"""
    r1 = r3 = r5 = mrr = ef1 = 0.0
    for sc, q in scored:
        order = list(np.argsort(-sc))
        gold = set(q["gold"])
        r1 += any(i in gold for i in order[:1])
        r3 += any(i in gold for i in order[:3])
        r5 += any(i in gold for i in order[:5])
        rank = min(order.index(i) for i in gold) + 1
        mrr += 1 / rank
        # predicted evidence set: passages above 0.5 probability (or the top-1 if none); QASPER evidence F1
        pred = {i for i in order if sc[i] >= 0.5} if prefix != "overlap" else set(order[:2])
        pred = pred or {order[0]}
        best = 0.0
        for ann in q["annotators"]:
            ann = set(ann)
            tp = len(pred & ann)
            if tp:
                pr, rc = tp / len(pred), tp / len(ann)
                best = max(best, 2 * pr * rc / (pr + rc))
        ef1 += best
    n = max(len(scored), 1)
    return {f"{prefix}_recall_at_1": r1 / n, f"{prefix}_recall_at_3": r3 / n, f"{prefix}_recall_at_5": r5 / n,
            f"{prefix}_mrr": mrr / n, f"{prefix}_evidence_f1": ef1 / n}


def pair_metrics(y, prob, prefix=None) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    y = np.asarray(y)
    pred = (np.asarray(prob) >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y, prob)
    except ValueError:
        auc = None
    out = {"f1": f, "precision": p, "recall": r, "accuracy": accuracy_score(y, pred), "roc_auc": auc}
    return {f"{prefix}_{k}": v for k, v in out.items()} if prefix else out


def get_args():
    p = argparse.ArgumentParser(description="SciBERT evidence finder on QASPER")
    p.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--model_tag", default="n3_evidence")
    p.add_argument("--run_name", default="scibert_evidence")
    p.add_argument("--data_dir", default=str(LOCAL_DATA / "qasper"))
    p.add_argument("--drive_root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--n_hard", type=int, default=3)
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    add_regularisation_args(p)
    p.add_argument("--early_stopping_patience", type=int, default=1)
    p.add_argument("--max_test_questions", type=int, default=None, help="limit full ranking (speed)")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--push_to_hub", action="store_true")
    add_storage_args(p)
    p.add_argument("--hub_repo", default="researchlens-scibert-evidence")
    return parse_args_with_config(p)


def main():
    args = get_args()
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer)

    set_seed(args.seed)
    dirs = run_dirs(args.model_tag, args.run_name, args.seed, **({"drive_root": args.drive_root} if args.drive_root else {}))
    save_config(args, dirs["results"] / "config.json")
    data = load_qasper(Path(args.data_dir))
    pairs = {s: sample_pairs(v, args.n_hard, args.n_random, args.seed + i) for i, (s, v) in enumerate(data.items())}
    for s, v in pairs.items():
        print(f"{s:<10} {len(v):>6} (question, passage) pairs, {sum(p['label'] for p in v)} positive")

    tok = AutoTokenizer.from_pretrained(args.model_name)

    def to_ds(items):
        ds = Dataset.from_list([{"q": p["q"], "text": p["text"], "labels": p["label"]} for p in items])
        return ds.map(lambda b: tok(b["q"], b["text"], truncation="only_second", max_length=args.max_length),
                      batched=True, remove_columns=["q", "text"])

    ds = {s: to_ds(v) for s, v in pairs.items()}
    train_eval = ds["train"].shuffle(seed=args.seed).select(range(min(3000, len(ds["train"]))))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label={0: "not_evidence", 1: "evidence"},
        label2id={"not_evidence": 0, "evidence": 1},
        **dropout_overrides(args.model_name, args.hidden_dropout, args.attention_dropout))
    apply_freezing(model, args.freeze_embeddings, args.freeze_layers)

    def compute_metrics(ep):
        lg = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        prob = torch.softmax(torch.tensor(lg, dtype=torch.float), -1)[:, 1].numpy()
        return pair_metrics(ep.label_ids, prob)

    targs = make_training_args(
        len(ds["train"]), args.warmup_ratio,
        output_dir=str(dirs["checkpoints"]), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=args.weight_decay, lr_scheduler_type="linear", label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="val_f1", greater_is_better=True, logging_steps=50,
        fp16=args.fp16 and torch.cuda.is_available(), report_to="none", seed=args.seed,
    )
    hist_cb = HistoryCallback(dirs["results"] / "history.csv", resume=latest_checkpoint(dirs["checkpoints"]) is not None)
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset={"train": train_eval, "val": ds["validation"]},
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                      callbacks=[hist_cb, EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])
    with Timer() as t:
        trainer.train(resume_from_checkpoint=latest_checkpoint(dirs["checkpoints"]))

    metrics = {"model": args.model_name, "run_name": args.run_name, "seed": args.seed,
               "n_pairs": {s: len(v) for s, v in pairs.items()},
               "n_questions": {s: sum(len(p["questions"]) for p in v) for s, v in data.items()}}
    for split, d in (("train", train_eval), ("val", ds["validation"]), ("test", ds["test"])):
        o = trainer.predict(d, metric_key_prefix=split)
        lg = o.predictions[0] if isinstance(o.predictions, tuple) else o.predictions
        metrics.update(pair_metrics(o.label_ids, torch.softmax(torch.tensor(lg, dtype=torch.float), -1)[:, 1].numpy(),
                                    prefix=split))
    metrics["gap_train_test_f1"] = metrics["train_f1"] - metrics["test_f1"]
    metrics.update(hist_cb.summary(trainer.state))
    print_split_table({s: {k: metrics[f"{s}_{k}"] or 0.0 for k in ("f1", "precision", "recall", "accuracy", "roc_auc")}
                       for s in ("train", "val", "test")}, ["f1", "precision", "recall", "accuracy", "roc_auc"])

    # ---- the real use: rank ALL passages of the paper for each test question ----
    print("\nRanking every passage of each test paper ...", flush=True)
    model.eval()
    dev = next(model.parameters()).device
    test_qs = [(p, q) for p in data["test"] for q in p["questions"]]
    if args.max_test_questions:
        test_qs = random.Random(args.seed).sample(test_qs, min(args.max_test_questions, len(test_qs)))
    scored_model, scored_overlap, records = [], [], []
    with Timer() as tr:
        for k, (p, q) in enumerate(test_qs):
            texts = [c["text"] for c in p["passages"]]
            probs = []
            for i in range(0, len(texts), 128):
                b = tok([q["question"]] * len(texts[i:i + 128]), texts[i:i + 128], truncation="only_second",
                        max_length=args.max_length, padding=True, return_tensors="pt").to(dev)
                with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
                    probs.append(torch.softmax(model(**b).logits.float(), -1)[:, 1].cpu().numpy())
            sc = np.concatenate(probs)
            scored_model.append((sc, q))
            scored_overlap.append((overlap_scores(q["question"], p["passages"]), q))
            top = list(np.argsort(-sc)[:5])
            records.append({"paper": p["id"], "question": q["question"], "gold": q["gold"], "top5": [int(i) for i in top],
                            "top5_prob": [round(float(sc[i]), 4) for i in top],
                            "top1_kind": p["passages"][top[0]]["kind"], "top1_text": p["passages"][top[0]]["text"][:300]})
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(test_qs)} questions", flush=True)
    metrics.update(ranking_metrics(scored_model, "rank"))
    metrics.update(ranking_metrics(scored_overlap, "overlap"))
    metrics.update({"ranking_questions": len(test_qs),
                    "ranking_passages_per_question": float(np.mean([len(p["passages"]) for p, _ in test_qs])),
                    "ranking_seconds": round(tr.seconds, 1), "best_val_f1": trainer.state.best_metric,
                    "train_seconds": round(t.seconds, 1), **count_parameters(model), **hardware_info()})
    print(f"\n{'Full-paper ranking (test)':<28}{'SciBERT':>10}{'word overlap':>14}")
    for m in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "evidence_f1"):
        print(f"{m:<28}{metrics['rank_' + m] * 100:>9.1f}%{metrics['overlap_' + m] * 100:>13.1f}%")
    save_metrics(metrics, dirs["results"] / "metrics.json")
    save_predictions(records, dirs["results"] / "predictions.jsonl")

    if should_keep_model(args):
        dirs["final_model"].mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(dirs["final_model"]))
        tok.save_pretrained(str(dirs["final_model"]))
        verify_model_saved(dirs["final_model"])
    cleanup_checkpoints(dirs["checkpoints"])
    if args.push_to_hub:
        from researchlens.common.hub import hf_login_from_colab, push_model

        hf_login_from_colab()
        push_model(dirs["final_model"], dirs["results"], args.hub_repo,
                   title="SciBERT evidence finder (QASPER question-passage relevance)", base_model=args.model_name,
                   dataset="QASPER (Dasigi et al., 2021)", task="text-classification")
    if should_keep_model(args):
        backup_to_drive(dirs["final_model"], dirs["drive_model"])


if __name__ == "__main__":
    main()
