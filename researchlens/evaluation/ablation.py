"""Ablation of the retrieval pipeline: switch off one step at a time and measure what the answer model would receive.

Same questions as the system evaluation (same seed): QASPER test (text evidence) and SPIQA test-A (figure evidence).
Every setting calls the real `retrieve()` with the app's settings; only the switched-off step differs.

  full                    the app (bge search -> evidence finder re-ranks paragraphs -> + named labels
                          -> + SigLIP image search -> + figures cited by the top paragraphs)
  no_evidence_finder      no re-ranking (bge order only)
  rerank_all_kinds        the earlier design: the evidence finder also re-ranks figure/table captions
  no_siglip               no SigLIP image search
  no_named_labels         "Figure 3" / "Table 2" named in the question are not looked up
  no_linked_figures       figures/tables cited by the top paragraphs are not added
  text_search_only        bge search only (every extra step off)
  cpu_12_candidates       the website's CPU setting (12 search candidates instead of 24)

Metrics on the evidence list (in order): hit@1 (first item is gold), recall@5 (a gold item in the first 5),
gold_in_evidence (a gold item anywhere in the list), MRR, mean number of items, and retrieval time per question.
No Gemini calls.

    python -m researchlens.evaluation.ablation --out_dir results/ablation --n_qasper 100 --n_spiqa 50
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from researchlens.evaluation.system_eval import (append_jsonl, bootstrap_ci, load_qasper_test, load_spiqa_test,
                                                 read_jsonl, write_doc)

SETTINGS = {
    "full": {},
    "no_evidence_finder": {"rerank": False},
    "rerank_all_kinds": {"rerank_all": True},
    "no_siglip": {"k_image": 0},
    "no_named_labels": {"use_named": False},
    "no_linked_figures": {"use_linked": False},
    "text_search_only": {"rerank": False, "k_image": 0, "use_named": False, "use_linked": False},
    "cpu_12_candidates": {"k_candidates": 12},
}
METRICS = ("hit@1", "recall@5", "gold_in_evidence", "mrr", "n_items", "ms")


def score(ranked: list, gold: set, ms: float) -> dict:
    first = next((i for i, x in enumerate(ranked) if x in gold), None)
    return {"hit@1": float(first == 0), "recall@5": float(first is not None and first < 5),
            "gold_in_evidence": float(first is not None), "mrr": 1.0 / (first + 1) if first is not None else 0.0,
            "n_items": len(ranked), "ms": ms}


def run_settings(bank, qc, question, pid, base_kw, to_key, gold) -> dict:
    from researchlens.pipeline.retrieve import retrieve

    res = {}
    for name, kw in SETTINGS.items():
        args = {"k_candidates": 24, **base_kw, **kw}
        t = time.time()
        ev = retrieve(bank, question, [pid], qc=qc, **args)
        ms = round((time.time() - t) * 1000, 1)
        res[name] = score([to_key(x) for x in ev], gold, ms)
    return res


def run_qasper(bank, qc, out: Path, args) -> None:
    from researchlens.pipeline.index import index_paper, paper_points
    from researchlens.pipeline.parse_pdf import MENTION_RE, norm_label

    rec_path = out / "ablation_qasper.jsonl"
    done = {r["qid"] for r in read_jsonl(rec_path)}
    for p in load_qasper_test(Path(args.data_dir) / "qasper", args.n_qasper, args.seed):
        pid = f"qasper_{p['id']}"
        blocks = []
        for c in p["passages"]:
            if c["kind"] == "paragraph":   # paragraphs remember the figures/tables they cite, as the PDF parser does
                blocks.append({"kind": "paragraph", "text": c["text"], "section": c["section"],
                               "mentions": sorted({norm_label(k, n) for k, n in MENTION_RE.findall(c["text"])})})
            else:
                m = re.match(r"\s*(fig(?:ure)?|table)\s*(\d+)", c["text"], re.I)
                lab = (("Figure" if m.group(1).lower().startswith("fig") else "Table") + " " + m.group(2)) if m else None
                blocks.append({"kind": c["kind"], "text": "", "caption": c["text"], "label": lab})
        folder = write_doc(out / "papers" / pid, pid, p["title"], blocks)
        if paper_points(pid, qc) == 0:
            index_paper(folder, bank, qc)
        bid2idx = {f"b{i:04d}": i for i in range(len(blocks))}
        for q in p["questions"]:
            if q["qid"] in done:
                continue
            res = run_settings(bank, qc, q["question"], pid, {"k_text": 5, "k_image": 2},
                               lambda x: bid2idx.get(x["block_id"], -1), set(q["gold"]))
            append_jsonl(rec_path, {"qid": q["qid"], "paper": p["id"], "settings": res})
        print(f"[ablation QASPER] {len(read_jsonl(rec_path))} questions", flush=True)


def run_spiqa(bank, qc, out: Path, args) -> None:
    from researchlens.pipeline.index import index_paper, paper_points

    rec_path = out / "ablation_spiqa.jsonl"
    done = {r["paper"] for r in read_jsonl(rec_path)}
    for it in load_spiqa_test(Path(args.data_dir) / "spiqa_raw", out, args.n_spiqa, args.seed):
        if it["id"] in done:
            continue
        pid = f"spiqa_{it['id']}"
        write_doc(it["folder"], pid, it["title"], it["blocks"])
        if paper_points(pid, qc) == 0:
            index_paper(it["folder"], bank, qc)
        bid2file = {f"b{i:04d}": b["file"] for i, b in enumerate(it["blocks"])}
        # same call as the system evaluation's "full system" (figures and tables only, 2 from search + 2 from SigLIP)
        res = run_settings(bank, qc, it["question"], pid, {"k_text": 2, "k_image": 2, "kinds": ["figure", "table"]},
                           lambda x: bid2file.get(x["block_id"]), {it["reference"]})
        append_jsonl(rec_path, {"paper": it["id"], "settings": res})
    print(f"[ablation SPIQA] {len(read_jsonl(rec_path))} questions", flush=True)


def summarize(path: Path) -> dict:
    recs = read_jsonl(path)
    out = {"n_questions": len(recs)}
    for name in SETTINGS:
        rows = [r["settings"][name] for r in recs if name in r["settings"]]
        out[name] = {m: bootstrap_ci([x[m] for x in rows]) for m in METRICS if m not in ("n_items", "ms")}
        out[name]["n_items"] = float(np.mean([x["n_items"] for x in rows])) if rows else None
        out[name]["ms"] = float(np.median([x["ms"] for x in rows])) if rows else None   # median: skips warm-up
    return out


def table(name: str, res: dict) -> str:
    def f(t):
        return "–" if not t or t[0] is None else f"{100 * t[0]:.1f} [{100 * t[1]:.0f}–{100 * t[2]:.0f}]"

    full = res["full"]
    lines = [f"### {name} (n = {res['n_questions']})", "",
             "| Setting | Hit@1 | Recall@5 | Gold in evidence | MRR | Items | ms / question | Δ gold in evidence |",
             "|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        r = res[s]
        d = (r["gold_in_evidence"][0] - full["gold_in_evidence"][0]) * 100 if r["gold_in_evidence"][0] is not None else 0
        lines.append(f"| {s} | {f(r['hit@1'])} | {f(r['recall@5'])} | {f(r['gold_in_evidence'])} | {f(r['mrr'])} | "
                     f"{r['n_items']:.1f} | {r['ms']:.0f} | {d:+.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/ablation")
    ap.add_argument("--data_dir", default="/kaggle/working/eval_data")
    ap.add_argument("--n_qasper", type=int, default=100)
    ap.add_argument("--n_spiqa", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from researchlens.pipeline.index import client
    from researchlens.pipeline.models import ModelBank

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank, qc = ModelBank(), client(out / "qdrant")
    run_qasper(bank, qc, out, args)
    run_spiqa(bank, qc, out, args)
    res = {"qasper": summarize(out / "ablation_qasper.jsonl"), "spiqa": summarize(out / "ablation_spiqa.jsonl"),
           "settings": SETTINGS, "seed": args.seed}
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    report = ("# Retrieval ablation\n\nOne step switched off at a time; 95% bootstrap CIs. Retrieval only (no Gemini).\n\n"
              + table("QASPER test: expert evidence passages (text)", res["qasper"]) + "\n\n"
              + table("SPIQA test-A: the reference figure / table (vision)", res["spiqa"]) + "\n")
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
