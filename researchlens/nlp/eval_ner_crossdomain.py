"""Cross-domain test of a trained NER model: SciERC test (AI, in-domain for N1) vs ScienceIE test (CS / Materials / Physics).

Labels are mapped to the shared set (Method, Material, Task, Metric); each domain is scored only on the types it
annotates (ScienceIE has no Metric). Shows how much an AI-only model drops on other fields.

    python -m researchlens.nlp.eval_ner_crossdomain --model Kavinvelavan/researchlens-scibert-ner --run_name scibert_scierc_only
Writes results/n1_ner_crossdomain/<run_name>/seed_42/metrics.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from researchlens.common.results_io import save_metrics
from researchlens.common.utils import LOCAL_DATA, run_dirs
from researchlens.nlp.data_multidomain import fix_bio, load_scienceie, restrict, to_unified
from researchlens.nlp.data_scierc import load_ner_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hub id or local folder of a token-classification model")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--data_dir", default=str(LOCAL_DATA / "scierc"))
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    import torch
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from researchlens.common.hub import hf_login_from_colab
    hf_login_from_colab()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    tests = {"scierc": [{**e, "ner_tags": fix_bio([to_unified(t) for t in e["ner_tags"]])}
                        for e in load_ner_splits(Path(args.data_dir))["test"]],
             "scienceie": load_scienceie()["test"]}

    def predict(exs, bs=32):
        preds = []
        for i in range(0, len(exs), bs):
            batch = exs[i:i + bs]
            enc = tok([e["tokens"] for e in batch], is_split_into_words=True, truncation=True,
                      max_length=args.max_length, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.argmax(-1).cpu().numpy()
            for j, e in enumerate(batch):
                wid, prev, row = enc.word_ids(batch_index=j), None, {}
                for k, w in enumerate(wid):
                    if w is not None and w != prev:
                        row[w] = id2label[int(logits[j, k])]
                    prev = w
                preds.append(fix_bio([to_unified(row.get(w, "O")) for w in range(len(e["tokens"]))]))
        return preds

    metrics = {"model": args.model, "run_name": args.run_name}
    for d, exs in tests.items():
        gold = [restrict(e["ner_tags"], d) for e in exs]
        pred = [restrict(p, d) for p in predict(exs)]
        rep = classification_report(gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
        for avg in ("micro", "macro"):
            for m, n in (("precision", "precision"), ("recall", "recall"), ("f1-score", "f1")):
                metrics[f"test_{d}_{n}_{avg}"] = rep[f"{avg} avg"][m]
        g, p = np.array([t for s in gold for t in s]), np.array([t for s in pred for t in s])
        metrics[f"test_{d}_accuracy"] = float((g == p).mean())
        metrics[f"test_{d}_per_type"] = {k: {m: v[m] for m in ("precision", "recall", "f1-score", "support")}
                                         for k, v in rep.items() if "avg" not in k}
        print(f"[{d}] n={len(exs)}  F1 {metrics[f'test_{d}_f1_micro']:.4f}  P {metrics[f'test_{d}_precision_micro']:.4f}  "
              f"R {metrics[f'test_{d}_recall_micro']:.4f}  per type: "
              + ", ".join(f"{k} {v['f1-score']:.2f}" for k, v in metrics[f'test_{d}_per_type'].items()))
    metrics["domain_drop_f1"] = metrics["test_scierc_f1_micro"] - metrics["test_scienceie_f1_micro"]
    print(f"Drop from AI papers to other fields: {metrics['domain_drop_f1'] * 100:.1f} F1 points")
    out = run_dirs("n1_ner_crossdomain", args.run_name, 42)["results"] / "metrics.json"
    save_metrics(metrics, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
