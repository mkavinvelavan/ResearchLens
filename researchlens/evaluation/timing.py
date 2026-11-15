"""Speed measurements for the report: how long the pipeline takes per paper, on a GPU and in website-like CPU mode.

For each PDF:
  parse            PDF -> text blocks, figure/table/equation crops (pdfplumber, rules)
  models           our fine-tuned models: NER, relations, figure types (ViT + caption SciBERT), formulas (TrOCR)
                   (+ the per-step times the pipeline prints)
  index            bge + SigLIP vectors -> Qdrant
  load_saved       loading the saved results of the same paper into a fresh Qdrant (= opening a demo paper)
  retrieval        retrieve() for a few questions (median ms)
  answer           (optional) Gemini answer time for a few questions (median s)
Website-like CPU mode: run with CUDA_VISIBLE_DEVICES="" RL_LOW_MEMORY=1 (2 PyTorch threads, the website's limits on
relation pairs and search candidates). The website itself (Streamlit Cloud) can be somewhat slower.

    python -m researchlens.evaluation.timing a.pdf b.pdf --label gpu --out results/timing [--answers 3]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

QUESTIONS = ["What problem does the paper address?", "Which datasets are used for the experiments?",
             "What are the main results compared with previous methods?", "Explain the architecture figure.",
             "What are the limitations of the approach?"]
STEP_RE = re.compile(r"\[time\] (.+?): (\d+)s")


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--label", default="gpu")
    ap.add_argument("--out", default="results/timing")
    ap.add_argument("--answers", type=int, default=0, help="also time this many Gemini answers")
    a = ap.parse_args()

    import numpy as np
    import torch

    from researchlens.pipeline.answer import ask
    from researchlens.pipeline.enrich import enrich
    from researchlens.pipeline.index import client, index_paper
    from researchlens.pipeline.models import ModelBank
    from researchlens.pipeline.parse_pdf import parse, paper_id_of
    from researchlens.pipeline.retrieve import retrieve
    from researchlens.pipeline.run import load_processed

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / f"work_{a.label}"
    shutil.rmtree(work, ignore_errors=True)
    t = time.time()
    bank = ModelBank()
    qc = client(work / "qdrant")
    device = "GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU, {torch.get_num_threads()} threads"
    rows = []
    for pdf in a.pdfs:
        r = {"pdf": Path(pdf).name, "device": device, "low_memory": bool(getattr(bank, "low_memory", False))}
        dst = work / paper_id_of(Path(pdf))
        t = time.time()
        doc = parse(Path(pdf), dst)
        r["parse_s"] = round(time.time() - t, 1)
        r["blocks"] = len(doc["blocks"])
        r["pages"] = max((b.get("page", 0) for b in doc["blocks"]), default=0) + 1
        buf = io.StringIO()
        t = time.time()
        with contextlib.redirect_stdout(Tee(sys.stdout, buf)):
            enrich(dst, bank)
        r["models_s"] = round(time.time() - t, 1)   # includes loading each model the first time
        r["model_steps_s"] = {k: int(v) for k, v in STEP_RE.findall(buf.getvalue())}
        t = time.time()
        index_paper(dst, bank, qc)
        r["index_s"] = round(time.time() - t, 1)
        r["total_s"] = round(r["parse_s"] + r["models_s"] + r["index_s"], 1)
        # opening the saved results (what a demo paper does), into an empty Qdrant
        qc2 = client(work / f"qdrant_load_{dst.name}")
        t = time.time()
        load_processed(dst, bank, qc2, log=lambda m: None)
        r["load_saved_s"] = round(time.time() - t, 1)
        pid = json.loads((dst / "doc.json").read_text(encoding="utf-8"))["paper_id"]
        ms = []
        for q in QUESTIONS:
            t = time.time()
            retrieve(bank, q, [pid], qc=qc)
            ms.append((time.time() - t) * 1000)
        r["retrieval_ms_median"] = round(float(np.median(ms[1:] or ms)), 0)   # first call warms up
        secs = []
        for q in QUESTIONS[:a.answers]:
            t = time.time()
            try:
                ask(bank, q, [pid], qc=qc)
                secs.append(time.time() - t)
            except Exception as e:
                print("answer failed:", str(e)[:120])
            time.sleep(5)
        if secs:
            r["answer_s_median"] = round(float(np.median(secs)), 1)
        rows.append(r)
        print(json.dumps(r, indent=1), flush=True)
    path = out / f"timing_{a.label}.json"
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("written", path)


if __name__ == "__main__":
    main()
