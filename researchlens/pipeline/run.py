"""Process one or two papers end to end: parse -> enrich (our models) -> index (Qdrant).

    python -m researchlens.pipeline.run paperA.pdf [paperB.pdf] --work_dir /content/rl_papers
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from researchlens.pipeline.enrich import enrich
from researchlens.pipeline.index import client, delete_paper, index_paper, paper_points
from researchlens.pipeline.parse_pdf import parse, paper_id_of


def _say(log, msg):
    log(msg)
    if log is not print:
        print(msg, flush=True)


def is_processed(out: Path) -> bool:
    try:
        return bool(json.loads((out / "doc.json").read_text(encoding="utf-8")).get("enriched"))
    except Exception:
        return False


def load_processed(out_dir, bank, qc=None, log=print) -> dict:
    """Load an already processed paper folder (doc.json + crops + saved vectors) and make sure it is in Qdrant.
    Used for a re-uploaded PDF and for the pre-processed demo papers (no model has to run again)."""
    from researchlens.pipeline.analysis import load_doc

    out = Path(out_dir)
    qc = qc or client()
    doc = load_doc(out)
    if paper_points(doc["paper_id"], qc) == 0:
        t = time.time()
        index_paper(out, bank, qc)
        _say(log, f"indexed in {time.time() - t:.0f}s")
    _say(log, "loaded the processed paper (no reprocessing needed)")
    return doc


def process_paper(pdf, work_dir, bank, qc=None, backend: str = "pdfplumber", log=print) -> dict:
    pdf, work_dir = Path(pdf), Path(work_dir)
    out = work_dir / paper_id_of(pdf)
    qc = qc or client()
    if is_processed(out):                       # same PDF again: reuse the earlier results
        return load_processed(out, bank, qc, log)
    t0 = time.time()
    doc = parse(pdf, out, backend=backend)
    _say(log, f"parsed in {time.time() - t0:.0f}s ({doc['backend']}): {len(doc['blocks'])} blocks")
    t1 = time.time()
    doc = enrich(out, bank)
    _say(log, f"our models ran in {time.time() - t1:.0f}s")
    t2 = time.time()
    delete_paper(doc["paper_id"], qc)
    index_paper(out, bank, qc)
    _say(log, f"indexed in {time.time() - t2:.0f}s (total {time.time() - t0:.0f}s)")
    doc["out_dir"] = str(out.resolve())
    return doc


if __name__ == "__main__":
    from researchlens.pipeline.models import ModelBank

    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--work_dir", default="rl_papers")
    a = ap.parse_args()
    bank, qc = ModelBank(), client()
    for p in a.pdfs:
        d = process_paper(p, a.work_dir, bank, qc)
        print(d["paper_id"], d.get("title"))
