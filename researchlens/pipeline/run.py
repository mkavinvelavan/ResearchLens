"""Process one or two papers end to end: parse -> enrich (our models) -> index (Qdrant).

    python -m researchlens.pipeline.run paperA.pdf [paperB.pdf] --work_dir /content/rl_papers
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from researchlens.pipeline.enrich import enrich
from researchlens.pipeline.index import client, delete_paper, index_paper
from researchlens.pipeline.parse_pdf import parse, paper_id_of


def process_paper(pdf, work_dir, bank, qc=None, backend: str = "auto", log=print) -> dict:
    pdf, work_dir = Path(pdf), Path(work_dir)
    out = work_dir / paper_id_of(pdf)
    qc = qc or client()
    t0 = time.time()
    doc = parse(pdf, out, backend=backend)
    log(f"parsed in {time.time() - t0:.0f}s ({doc['backend']}): {len(doc['blocks'])} blocks")
    t1 = time.time()
    doc = enrich(out, bank)
    log(f"our models ran in {time.time() - t1:.0f}s")
    t2 = time.time()
    delete_paper(doc["paper_id"], qc)
    index_paper(out, bank, qc)
    log(f"indexed in {time.time() - t2:.0f}s")
    doc["out_dir"] = str(out.resolve())
    return doc


if __name__ == "__main__":
    from researchlens.pipeline.models import ModelBank

    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--work_dir", default="rl_papers")
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    bank, qc = ModelBank(), client()
    for p in a.pdfs:
        d = process_paper(p, a.work_dir, bank, qc, a.backend)
        print(d["paper_id"], d.get("title"))
