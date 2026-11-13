"""Prepare demo papers for the website: process PDFs with the full pipeline (best on a GPU), write the Gemini overview,
and upload each processed folder to the private results dataset (demo_papers/<paper_id>/ + demo_papers/index.json).
The website then loads a demo paper in seconds instead of running every model on its free CPU.

    python -m researchlens.pipeline.make_demo paper1.pdf paper2.pdf --work_dir /kaggle/working/demo
Needs HF_TOKEN (write) and GEMINI_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = f"{os.environ.get('RL_HF_USER', 'Kavinvelavan')}/researchlens-results"


def make_demo(pdfs: list[str], work_dir: str, with_overview: bool = True) -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    from researchlens.pipeline.analysis import overview, save_overview
    from researchlens.pipeline.index import client
    from researchlens.pipeline.models import ModelBank
    from researchlens.pipeline.run import process_paper

    bank, qc = ModelBank(), client(Path(work_dir) / "qdrant")
    api, token = HfApi(), os.environ.get("HF_TOKEN")
    try:
        index = json.loads(Path(hf_hub_download(REPO, "demo_papers/index.json", repo_type="dataset",
                                                 token=token)).read_text(encoding="utf-8"))
    except Exception:
        index = []
    for pdf in pdfs:
        doc = process_paper(pdf, work_dir, bank, qc)
        out = Path(doc["out_dir"])
        if with_overview:
            res = overview(bank, doc, qc=qc)
            save_overview(doc, res)
            print(res["overview"][:600], "...\n")
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(out),
                          path_in_repo=f"demo_papers/{doc['paper_id']}", token=token,
                          ignore_patterns=["*.pdf"], commit_message=f"demo paper: {doc.get('title', '')[:60]}")
        index = [d for d in index if d["paper_id"] != doc["paper_id"]]
        index.append({"paper_id": doc["paper_id"], "title": doc.get("title") or Path(pdf).stem})
        print("uploaded demo paper:", doc["paper_id"], doc.get("title"))
    tmp = Path(work_dir) / "index.json"
    tmp.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(tmp), path_in_repo="demo_papers/index.json", repo_id=REPO,
                    repo_type="dataset", token=token, commit_message="demo paper index")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--work_dir", default="demo_work")
    ap.add_argument("--no_overview", action="store_true")
    a = ap.parse_args()
    for d in make_demo(a.pdfs, a.work_dir, not a.no_overview):
        print(d)


def retitle(paper_id: str, title: str) -> None:
    """Correct a demo paper's title on the Hub (doc.json + index.json), e.g. when a page header was taken as the title."""
    import tempfile

    from huggingface_hub import HfApi, hf_hub_download

    api, token = HfApi(), os.environ.get("HF_TOKEN")
    tmp = Path(tempfile.mkdtemp())
    for name in (f"demo_papers/{paper_id}/doc.json", "demo_papers/index.json"):
        data = json.loads(Path(hf_hub_download(REPO, name, repo_type="dataset", token=token)).read_text(encoding="utf-8"))
        if isinstance(data, list):
            for d in data:
                if d["paper_id"] == paper_id:
                    d["title"] = title
        else:
            data["title"] = title
        f = tmp / Path(name).name
        f.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=name, repo_id=REPO, repo_type="dataset", token=token,
                        commit_message=f"retitle {paper_id}")
    print("retitled", paper_id, "->", title)
