"""Build a figure-type classification dataset from SPIQA (Pramanick et al., 2024).

SPIQA labels every figure/table of every paper with:
    content type: figure | table
    figure type : schematic | plot | photo (visualization) | others
We map these to 5 classes: plot, schematic, photo, table, other.

Splits: SPIQA train -> train, SPIQA val -> validation, SPIQA test-A -> test.
Output (ImageFolder-style + manifest):
    <out_dir>/<split>/<class>/<image>.png
    <out_dir>/manifest.csv   columns: split,label,path,paper_id,caption

Example:
    python -m researchlens.vision.prepare_spiqa_figures --out_dir /content/data/spiqa_figs --max_per_class 3000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["plot", "schematic", "photo", "table", "other"]
REPO = "google/spiqa"
FILES = {
    "train": ("train_val/SPIQA_train.json", "train_val/SPIQA_train_val_Images.zip"),
    "validation": ("train_val/SPIQA_val.json", "train_val/SPIQA_train_val_Images.zip"),
    "test": ("test-A/SPIQA_testA.json", "test-A/SPIQA_testA_Images.zip"),
}


def map_label(meta: dict) -> str | None:
    content = str(meta.get("content_type", meta.get("type", ""))).lower()
    ftype = str(meta.get("figure_type", meta.get("figure type", ""))).lower()
    if "table" in content or "table" in ftype:
        return "table"
    if "plot" in ftype:
        return "plot"
    if "schematic" in ftype:
        return "schematic"
    if "photo" in ftype or "visual" in ftype:
        return "photo"
    if ftype or content:
        return "other"
    return None


def iter_figures(data: dict):
    """Yield (paper_id, image_name, meta). Tolerates small schema differences between splits."""
    for pid, paper in data.items():
        if not isinstance(paper, dict):
            continue
        figs = paper.get("all_figures") or paper.get("figures") or paper.get("all_figures_tables") or {}
        if isinstance(figs, dict):
            for name, meta in figs.items():
                if isinstance(meta, dict):
                    yield pid, meta.get("figure", name), meta
        elif isinstance(figs, list):
            for meta in figs:
                if isinstance(meta, dict):
                    name = meta.get("figure") or meta.get("name") or meta.get("file")
                    if name:
                        yield pid, name, meta


def download(repo_file: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, repo_file, repo_type="dataset", local_dir=str(cache)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/content/data/spiqa_figs")
    ap.add_argument("--cache_dir", default="/content/data/spiqa_raw")
    ap.add_argument("--max_per_class", type=int, default=3000, help="cap for train (val/test use 1/4 of it)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out, cache = Path(args.out_dir), Path(args.cache_dir)
    rng = random.Random(args.seed)
    rows = []

    for split, (json_file, zip_file) in FILES.items():
        data = json.loads(download(json_file, cache).read_text(encoding="utf-8"))
        by_class = defaultdict(list)
        for pid, name, meta in iter_figures(data):
            label = map_label(meta)
            if label:
                by_class[label].append((pid, Path(name).name, meta.get("caption", "")))
        if not by_class:
            first = next(iter(data.values()))
            raise RuntimeError(
                f"No figures parsed from {json_file}. Keys of first paper: {list(first)[:20]}. "
                "Update iter_figures()/map_label() to match the schema."
            )
        cap = args.max_per_class if split == "train" else max(args.max_per_class // 4, 200)
        chosen = []
        for label, items in by_class.items():
            rng.shuffle(items)
            chosen += [(label, *it) for it in items[:cap]]
        print(split, "available:", {k: len(v) for k, v in by_class.items()}, "-> using", Counter(c[0] for c in chosen))

        zpath = download(zip_file, cache)
        with zipfile.ZipFile(zpath) as zf:
            members = {Path(m).name: m for m in zf.namelist() if m.lower().endswith((".png", ".jpg", ".jpeg"))}
            missing = 0
            for label, pid, name, caption in chosen:
                m = members.get(name)
                if m is None:
                    missing += 1
                    continue
                dst = out / split / label / f"{pid}__{name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.write_bytes(zf.read(m))
                rows.append({"split": split, "label": label, "path": str(dst.relative_to(out)), "paper_id": pid, "caption": caption})
            if missing:
                print(f"  {missing} images listed in {json_file} were not found in {zip_file}")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "label", "path", "paper_id", "caption"])
        w.writeheader()
        w.writerows(rows)
    print("Final counts:", Counter((r["split"], r["label"]) for r in rows))
    print(f"Manifest: {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
