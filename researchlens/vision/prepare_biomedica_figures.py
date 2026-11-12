"""Add biomedical figures (BIOMEDICA, PubMed Central) to the figure-type data -> multi-domain figure classifier.

BIOMEDICA (Lozano et al., CVPR 2025) is a gated WebDataset (accept the terms on Hugging Face first). Each sample has an
image, a caption and JSON metadata with expert-curated "primary" concept labels. We stream a few shards of the
Commercial-Use subset (CC0 / CC BY / CC BY-SA / CC BY-ND), map the concepts to our 5 classes and keep a small,
class-balanced sample. Shards are deleted after reading (each is ~1.5 GB).

    Plots and Charts                                   -> plot
    Illustrative Diagrams, Graph and Network           -> schematic
    Tables                                             -> table
    Microscopy, Clinical Imaging, Tools and Materials,
    Immuno Assays, (photographs)                       -> photo
    everything else (formulae, chemical structures, maps, hand drawn / screen visuals, PCR, ...) -> other

Output: <out_dir>/manifest.csv covering SPIQA (linked as <out_dir>/spiqa) + biomedical figures, with a `domain` column.
    python -m researchlens.vision.prepare_biomedica_figures --spiqa_dir /content/data/spiqa_figs \
        --out_dir /content/data/figs_multidomain --n_train 800 --n_val 100 --n_test 300
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from researchlens.vision.prepare_spiqa_figures import CLASSES

REPO = "BIOMEDICA/biomedica_webdataset_24M"
RULES = [  # (keywords in the concept name, class) - first match wins
    (("plot", "chart"), "plot"),
    (("table",), "table"),
    (("illustrative", "diagram", "graph and network", "network"), "schematic"),
    (("microscop", "clinical imaging", "tools and materials", "immuno", "photograph", "radiolog"), "photo"),
]


def map_concept(name: str) -> str:
    n = name.lower()
    for keys, cls in RULES:
        if any(k in n for k in keys):
            return cls
    return "other"


def find_label(meta: dict):
    """Primary content label, whatever the exact key is called (image_primary_label, primary_label, ...)."""
    keys = [k for k in meta if "primary" in k.lower() and "label" in k.lower()]
    keys.sort(key=lambda k: ("image" not in k.lower(), len(k)))
    for k in keys:
        v = meta[k]
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def find_caption(meta: dict, txt: str | None) -> str:
    if txt:
        return txt.strip()
    for k in ("caption", "image_caption", "figure_caption"):
        if isinstance(meta.get(k), str):
            return meta[k].strip()
    return ""


def iter_samples(tar_path: Path):
    """Yield (key, {ext: bytes}) from a WebDataset shard (files of one sample are adjacent)."""
    cur, parts = None, {}
    with tarfile.open(tar_path, "r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            key, _, ext = base.partition(".")
            key = os.path.join(os.path.dirname(m.name), key)
            if cur is not None and key != cur:
                yield cur, parts
                parts = {}
            cur = key
            parts[ext.lower()] = tf.extractfile(m).read()
        if cur is not None:
            yield cur, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spiqa_dir", default="/content/data/spiqa_figs")
    ap.add_argument("--out_dir", default="/content/data/figs_multidomain")
    ap.add_argument("--subset", default="commercial")
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--max_shards", type=int, default=6)
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    out = Path(args.out_dir)
    (out / "biomedical").mkdir(parents=True, exist_ok=True)
    spiqa_link = out / "spiqa"
    if not spiqa_link.exists():
        spiqa_link.symlink_to(Path(args.spiqa_dir).resolve(), target_is_directory=True)

    shards = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                    if f.startswith(args.subset + "/") and f.endswith(".tar"))
    rng = random.Random(args.seed)
    rng.shuffle(shards)
    need = args.n_train + args.n_val + args.n_test
    kept: dict[str, list[dict]] = defaultdict(list)
    concepts = Counter()
    shown_keys = False

    for i, shard in enumerate(shards[: args.max_shards]):
        if all(len(kept[c]) >= need for c in CLASSES):
            break
        print(f"[{i + 1}/{args.max_shards}] downloading {shard} ...", flush=True)
        path = Path(hf_hub_download(REPO, shard, repo_type="dataset", local_dir="/tmp/biomedica"))
        n_seen = 0
        for key, parts in iter_samples(path):
            meta_raw = parts.get("json")
            img_ext = next((e for e in ("jpg", "jpeg", "png", "webp") if e in parts), None)
            if not meta_raw or not img_ext:
                continue
            meta = json.loads(meta_raw)
            if not shown_keys:
                print("metadata keys:", sorted(meta)[:40], flush=True)
                shown_keys = True
            concept = find_label(meta)
            if concept is None:
                continue
            n_seen += 1
            concepts[concept] += 1
            cls = map_concept(concept)
            if len(kept[cls]) >= need:
                continue
            try:
                img = Image.open(io.BytesIO(parts[img_ext])).convert("RGB")
            except Exception:
                continue
            img.thumbnail((args.max_side, args.max_side))
            name = key.replace("/", "_").replace(".", "_")
            dst = out / "biomedical" / cls / f"{name}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, optimize=True)
            txt = parts.get("txt").decode("utf-8", "ignore") if parts.get("txt") else None
            kept[cls].append({"label": cls, "path": str(dst.relative_to(out)), "concept": concept,
                              "paper_id": str(meta.get("PMID", meta.get("pmid", meta.get("article_id", "")))),
                              "caption": find_caption(meta, txt)})
        path.unlink(missing_ok=True)
        print(f"  read {n_seen} labelled figures | kept so far: { {c: len(kept[c]) for c in CLASSES} }", flush=True)

    print("BIOMEDICA concepts seen (top 20):", concepts.most_common(20))
    rows = []
    with open(Path(args.spiqa_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({**r, "path": f"spiqa/{r['path']}", "domain": "spiqa"})
    for cls in CLASSES:
        import hashlib
        # order by a hash of the article id: figures of one article stay together -> no train/test leakage
        items = sorted(kept[cls], key=lambda it: hashlib.md5((it["paper_id"] or it["path"]).encode()).hexdigest())
        n_test = min(args.n_test, len(items) // 4)
        n_val = min(args.n_val, len(items) // 10)
        for j, it in enumerate(items):
            split = "test" if j < n_test else "validation" if j < n_test + n_val else "train"
            rows.append({"split": split, "label": it["label"], "path": it["path"], "paper_id": it["paper_id"],
                         "caption": it["caption"], "domain": "biomedical"})
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "label", "path", "paper_id", "caption", "domain"])
        w.writeheader()
        w.writerows(rows)
    print("Biomedical counts:", Counter((r["split"], r["label"]) for r in rows if r["domain"] == "biomedical"))
    print(f"Manifest: {out / 'manifest.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
