"""Add biomedical figures (BIOMEDICA, PubMed Central) to the figure-type data -> multi-domain figure classifier.

BIOMEDICA (Lozano et al., CVPR 2025) is a gated WebDataset (accept the terms on Hugging Face first). Every shard of
the release holds figures of ONE fine-grained concept (e.g. "surgical procedure"), stored in `image_primary_label`.
So instead of downloading whole shards (1-2.5 GB each) we:
  1. scan:   stream only the first sample of every shard -> {shard: concept}            (~1-3 min)
  2. map:    concept -> our 5 classes with keyword rules (RULES below; printed for the report)
  3. sample: for each class, stream ~N figures from shards of MANY different concepts, stopping early
             -> varied data (bar charts, line plots, scatter plots, ... all count as "plot")
Only the Commercial-Use subset is used (CC0 / CC BY / CC BY-SA / CC BY-ND).

Output: <out_dir>/manifest.csv = SPIQA (linked as <out_dir>/spiqa) + biomedical figures, with a `domain` column.
Biomedical figures are split by article (article_accession_id), so one article never spans train and test.

    python -m researchlens.vision.prepare_biomedica_figures --scan_only          # list concepts -> class
    python -m researchlens.vision.prepare_biomedica_figures --spiqa_dir /content/data/spiqa_figs \
        --out_dir /content/data/figs_multidomain --n_train 800 --n_val 100 --n_test 300
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from researchlens.vision.prepare_spiqa_figures import CLASSES

REPO = "BIOMEDICA/biomedica_webdataset_24M"
# (regex on the lower-cased concept, class) - first match wins; everything else -> "other"
RULES = [
    (r"\btable", "table"),
    (r"diagram|schem|flow ?chart|illustrat|pathway|graph and network|network|cartoon|infographic|timeline|workflow|"
     r"mechanism|model", "schematic"),
    (r"chart|plot|graph|histogram|curve|scatter|heat ?map|forest|kaplan|survival|spectr|statistic|quantif", "plot"),
    (r"microscop|histolog|patholog|staining|photo|surgical|surgery|endoscop|x-ray|xray|radiograph|\bct\b|\bmri\b|"
     r"ultrasound|clinical|dermatolog|fundus|retina|animal|specimen|tissue|cell|imaging|scan|tomograph|"
     r"angiograph|mammogra|patient|lesion|wound|plant|organism|blot|\bgel\b|electrophoresis", "photo"),
]


def map_concept(name: str) -> str:
    n = name.lower()
    for pat, cls in RULES:
        if re.search(pat, n):
            return cls
    return "other"


def concept_of(meta: dict) -> str | None:
    v = meta.get("image_primary_label")
    if v is None:
        keys = [k for k in meta if "primary" in k.lower() and "label" in k.lower()]
        v = meta[keys[0]] if keys else None
    if isinstance(v, list):
        v = v[0] if v else None
    return v.strip() if isinstance(v, str) and v.strip() else None


def stream_samples(repo_file: str, token: str | None, max_samples: int):
    """Yield (key, {ext: bytes}) from the start of a shard over HTTP, stopping after max_samples (no full download)."""
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(REPO, repo_file, repo_type="dataset")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        cur, parts, n = None, {}, 0
        try:
            with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    base = os.path.basename(m.name)
                    key = os.path.join(os.path.dirname(m.name), base.partition(".")[0])
                    if cur is not None and key != cur:
                        yield cur, parts
                        n += 1
                        if n >= max_samples:
                            return
                        parts = {}
                    cur = key
                    parts[base.partition(".")[2].lower()] = tf.extractfile(m).read()
        except (tarfile.ReadError, EOFError):
            pass
        if cur is not None and parts and n < max_samples:
            yield cur, parts


def scan(shards: list[str], token: str | None, workers: int = 16) -> dict[str, str]:
    def first_concept(f):
        try:
            for _, parts in stream_samples(f, token, 1):
                if "json" in parts:
                    return f, concept_of(json.loads(parts["json"]))
        except Exception as e:
            return f, f"__error__ {type(e).__name__}"
        return f, None

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(first_concept, shards))
    errors = [c for _, c in res if c and c.startswith("__error__")]
    if errors:
        print(f"  {len(errors)} shards could not be read: {Counter(errors).most_common(3)}")
    return {f: c for f, c in res if c and not c.startswith("__error__")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spiqa_dir", default="/content/data/spiqa_figs")
    ap.add_argument("--out_dir", default="/content/data/figs_multidomain")
    ap.add_argument("--subset", default="commercial")
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--per_concept", type=int, default=120, help="max figures taken from one concept (variety)")
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan_only", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"biomedica_{args.subset}_shard_concepts.json"
    if cache.exists():
        shard_concept = json.loads(cache.read_text())
    else:
        shards = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                        if f.startswith(args.subset + "/") and f.endswith(".tar"))
        print(f"Scanning the first sample of {len(shards)} shards ...", flush=True)
        shard_concept = scan(shards, token)
        cache.write_text(json.dumps(shard_concept, indent=1))
    by_concept: dict[str, list[str]] = defaultdict(list)
    for f, c in shard_concept.items():
        by_concept[c].append(f)
    by_class: dict[str, list[str]] = defaultdict(list)
    for c in by_concept:
        by_class[map_concept(c)].append(c)
    print(f"{len(shard_concept)} shards, {len(by_concept)} concepts")
    for cls in CLASSES:
        print(f"  {cls:<10} {len(by_class[cls]):>4} concepts: {sorted(by_class[cls])}")
    if args.scan_only:
        return

    (out / "biomedical").mkdir(exist_ok=True)
    spiqa_link = out / "spiqa"
    if not spiqa_link.exists():
        spiqa_link.symlink_to(Path(args.spiqa_dir).resolve(), target_is_directory=True)

    rng = random.Random(args.seed)
    need = args.n_train + args.n_val + args.n_test
    kept: dict[str, list[dict]] = defaultdict(list)
    for cls in CLASSES:
        concepts = by_class[cls][:]
        rng.shuffle(concepts)
        for c in concepts:
            if len(kept[cls]) >= need:
                break
            take = min(args.per_concept, need - len(kept[cls]))
            shard = rng.choice(by_concept[c])
            got = 0
            try:
                for key, parts in stream_samples(shard, token, take * 2):
                    img_ext = next((e for e in ("jpg", "jpeg", "png", "webp") if e in parts), None)
                    if "json" not in parts or not img_ext:
                        continue
                    meta = json.loads(parts["json"])
                    try:
                        img = Image.open(io.BytesIO(parts[img_ext])).convert("RGB")
                    except Exception:
                        continue
                    img.thumbnail((args.max_side, args.max_side))
                    dst = out / "biomedical" / cls / f"{re.sub(r'[^A-Za-z0-9_-]', '_', key)}.png"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img.save(dst, optimize=True)
                    txt = parts.get("txt")
                    kept[cls].append({"label": cls, "path": str(dst.relative_to(out)), "concept": c,
                                      "paper_id": str(meta.get("article_accession_id") or meta.get("PMID") or ""),
                                      "caption": (txt.decode("utf-8", "ignore") if txt else "").strip()})
                    got += 1
                    if got >= take:
                        break
            except Exception as e:
                print(f"  skip {shard} ({c}): {type(e).__name__} {e}")
            print(f"  {cls:<10} +{got:<4} from '{c}'  -> {len(kept[cls])}/{need}", flush=True)

    rows = []
    with open(Path(args.spiqa_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({**r, "path": f"spiqa/{r['path']}", "domain": "spiqa"})
    used = []
    for cls in CLASSES:
        # order by a hash of the article id: figures of one article stay together -> no train/test leakage
        items = sorted(kept[cls], key=lambda it: hashlib.md5((it["paper_id"] or it["path"]).encode()).hexdigest())
        n_test = min(args.n_test, len(items) // 4)
        n_val = min(args.n_val, len(items) // 10)
        for j, it in enumerate(items):
            split = "test" if j < n_test else "validation" if j < n_test + n_val else "train"
            rows.append({"split": split, "label": it["label"], "path": it["path"], "paper_id": it["paper_id"],
                         "caption": it["caption"], "domain": "biomedical"})
            used.append({"class": cls, "concept": it["concept"], "split": split})
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "label", "path", "paper_id", "caption", "domain"])
        w.writeheader()
        w.writerows(rows)
    (out / "biomedica_concepts_used.json").write_text(json.dumps(
        {cls: Counter(u["concept"] for u in used if u["class"] == cls) for cls in CLASSES}, indent=1))
    print("Biomedical counts:", sorted(Counter((r["split"], r["label"]) for r in rows if r["domain"] == "biomedical").items()))
    print(f"Manifest: {out / 'manifest.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
