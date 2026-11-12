"""SciERC loader (Luan et al., 2018): scientific entities + relations from 500 AI abstracts.

Entity types : Task, Method, Metric, Material, OtherScientificTerm, Generic
Relation types: USED-FOR, FEATURE-OF, HYPONYM-OF, PART-OF, COMPARE, CONJUNCTION, EVALUATE-FOR

We use the processed JSON release (same files used by DyGIE++/PURE):
one document per line with "sentences", "ner", "relations"; span indices are
document-level token offsets with inclusive ends.
"""
from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

SCIERC_URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
ENTITY_TYPES = ["Task", "Method", "Metric", "Material", "OtherScientificTerm", "Generic"]
RELATION_TYPES = ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "COMPARE", "CONJUNCTION", "EVALUATE-FOR"]
BIO_LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]


def download_scierc(data_dir: Path) -> Path:
    """Download + extract SciERC once; returns the folder holding train/dev/test.json."""
    data_dir = Path(data_dir)
    for cand in [data_dir, *data_dir.rglob("json")]:
        if (cand / "train.json").exists():
            return cand
    data_dir.mkdir(parents=True, exist_ok=True)
    tgz = data_dir / "sciERC_processed.tar.gz"
    if not tgz.exists():
        print(f"Downloading SciERC from {SCIERC_URL} ...")
        urllib.request.urlretrieve(SCIERC_URL, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(data_dir)
    for cand in data_dir.rglob("train.json"):
        return cand.parent
    raise FileNotFoundError(
        "SciERC train.json not found after extraction. Download sciERC_processed.tar.gz manually "
        f"from {SCIERC_URL} and pass --data_dir to the folder containing train/dev/test.json."
    )


def read_docs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sentence_examples(docs: list[dict]) -> list[dict]:
    """Split docs into sentences with sentence-local spans.

    Returns dicts: {id, tokens, entities:[(s,e,type)], relations:[(s1,e1,s2,e2,type)], n_dropped_nested}
    """
    out = []
    for doc in docs:
        offset = 0
        for si, toks in enumerate(doc["sentences"]):
            ents = [(s - offset, e - offset, t) for s, e, t in doc["ner"][si]]
            rels = [(a - offset, b - offset, c - offset, d - offset, t) for a, b, c, d, t in doc["relations"][si]]
            out.append({"id": f"{doc['doc_key']}_{si}", "tokens": toks, "entities": ents, "relations": rels})
            offset += len(toks)
    return out


def spans_to_bio(n_tokens: int, entities: list[tuple]) -> tuple[list[str], int]:
    """Convert spans to BIO tags. SciERC has a few nested spans; BIO cannot hold them,
    so we keep the longest non-overlapping spans and report how many were dropped."""
    tags = ["O"] * n_tokens
    dropped = 0
    for s, e, t in sorted(entities, key=lambda x: -(x[1] - x[0])):
        if any(tags[i] != "O" for i in range(s, e + 1)):
            dropped += 1
            continue
        tags[s] = f"B-{t}"
        for i in range(s + 1, e + 1):
            tags[i] = f"I-{t}"
    return tags, dropped


def load_ner_splits(data_dir: Path) -> dict[str, list[dict]]:
    folder = download_scierc(data_dir)
    splits = {}
    total_dropped = 0
    for split, fname in [("train", "train.json"), ("validation", "dev.json"), ("test", "test.json")]:
        exs = sentence_examples(read_docs(folder / fname))
        for ex in exs:
            ex["ner_tags"], d = spans_to_bio(len(ex["tokens"]), ex["entities"])
            total_dropped += d
        splits[split] = exs
    print({k: len(v) for k, v in splits.items()}, f"nested spans dropped for BIO: {total_dropped}")
    return splits
