"""Multi-domain scientific NER data: SciERC (AI) + SemEval-2017 ScienceIE (Computer Science, Materials Science, Physics).

Both are mapped to one shared label set:
    Method   <- SciERC Method,   ScienceIE Process
    Material <- SciERC Material, ScienceIE Material   (datasets, materials, resources)
    Task     <- SciERC Task,     ScienceIE Task
    Metric   <- SciERC Metric    (ScienceIE has no metric label -> Metric is scored on SciERC only)
SciERC OtherScientificTerm / Generic -> O (no counterpart in ScienceIE).
"""
from __future__ import annotations

from pathlib import Path

UNIFIED_TYPES = ["Method", "Material", "Task", "Metric"]
UNIFIED_LABELS = ["O"] + [f"{p}-{t}" for t in UNIFIED_TYPES for p in ("B", "I")]
SCIENCEIE_TAGS = ["O", "B-Material", "I-Material", "B-Process", "I-Process", "B-Task", "I-Task"]
SCIENCEIE_HUB = "DFKI-SLT/science_ie"
DOMAIN_TYPES = {"scierc": set(UNIFIED_TYPES), "scienceie": {"Method", "Material", "Task"}}


def to_unified(tag: str) -> str:
    if tag == "O":
        return "O"
    p, t = tag.split("-", 1)
    t = {"Process": "Method"}.get(t, t)
    return f"{p}-{t}" if t in UNIFIED_TYPES else "O"


def restrict(tags: list[str], domain: str) -> list[str]:
    """Drop entity types a domain was never annotated with (e.g. Metric on ScienceIE) before scoring."""
    keep = DOMAIN_TYPES[domain]
    return [t if t == "O" or t.split("-", 1)[1] in keep else "O" for t in tags]


def fix_bio(tags: list[str]) -> list[str]:
    """I-X that does not continue an X span becomes B-X (keeps strict IOB2 valid after relabelling)."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def load_scienceie() -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset(SCIENCEIE_HUB, "ner")
    names = ds["train"].features["tags"].feature.names
    out = {}
    for split, key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        out[split] = [{"id": f"scienceie_{r['id']}", "tokens": r["tokens"], "domain": "scienceie",
                       "ner_tags": fix_bio([to_unified(names[i]) for i in r["tags"]])} for r in ds[key]]
    return out


def load_multidomain_splits(scierc_dir: Path, datasets: list[str]) -> dict[str, list[dict]]:
    """Merged train / validation / test with a `domain` field and unified tags."""
    from researchlens.nlp.data_scierc import load_ner_splits

    splits = {"train": [], "validation": [], "test": []}
    if "scierc" in datasets:
        for s, exs in load_ner_splits(Path(scierc_dir)).items():
            splits[s] += [{**e, "domain": "scierc", "ner_tags": fix_bio([to_unified(t) for t in e["ner_tags"]])}
                          for e in exs]
    if "scienceie" in datasets:
        for s, exs in load_scienceie().items():
            splits[s] += exs
    for s, exs in splits.items():
        by = {d: sum(e["domain"] == d for e in exs) for d in datasets}
        print(f"{s}: {len(exs)} sentences {by}")
    return splits
