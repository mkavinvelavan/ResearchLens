"""Part 3 features built on retrieval + Qwen3-VL: overview, figure/formula explanation, two-paper comparison,
Research Gap Map and Idea Generator. Every output cites evidence ids, so each has an Evidence Trace.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from researchlens.pipeline.answer import build_messages, evidence_trace, generate, parse_json
from researchlens.pipeline.retrieve import by_labels, retrieve

OVERVIEW_ASPECTS = {
    "Research problem": "What problem does the paper address and why does it matter?",
    "Objectives": "What are the goals and contributions of this paper?",
    "Methodology": "What method, model architecture or proposed approach is used?",
    "Datasets": "Which datasets or data are used for training and evaluation?",
    "Key results": "What are the main experimental results and numbers?",
    "Limitations": "What are the limitations or weaknesses?",
    "Future work": "What future work is suggested?",
}
COMPARE_ASPECTS = ["Research problem", "Objectives", "Methodology", "Datasets", "Models", "Mathematical approach",
                   "Experiments and metrics", "Results", "Strengths", "Limitations", "Future work"]


def _merge(lists, cap):
    seen, out = set(), []
    for lst in lists:
        for it in lst:
            k = (it["paper_id"], it["block_id"])
            if k not in seen:
                seen.add(k)
                out.append(dict(it))
    out = out[:cap]
    for i, it in enumerate(out, 1):
        it["eid"] = f"E{i}"
    return out


def load_doc(out_dir) -> dict:
    doc = json.loads((Path(out_dir) / "doc.json").read_text(encoding="utf-8"))
    doc["out_dir"] = str(Path(out_dir).resolve())
    return doc


def entity_summary(doc: dict, top: int = 8) -> dict:
    by_type = {}
    for e in doc.get("entities", []):
        by_type.setdefault(e["type"], [])
        if len(by_type[e["type"]]) < top:
            by_type[e["type"]].append(e["text"])
    return by_type


# ------------------------------------------------------------------ single paper
def overview(bank, doc: dict, qc=None, on_text=None) -> dict:
    pid = doc["paper_id"]
    lists = [retrieve(bank, q, [pid], qc=qc, k_text=2, k_image=0) for q in OVERVIEW_ASPECTS.values()]
    lists.append(retrieve(bank, "main results table or results figure", [pid], qc=qc, k_text=1, k_image=1,
                          kinds=["figure", "table"]))
    eqs = [dict(b, paper_id=pid, block_id=b["id"], paper_title=doc.get("title", ""), via="equation in paper",
                image=str(Path(doc.get("out_dir", ".")) / b["image"]) if b.get("image") else None)
           for b in doc["blocks"] if b["kind"] == "equation"][:3]
    ev = _merge(lists + [eqs], 15)
    ents = entity_summary(doc)
    instr = (
        f"Write a structured overview of the paper \"{doc.get('title', '')}\" with these markdown sections: "
        + ", ".join(list(OVERVIEW_ASPECTS)[:5]) + ", Important formulas, " + ", ".join(list(OVERVIEW_ASPECTS)[5:])
        + ". Under each heading write 2-4 bullet points with citations [E#]. Under 'Important formulas' explain what "
        "each equation computes and what its variables mean. If a section is not covered by the evidence, write "
        "'Not stated in the retrieved evidence.'\n"
        f"Entities our NER model found most often (for reference): {json.dumps(ents)}")
    text = generate(bank, build_messages(ev, instr), max_new_tokens=1400, on_text=on_text)
    return {"overview": text, "trace": evidence_trace(text, ev), "evidence": ev, "entities": ents}


def explain_item(bank, doc: dict, label: str, qc=None, on_text=None) -> dict:
    """Figure / table / equation explanation, incl. Formula -> Concept -> Result."""
    pid = doc["paper_id"]
    target = by_labels(qc, [pid], [label]) if qc is not None else []
    if not target:
        from researchlens.pipeline.index import client
        target = by_labels(client(), [pid], [label])
    if not target:
        return {"answer": f"{label} was not found in this paper.", "trace": [], "evidence": []}
    t = target[0]
    kind = t["kind"]
    q = (f"{label} {t.get('caption') or ''} {t.get('latex') or ''}").strip()
    ctx = retrieve(bank, q, [pid], qc=qc, k_text=4, k_image=0, kinds=["paragraph"])
    if kind == "equation":
        res = retrieve(bank, f"results of the method using {label} " + " ".join(t.get("variables") or []), [pid],
                       qc=qc, k_text=2, k_image=1, kinds=["table", "figure", "paragraph"])
        instr = (f"Explain {label} for a student in three parts: (1) Formula - what it computes, term by term, and "
                 f"what each variable means; (2) Concept - where and why it is used in the methodology; (3) Result - "
                 "which experiment or result depends on it. Cite evidence ids.")
        ev = _merge([[dict(t, via="requested item")], ctx, res], 10)
    else:
        instr = (f"Explain {label}: what type of {kind} it is, what it shows (axes, legend, rows/columns), the key "
                 "numbers or trends, and what conclusion the paper draws from it. Cite evidence ids.")
        ev = _merge([[dict(t, via="requested item")], ctx], 8)
    text = generate(bank, build_messages(ev, instr), max_new_tokens=1200, on_text=on_text)
    return {"answer": text, "trace": evidence_trace(text, ev), "evidence": ev}


# ------------------------------------------------------------------ two papers
def shared_entities(doc_a: dict, doc_b: dict) -> dict:
    """Deterministic overlap of what NER found (no language model): shared / only-A / only-B per type."""
    out = {}
    for ty in ("Method", "Material", "Task", "Metric"):
        a = {e["text"] for e in doc_a.get("entities", []) if e["type"] == ty}
        b = {e["text"] for e in doc_b.get("entities", []) if e["type"] == ty}
        out[ty] = {"shared": sorted(a & b)[:15], "only_a": sorted(a - b)[:15], "only_b": sorted(b - a)[:15]}
    return out


def compare(bank, doc_a: dict, doc_b: dict, qc=None, on_text=None) -> dict:
    ids = [doc_a["paper_id"], doc_b["paper_id"]]
    qs = ["research problem and objectives", "proposed method and model architecture", "datasets used",
          "evaluation metrics and experimental setup", "main results", "limitations and future work"]
    lists = [retrieve(bank, q, ids, qc=qc, k_text=2, k_image=0) for q in qs]
    lists.append(retrieve(bank, "main results table or figure", ids, qc=qc, k_text=1, k_image=1, kinds=["figure", "table"]))
    ev = _merge(lists, 18)
    overlap = shared_entities(doc_a, doc_b)
    instr = (
        f"Compare Paper A \"{doc_a.get('title', '')}\" and Paper B \"{doc_b.get('title', '')}\". "
        "First write a markdown table with rows: " + ", ".join(COMPARE_ASPECTS) + " and columns: Aspect | Paper A | "
        "Paper B, with citations [E#] in the cells. Then write 'Key similarities' and 'Key differences' as bullet "
        "points, and compare their figures/tables where images are provided.\n"
        f"Entity overlap found by our NER model: {json.dumps(overlap)}")
    text = generate(bank, build_messages(ev, instr, max_images=2), max_new_tokens=1600, on_text=on_text)
    return {"comparison": text, "trace": evidence_trace(text, ev), "evidence": ev, "entity_overlap": overlap}


GAP_TYPES = ["methodological", "dataset", "evaluation", "performance", "shared limitation"]


def gap_map(bank, doc_a: dict, doc_b: dict, qc=None) -> dict:
    ids = [doc_a["paper_id"], doc_b["paper_id"]]
    qs = ["limitations and weaknesses", "future work", "datasets and their size or domain",
          "evaluation metrics and baselines", "assumptions of the method", "results where performance is low"]
    ev = _merge([retrieve(bank, q, ids, qc=qc, k_text=2, k_image=0) for q in qs], 16)
    instr = (
        "Identify research gaps across the two papers. Each gap must be supported by the evidence. Return ONLY JSON: "
        '{"gaps": [{"title": "...", "type": one of ' + json.dumps(GAP_TYPES) + ', "papers": ["A"|"B"|"both"], '
        '"description": "...", "evidence": ["E1", ...]}]}. Give 3 to 6 gaps. Paper A = "' + doc_a.get("title", "")[:80]
        + '", Paper B = "' + doc_b.get("title", "")[:80] + '".')
    text = generate(bank, build_messages(ev, instr, max_images=0), max_new_tokens=1000)
    data = parse_json(text) or {}
    gaps = [g for g in (data.get("gaps") if isinstance(data, dict) else data) or [] if isinstance(g, dict)]
    by_id = {e["eid"]: e for e in ev}
    for g in gaps:   # keep only evidence ids that exist -> no unsupported gaps
        g["evidence"] = [x for x in g.get("evidence", []) if x in by_id]
        g["supported"] = bool(g["evidence"])
    return {"gaps": gaps, "raw": text, "evidence": ev,
            "trace": evidence_trace(" ".join(f"[{x}]" for g in gaps for x in g["evidence"]), ev)}


def ideas(bank, gaps: list[dict], doc_a: dict, doc_b: dict | None = None) -> dict:
    titles = [doc_a.get("title", "")] + ([doc_b.get("title", "")] if doc_b else [])
    instr = (
        "Based ONLY on these research gaps, propose one potential research direction per gap. Return ONLY JSON: "
        '{"ideas": [{"gap": "<gap title>", "idea": "...", "why_relevant": "...", "methodology": "...", '
        '"datasets": "...", "evaluation": "...", "risk": "..."}]}. These are suggestions to explore, not proven '
        f"solutions. Papers: {titles}.\nGAPS:\n{json.dumps(gaps, indent=1)[:6000]}")
    msgs = [{"role": "system", "content": [{"type": "text", "text": "You are a careful research advisor."}]},
            {"role": "user", "content": [{"type": "text", "text": instr}]}]
    text = generate(bank, msgs, max_new_tokens=1200, temperature=0.3)
    data = parse_json(text) or {}
    return {"ideas": (data.get("ideas") if isinstance(data, dict) else data) or [], "raw": text}
