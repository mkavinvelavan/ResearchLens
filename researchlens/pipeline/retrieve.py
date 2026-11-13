"""Step 4 of Part 2: dynamic multimodal retrieval (no intent classifier).

For a question:
  1. text search   bge(question) against every block's text vector         -> top `k_candidates`
  2. re-rank       our SciBERT evidence finder scores (question, block text)  -> top `k_text`
  3. image search  SigLIP(question) against figure/table image vectors     -> top `k_image`
  4. explicit      "Figure 3", "Table 2", "Eq. (1)" named in the question     -> always included
  5. linked        figures/tables mentioned by the top paragraphs            -> added (text<->figure link)
Every item gets an evidence id (E1, E2, ...) that the answer must cite -> Evidence Trace.
"""
from __future__ import annotations

import re

from researchlens.pipeline.index import COLLECTION, client, embed_siglip_text, embed_texts
from researchlens.pipeline.parse_pdf import MENTION_RE, norm_label


def _filter(paper_ids, kinds=None):
    from qdrant_client import models as qm

    must = [qm.FieldCondition(key="paper_id", match=qm.MatchAny(any=list(paper_ids)))]
    if kinds:
        must.append(qm.FieldCondition(key="kind", match=qm.MatchAny(any=list(kinds))))
    return qm.Filter(must=must)


def _search(qc, vec, using, flt, limit):
    res = qc.query_points(COLLECTION, query=vec.tolist(), using=using, query_filter=flt, limit=limit, with_payload=True)
    return [(p.score, p.payload) for p in res.points]


def evidence_scores(bank, question: str, texts: list[str], batch_size: int = 32) -> list[float]:
    import torch

    tok, m = bank.evidence
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok([question] * len(texts[i:i + batch_size]), texts[i:i + batch_size], truncation="only_second",
                  max_length=256, padding=True, return_tensors="pt").to(bank.device)
        with torch.no_grad():
            out += torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().tolist()
    return out


def by_labels(qc, paper_ids, labels):
    from qdrant_client import models as qm

    if not labels:
        return []
    flt = _filter(paper_ids)
    flt.must.append(qm.FieldCondition(key="label", match=qm.MatchAny(any=list(labels))))
    pts, _ = qc.scroll(COLLECTION, scroll_filter=flt, limit=20, with_payload=True)
    return [p.payload for p in pts]


def retrieve(bank, question: str, paper_ids: list[str], qc=None, k_candidates: int = 24, k_text: int = 5,
             k_image: int = 2, rerank: bool = True, kinds: list[str] | None = None) -> list[dict]:
    qc = qc or client()
    qv = embed_texts(bank, [question], query=True)[0]
    per_paper = max(k_candidates // max(len(paper_ids), 1), 8)
    cands = []
    for pid in paper_ids:   # search each paper separately so a comparison gets evidence from both papers
        cands += _search(qc, qv, "text", _filter([pid], kinds), per_paper)
    items = [{**p, "retrieval_score": round(s, 4), "via": "text search"} for s, p in cands]
    if rerank and items:
        scores = evidence_scores(bank, question, [p.get("search_text") or p.get("text", "") for p in items])
        for it, s in zip(items, scores):
            it["evidence_score"] = round(s, 4)
        items.sort(key=lambda x: (-x["evidence_score"], -x["retrieval_score"]))
    else:
        items.sort(key=lambda x: -x["retrieval_score"])
    keep_n = k_text * len(paper_ids) if len(paper_ids) > 1 else k_text
    if len(paper_ids) > 1:   # balanced: top items of each paper
        keep = []
        for pid in paper_ids:
            keep += [x for x in items if x["paper_id"] == pid][:k_text]
    else:
        keep = items[:keep_n]

    seen = {(x["paper_id"], x["block_id"]) for x in keep}

    def add(p, via, score=None):
        key = (p["paper_id"], p["block_id"])
        if key not in seen:
            seen.add(key)
            keep.append({**p, "via": via, **({"retrieval_score": round(score, 4)} if score is not None else {})})

    # explicit references in the question
    labels = {norm_label(k, n) for k, n in MENTION_RE.findall(question)}
    for p in by_labels(qc, paper_ids, labels):
        add(p, "named in question")
    # image search with SigLIP
    if k_image and (kinds is None or {"figure", "table"} & set(kinds)):
        sv = embed_siglip_text(bank, [question])[0]
        for pid in paper_ids:
            for s, p in _search(qc, sv, "image", _filter([pid], ["figure", "table"]), k_image):
                add(p, "image search (SigLIP)", s)
    # figures / tables the top paragraphs refer to
    linked = {lab for x in keep[:4] for lab in (x.get("mentions") or [])}
    for p in by_labels(qc, paper_ids, linked):
        add(p, "mentioned by retrieved text")

    for i, x in enumerate(keep, 1):
        x["eid"] = f"E{i}"
    return keep


def describe(item: dict, max_chars: int = 700) -> str:
    """How one evidence item is shown to the language model."""
    loc = f"[{item['eid']}] {item.get('paper_title', '')[:50]} | p.{item['page']} | {item.get('section') or '-'}"
    k = item["kind"]
    if k in ("figure", "table"):
        body = f"{item.get('label') or k}: {item.get('caption') or ''}"
        if item.get("table_text"):
            body += ("\nTABLE CONTENT (rows, cells separated by |; section headers like '(a) ...' / '(b) ...' apply to "
                     f"the rows below them):\n{item['table_text'][:1800]}")
            max_chars = max(max_chars, 2300)
        else:
            extra = " ".join(x for x in (item.get("text"), item.get("ocr_text")) if x)
            if extra:
                body += f"\nText/values inside: {extra[:500]}"
        if item.get("figure_type"):
            body += f"\n(figure type: {item['figure_type']}; image attached)" if item.get("image") else ""
    elif k == "equation":
        lx = item.get("latex") if item.get("latex_ok", True) else None
        body = f"{item.get('label')}: " + (f"$${lx}$$" if lx else "(read the equation from its image)") + f"\nSurrounding text: {(item.get('context') or '')[:500]}"
    else:
        body = item.get("text", "")
    return f"{loc}\n{body[:max_chars]}"
