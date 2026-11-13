"""Step 3 of Part 2: turn every block into vectors and store it in Qdrant with its metadata.

One collection, two named vectors per point:
  text   BAAI/bge-small-en-v1.5 (384-d)  - paragraphs, figure/table captions + text inside them, equations + context
  image  SigLIP (768-d, same space as SigLIP text) - figure and table images, so a question can find a figure directly
Payload: paper_id, paper_title, block_id, kind, page, section, label, caption, text, image, figure_type, latex,
         variables, entities, relations, mentions, mentioned_in.

Qdrant runs locally (embedded, a folder on disk) by default, or on Qdrant Cloud when QDRANT_URL (+ QDRANT_API_KEY) is set.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np

COLLECTION = os.environ.get("RL_COLLECTION", "researchlens")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def client(path: str | Path | None = None):
    from qdrant_client import QdrantClient

    if os.environ.get("QDRANT_URL"):
        return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    return QdrantClient(path=str(path or os.environ.get("RL_QDRANT_PATH", "qdrant_data")))


def ensure_collection(qc, text_dim: int = 384, image_dim: int = 768):
    from qdrant_client import models as qm

    if not qc.collection_exists(COLLECTION):
        qc.create_collection(COLLECTION, vectors_config={
            "text": qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE),
            "image": qm.VectorParams(size=image_dim, distance=qm.Distance.COSINE)})
        for field in ("paper_id", "kind", "label"):
            try:
                qc.create_payload_index(COLLECTION, field, qm.PayloadSchemaType.KEYWORD)
            except Exception:
                pass   # local mode has no payload indexes


# ------------------------------------------------------------------ embedders
def embed_texts(bank, texts: list[str], query: bool = False, batch_size: int = 32) -> np.ndarray:
    import torch

    tok, m = bank.text_embedder
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [(QUERY_PREFIX + t) if query else t for t in texts[i:i + batch_size]]
        enc = tok(chunk, truncation=True, max_length=512, padding=True, return_tensors="pt").to(bank.device)
        with torch.no_grad():
            h = m(**enc).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


def _feat(x):
    """transformers v5 may return a model output instead of a tensor from get_*_features."""
    return x if hasattr(x, "norm") else (getattr(x, "pooler_output", None) if getattr(x, "pooler_output", None) is not None
                                        else x[0])


def embed_images(bank, paths: list[Path], batch_size: int = 16) -> np.ndarray:
    import torch
    from PIL import Image

    proc, m = bank.siglip
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        px = proc(images=imgs, return_tensors="pt").to(bank.device)
        with torch.no_grad():
            f = _feat(m.get_image_features(**px))
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)


def embed_siglip_text(bank, texts: list[str]) -> np.ndarray:
    import torch

    proc, m = bank.siglip
    enc = proc(text=texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(bank.device)
    with torch.no_grad():
        f = _feat(m.get_text_features(**enc))
    return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


# ------------------------------------------------------------------ block -> searchable text
def block_text(b: dict) -> str:
    k = b["kind"]
    if k == "paragraph":
        return f"{b.get('section', '')}: {b['text']}".strip(": ")
    if k in ("figure", "table"):
        inner = " ".join(x for x in (b.get("text"), b.get("ocr_text")) if x)[:800]
        typ = f" ({b['figure_type']})" if b.get("figure_type") else ""
        return f"{b.get('label', k)}{typ}: {b.get('caption', '')} {('Content: ' + inner) if inner else ''}".strip()
    if k == "equation":
        return f"{b.get('label', 'Equation')}: {b.get('latex') or b.get('text', '')}. Context: {b.get('context', '')}"[:1500]
    return b.get("text", "")


def index_paper(out_dir: str | Path, bank, qc=None) -> int:
    from qdrant_client import models as qm

    out_dir = Path(out_dir)
    doc = json.loads((out_dir / "doc.json").read_text(encoding="utf-8"))
    qc = qc or client()
    ensure_collection(qc)
    blocks = [b for b in doc["blocks"] if b["kind"] in ("paragraph", "figure", "table", "equation")
              and (b["kind"] != "paragraph" or len(b["text"]) > 40)]
    tvec = embed_texts(bank, [block_text(b) for b in blocks])
    img_blocks = [b for b in blocks if b["kind"] in ("figure", "table") and b.get("image") and (out_dir / b["image"]).exists()]
    ivec = dict(zip([b["id"] for b in img_blocks], embed_images(bank, [out_dir / b["image"] for b in img_blocks])))
    points = []
    for b, tv in zip(blocks, tvec):
        vec = {"text": tv.tolist()}
        if b["id"] in ivec:
            vec["image"] = ivec[b["id"]].tolist()
        payload = {"paper_id": doc["paper_id"], "paper_title": doc.get("title", ""), "block_id": b["id"],
                   "kind": b["kind"], "page": b["page"], "section": b.get("section", ""), "label": b.get("label"),
                   "caption": b.get("caption"), "text": b.get("text", "")[:4000], "search_text": block_text(b)[:2000],
                   "image": str((out_dir / b["image"]).resolve()) if b.get("image") else None,
                   "figure_type": b.get("figure_type"), "latex": b.get("latex"), "variables": b.get("variables"),
                   "context": b.get("context"), "entities": b.get("entities", [])[:40],
                   "relations": b.get("relations", [])[:30], "mentions": b.get("mentions", []),
                   "mentioned_in": b.get("mentioned_in", []), "ocr_text": b.get("ocr_text")}
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['paper_id']}/{b['id']}"))
        points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))
    for i in range(0, len(points), 128):
        qc.upsert(COLLECTION, points[i:i + 128])
    print(f"indexed {len(points)} blocks of '{doc.get('title', '')[:60]}' ({len(ivec)} with image vectors)")
    return len(points)


def delete_paper(paper_id: str, qc=None) -> None:
    from qdrant_client import models as qm

    qc = qc or client()
    if qc.collection_exists(COLLECTION):
        qc.delete(COLLECTION, points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))])))
