"""Step 2 of Part 2: run our fine-tuned models over the parsed paper (doc.json -> doc.json with annotations).

  paragraphs  -> sentences -> NER (Method / Material / Task / Metric)
              -> relation extractor on every entity pair in a sentence (USED-FOR, EVALUATE-FOR, ...)
  figures/tables -> figure type = average of ViT (image) and SciBERT (caption) probabilities
                    (the late fusion that scored best in our tests: 93.6% accuracy) + text inside the figure
  equations   -> TrOCR LaTeX (checked against the PDF text) + the surrounding explanation
  links       -> "Figure 2 / Table 1 / Eq. (3)" mentions in paragraphs <-> the figure/table/equation blocks
"""
from __future__ import annotations

import json
import time
import re
from pathlib import Path

import numpy as np

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
WORD_RE = re.compile(r"\w+(?:[-.]\w+)*|[^\w\s]")
PTB = {"(": "-LRB-", ")": "-RRB-", "[": "-LSB-", "]": "-RSB-", "{": "-LCB-", "}": "-RCB-"}
FIG_CLASSES = ["plot", "schematic", "photo", "table", "other"]
REL_TEXT = {"USED-FOR": "used for", "FEATURE-OF": "feature of", "HYPONYM-OF": "is a kind of", "PART-OF": "part of",
            "COMPARE": "compared with", "CONJUNCTION": "together with", "EVALUATE-FOR": "evaluates"}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# ------------------------------------------------------------------ NER
def run_ner(bank, sents: list[str], batch_size: int = 32) -> list[list[dict]]:
    import torch

    tok, model = bank.ner
    id2label = model.config.id2label
    out = []
    for chunk in batched(sents, batch_size):
        words = [WORD_RE.findall(s)[:200] for s in chunk]
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            pred = model(**{k: v.to(bank.device) for k, v in enc.items()}).logits.argmax(-1).cpu().numpy()
        for bi, ws in enumerate(words):
            tags, seen = ["O"] * len(ws), set()
            for ti, wi in enumerate(enc.word_ids(bi)):
                if wi is not None and wi not in seen:
                    seen.add(wi)
                    tags[wi] = id2label[int(pred[bi, ti])]
            ents, cur = [], None
            for wi, t in enumerate(tags + ["O"]):
                if t.startswith("B-") or t == "O" or (t.startswith("I-") and (cur is None or cur["type"] != t[2:])):
                    if cur:
                        cur["text"] = " ".join(ws[cur["start"]:cur["end"] + 1])
                        ents.append(cur)
                        cur = None
                    if t != "O":
                        cur = {"type": t[2:], "start": wi, "end": wi}
                else:
                    cur["end"] = wi
            out.append([{**e, "words": ws} for e in ents])
    return out


# ------------------------------------------------------------------ relations
# On CPU hosting (low-memory mode) only nearby entity pairs are scored: most SciERC relations link entities a few words
# apart, and scoring every pair of a long sentence is too slow on 2 CPU cores.
MAX_PAIR_GAP_CPU = 15
MAX_PAIRS_CPU = 1500


def run_relations(bank, sent_ents: list[tuple[str, list[dict]]], min_conf: float = 0.5, batch_size: int = 64):
    """sent_ents: [(sentence_id, entities with 'words', 'start', 'end')] -> list of triples."""
    import torch

    from researchlens.nlp.train_relation import mark

    tok, enc_model, head, cfg = bank.relation
    labels = cfg["labels"]
    items = []
    for sid, ents in sent_ents:
        for a in ents:
            for b in ents:
                if a is b or (a["start"] <= b["end"] and b["start"] <= a["end"]):
                    continue
                gap = max(a["start"], b["start"]) - min(a["end"], b["end"])   # words between the two entities
                if bank.low_memory and gap > MAX_PAIR_GAP_CPU:
                    continue
                ptb = [PTB.get(w, w) for w in a["words"]]   # SciERC uses -LRB- / -RRB- for brackets
                ws, i1, i2 = mark(ptb, (a["start"], a["end"]), (b["start"], b["end"]))
                items.append((sid, a, b, ws, i1, i2, gap))
    if bank.low_memory and len(items) > MAX_PAIRS_CPU:     # CPU hosting: keep the closest pairs
        items = sorted(items, key=lambda c: c[6])[:MAX_PAIRS_CPU]
    print(f"[relations] scoring {len(items)} entity pairs", flush=True)
    items.sort(key=lambda c: len(c[3]))                     # similar lengths per batch -> less padding, faster
    triples = []
    for chunk in batched(items, batch_size):
        enc = tok([c[3] for c in chunk], is_split_into_words=True, truncation=True, max_length=cfg.get("max_length", 160),
                  padding=True, return_tensors="pt")
        m1, m2 = [], []
        for bi, c in enumerate(chunk):
            first = {}
            for ti, wi in enumerate(enc.word_ids(bi)):
                if wi is not None and wi not in first:
                    first[wi] = ti
            m1.append(first.get(c[4], 0))
            m2.append(first.get(c[5], 0))
        with torch.no_grad():
            hs = enc_model(**{k: v.to(bank.device) for k, v in enc.items()}).last_hidden_state
            idx = torch.arange(hs.size(0), device=hs.device)
            rep = torch.cat([hs[idx, torch.tensor(m1, device=hs.device)], hs[idx, torch.tensor(m2, device=hs.device)]], -1)
            probs = torch.softmax(head(rep).float(), -1).cpu().numpy()
        for c, p in zip(chunk, probs):
            k = int(p.argmax())
            if labels[k] != "NONE" and p[k] >= min_conf:
                triples.append({"sentence_id": c[0], "head": c[1]["text"], "head_type": c[1]["type"],
                                "relation": labels[k], "relation_text": REL_TEXT.get(labels[k], labels[k].lower()),
                                "tail": c[2]["text"], "tail_type": c[2]["type"], "conf": round(float(p[k]), 3)})
    return triples


# ------------------------------------------------------------------ figures
def figure_types(bank, out_dir: Path, figs: list[dict]) -> None:
    import torch
    from PIL import Image

    from researchlens.vision.figdata import build_transforms

    proc, vit = bank.figtype
    tok, cap = bank.captype
    tf = build_transforms(proc, train=False)
    vit_labels = [vit.config.id2label[i] for i in range(vit.config.num_labels)]
    cap_labels = [cap.config.id2label[i] for i in range(cap.config.num_labels)]
    for b in figs:
        p_img = p_cap = None
        if b.get("image") and (out_dir / b["image"]).exists():
            x = tf(Image.open(out_dir / b["image"])).unsqueeze(0).to(bank.device)
            with torch.no_grad():
                pr = torch.softmax(vit(pixel_values=x).logits.float(), -1)[0].cpu().numpy()
            p_img = np.array([pr[vit_labels.index(c)] for c in FIG_CLASSES])
        if b.get("caption"):
            enc = tok(b["caption"], truncation=True, max_length=128, return_tensors="pt").to(bank.device)
            with torch.no_grad():
                pr = torch.softmax(cap(**enc).logits.float(), -1)[0].cpu().numpy()
            p_cap = np.array([pr[cap_labels.index(c)] for c in FIG_CLASSES])
        ps = [p for p in (p_img, p_cap) if p is not None]
        if not ps:
            continue
        p = np.mean(ps, 0)
        b["figure_type"] = FIG_CLASSES[int(p.argmax())]
        b["figure_type_conf"] = round(float(p.max()), 3)
        b["figure_type_sources"] = {"image": None if p_img is None else FIG_CLASSES[int(p_img.argmax())],
                                    "caption": None if p_cap is None else FIG_CLASSES[int(p_cap.argmax())]}


def figure_ocr(bank, out_dir: Path, figs: list[dict]) -> None:
    ocr = bank.ocr
    if ocr is None:
        return
    for b in figs:
        if not b.get("image"):
            continue
        try:
            res = ocr.predict(str(out_dir / b["image"]))
            texts = [t for r in res for t in (r.get("rec_texts") or [])]
        except Exception:
            try:
                res = ocr.ocr(str(out_dir / b["image"]))
                texts = [ln[1][0] for page in res or [] for ln in page or []]
            except Exception:
                texts = []
        if texts:
            b["ocr_text"] = " ".join(texts)[:2000]


# ------------------------------------------------------------------ formulas
def split_lines(img, min_gap: int = 6, min_h: int = 12):
    """Cut a multi-line equation image into one image per line (TrOCR was trained on single-line formulas)."""
    import numpy as np

    g = np.asarray(img.convert("L"), dtype=np.float32)
    ink = (g < 160).sum(1)
    rows = ink > max(1, 0.002 * g.shape[1])
    lines, start, gap = [], None, 0
    for y, on in enumerate(rows):
        if on:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                if y - gap - start + 1 >= min_h:
                    lines.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None and len(rows) - start >= min_h:
        lines.append((start, len(rows)))
    if len(lines) <= 1:
        return [img]
    return [img.crop((0, max(0, s - 3), img.width, min(img.height, e + 3))) for s, e in lines]


def latex_is_garbage(lx: str) -> bool:
    """TrOCR on an out-of-distribution image loops: '\\ \\ \\ ...' or one token repeated."""
    from collections import Counter

    toks = lx.split()
    if len(toks) < 2:
        return not lx.strip()
    tok, n = Counter(toks).most_common(1)[0]
    return (n / len(toks) > 0.45 and len(toks) > 8) or sum(t in ("\\", "\\!", "\\,", "\\;") for t in toks) / len(toks) > 0.4


def latex_agrees_with_pdf(latex: str, pdf_text: str):
    """Share of the multi-letter names in the PDF's own equation text (MSA, MLP, LN, softmax ...) that the LaTeX also
    contains. None when the PDF text has too few names to judge. Low -> TrOCR misread the image."""
    names = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", pdf_text)} - {"cid", "where", "and", "the", "of", "for"}
    if len(names) < 2:
        return None
    lx = re.sub(r"\\(mathrm|mathbf|boldsymbol|text|operatorname|mathcal|bf|cal|rm|left|right|begin|end|array)", " ", latex)
    lx = re.sub(r"[^A-Za-z]", "", lx).lower()
    return round(sum(n in lx for n in names) / len(names), 2)


def formulas(bank, out_dir: Path, eqs: list[dict], batch_size: int = 8) -> None:
    import torch
    from PIL import Image

    proc, model = bank.formula
    for b in eqs:
        if not (b.get("image") and (out_dir / b["image"]).exists()):
            continue
        lines = split_lines(Image.open(out_dir / b["image"]).convert("RGB"))
        outs = []
        for chunk in batched(lines, batch_size):
            px = proc(images=chunk, return_tensors="pt").pixel_values.to(bank.device)
            with torch.no_grad():
                ids = model.generate(px, max_new_tokens=160, num_beams=3)
            outs += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
        good = [o for o in outs if not latex_is_garbage(o)]
        b["latex_lines"] = len(lines)
        b["latex_trocr"] = " \\\\ ".join(good)
        agree = latex_agrees_with_pdf(b["latex_trocr"], b.get("text") or "")
        b["latex_agreement"] = agree
        b["latex_ok"] = bool(good) and len(good) >= max(1, len(outs) // 2) and (agree is None or agree >= 0.5)
        if not b.get("latex") and b["latex_ok"]:
            b["latex"] = b["latex_trocr"]
    for b in eqs:   # variables = single letters / greek commands appearing in the LaTeX
        lx = b.get("latex") or ""
        b["variables"] = sorted(set(re.findall(r"\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|tau|phi|"
                                               r"omega|pi|eta|rho|psi|xi|nu|kappa|zeta)|(?<![\\a-zA-Z])[a-zA-Z](?![a-zA-Z])", lx)))[:15]


# ------------------------------------------------------------------ main
def enrich(out_dir: str | Path, bank=None, steps=("ner", "relations", "figures", "ocr", "formulas")) -> dict:
    from researchlens.pipeline.models import ModelBank

    out_dir = Path(out_dir)
    doc = json.loads((out_dir / "doc.json").read_text(encoding="utf-8"))
    bank = bank or ModelBank()
    blocks = doc["blocks"]

    # mentions <-> figures/tables/equations (the text-figure link)
    by_label = {b["label"]: b for b in blocks if b.get("label")}
    for b in blocks:
        for lab in b.get("mentions") or []:
            if lab in by_label:
                by_label[lab].setdefault("mentioned_in", []).append(b["id"])

    paras = [b for b in blocks if b["kind"] == "paragraph" and len(b["text"]) > 30]
    sents = [(b["id"], i, s) for b in paras for i, s in enumerate(sentences(b["text"]))]
    if "ner" in steps and sents:
        t = time.time()
        ents = run_ner(bank, [s for _, _, s in sents])
        bank.release("ner")
        print(f"[time] NER on {len(sents)} sentences: {time.time() - t:.0f}s", flush=True)
        for (bid, i, s), es in zip(sents, ents):
            b = next(x for x in paras if x["id"] == bid)
            b.setdefault("entities", []).extend(
                {"text": e["text"], "type": e["type"], "sentence": i} for e in es)
        if "relations" in steps:
            t_rel = time.time()
            sent_ents = [(f"{bid}:{i}", es) for (bid, i, _), es in zip(sents, ents) if len(es) >= 2]
            for t in run_relations(bank, sent_ents):
                bid, i = t["sentence_id"].split(":")
                b = next(x for x in paras if x["id"] == bid)
                b.setdefault("relations", []).append({**{k: v for k, v in t.items() if k != "sentence_id"},
                                                     "sentence": int(i)})
            bank.release("relation")
            print(f"[time] relations: {time.time() - t_rel:.0f}s", flush=True)
    figs = [b for b in blocks if b["kind"] in ("figure", "table")]
    if "figures" in steps and figs:
        t = time.time()
        figure_types(bank, out_dir, figs)
        bank.release("figtype", "captype")
        print(f"[time] figure types ({len(figs)}): {time.time() - t:.0f}s", flush=True)
    if "ocr" in steps and figs:
        figure_ocr(bank, out_dir, figs)
    eqs = [b for b in blocks if b["kind"] == "equation"]
    if "formulas" in steps and eqs:
        t = time.time()
        formulas(bank, out_dir, eqs)
        bank.release("formula")
        print(f"[time] formulas ({len(eqs)}): {time.time() - t:.0f}s", flush=True)

    # paper-level entity / relation summary (used by the overview and comparison)
    from collections import Counter

    ent_count = Counter((e["text"].lower(), e["type"]) for b in paras for e in b.get("entities", []))
    doc["entities"] = [{"text": t, "type": ty, "count": c} for (t, ty), c in ent_count.most_common(200)]
    rel_seen = {}
    for b in paras:
        for r in b.get("relations", []):
            k = (r["head"].lower(), r["relation"], r["tail"].lower())
            if k not in rel_seen or r["conf"] > rel_seen[k]["conf"]:
                rel_seen[k] = {**r, "block": b["id"], "page": b["page"]}
    doc["relations"] = sorted(rel_seen.values(), key=lambda r: -r["conf"])[:300]
    doc["enriched"] = True
    (out_dir / "doc.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"enriched: {sum(len(b.get('entities', [])) for b in paras)} entity mentions, {len(doc['relations'])} relations, "
          f"{len(figs)} figures/tables, {len(eqs)} equations")
    return doc
