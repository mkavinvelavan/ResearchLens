"""Step 1 of Part 2: split a paper PDF into typed blocks with page numbers, sections and image crops.

Our own rule-based parser on top of pdfplumber (CPU only, no models, instant - so it also runs on the website):
  text lines -> columns (gutter split) -> paragraphs / headings, captions "Figure n: / Table n:", figure/table regions = the
  free space next to the caption, table cells -> rows of text, equations = lines with a set-off "(n)" number, page crops.
  (A layout-model parser such as MinerU was considered and left out: several GB of models, slow on CPU, hard to deploy.)

Output: <out_dir>/doc.json
  {paper_id, title, n_pages, backend, blocks: [
     {id, kind: heading|paragraph|figure|table|equation, text, page (1-based), bbox [x0,top,x1,bottom] (PDF points),
      section, image (path relative to out_dir, figures/tables/equations), label ("Figure 2"), caption, latex,
      mentions: ["Figure 2", "Table 1", "Equation 1"]  (paragraphs that refer to them)}]}
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|table|tab\.)\s*(\d+)\s*[:.|]\s*", re.I)  # "Table 3 shows" is not a caption
MENTION_RE = re.compile(r"\b(fig(?:ure)?s?\.?|tables?|tab\.|eq(?:uation)?s?\.?)\s*~?\(?(\d+)\)?", re.I)
EQNUM_RE = re.compile(r"\((\d{1,3})\)\s*$")
HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.)?\s*[A-Z][A-Za-z0-9 ,:&\-()/]{1,80}$")


def norm_label(kind_word: str, num: str) -> str:
    k = kind_word.lower()
    if k.startswith("fig"):
        return f"Figure {num}"
    if k.startswith("tab"):
        return f"Table {num}"
    return f"Equation {num}"


def paper_id_of(pdf: Path) -> str:
    return hashlib.md5(pdf.read_bytes()).hexdigest()[:12]


# =========================================================================== pdfplumber parser
def _lines(page) -> list[dict]:
    """Words -> rows (same height) -> split each row at the column gutter -> lines with a column id.

    col 0 = spans the page middle (title, wide figure captions), 1 = left/only column, 2 = right column."""
    W = page.width
    mid = W / 2
    words = page.extract_words(extra_attrs=["size", "fontname"], x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False)
    # group by baseline (bottom), not top: small capitals / drop caps share the baseline but not the top
    words.sort(key=lambda w: (round(w["bottom"]), w["x0"]))
    words.sort(key=lambda w: w["bottom"])
    rows, cur = [], []
    for w in words:
        if cur and abs(w["bottom"] - cur[0]["bottom"]) > 2.0:
            rows.append(cur)
            cur = []
        cur.append(w)
    if cur:
        rows.append(cur)
    # is the page two-column? many rows have a gap that covers the middle of the page
    def gutter_split(row):
        row = sorted(row, key=lambda w: w["x0"])
        for a, b in zip(row, row[1:]):
            if b["x0"] - a["x1"] > 6 and a["x1"] < mid + 5 and b["x0"] > mid - 5:
                return [w for w in row if w["x0"] < b["x0"]], [w for w in row if w["x0"] >= b["x0"]]
        return None
    n_split = sum(gutter_split(r) is not None for r in rows)
    left_only = sum(max(w["x1"] for w in r) < mid + 5 for r in rows)
    right_only = sum(min(w["x0"] for w in r) > mid - 5 for r in rows)
    two_col = n_split + min(left_only, right_only) > 4 and right_only + n_split > 3
    out = []
    for row in rows:
        segs = [row]
        if two_col:
            s = gutter_split(row)
            if s:
                segs = list(s)
        for seg in segs:
            seg = sorted(seg, key=lambda w: w["x0"])
            x0, x1 = seg[0]["x0"], max(w["x1"] for w in seg)
            col = 1 if not two_col else (2 if x0 >= mid - 10 else 0 if x1 > mid + 20 else 1)
            text = seg[0]["text"]
            for a, b in zip(seg, seg[1:]):   # pieces split only by a font change (small caps) are one word
                text += ("" if b["x0"] - a["x1"] < 0.12 * max(a.get("size", 10), 6) else " ") + b["text"]
            text = re.sub(r"\(cid:\d+\)", "?", text)
            sizes = [round(w.get("size", 0), 1) for w in seg]
            fonts = [w.get("fontname", "") for w in seg]
            out.append({"text": text, "last_gap": (seg[-1]["x0"] - seg[-2]["x1"]) if len(seg) > 1 else 0,
                        "x0": x0, "x1": x1, "top": min(w["top"] for w in seg),
                        "bottom": max(w["bottom"] for w in seg), "col": col,
                        "size": Counter(sizes).most_common(1)[0][0] if sizes else 0,
                        "bold": sum(any(k in f.lower() for k in ("bold", "medi", "semibold", "black", "heavy", "-bd", "cmbx"))
                                for f in fonts) > len(fonts) / 2})
    return out, two_col


def _numeric_ratio(text: str) -> float:
    toks = text.split()
    return sum(bool(re.fullmatch(r"[-+±]?\d+([.,]\d+)?%?", t)) for t in toks) / max(len(toks), 1)


def _crop(pdf_doc, page_no: int, bbox, out: Path, name: str, scale: float = 2.5) -> str | None:
    x0, top, x1, bottom = bbox
    if x1 - x0 < 15 or bottom - top < 8:
        return None
    img = pdf_doc[page_no - 1].render(scale=scale).to_pil()
    pad = 3
    box = (max(0, (x0 - pad) * scale), max(0, (top - pad) * scale), min(img.width, (x1 + pad) * scale),
           min(img.height, (bottom + pad) * scale))
    path = out / "images" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    crop = img.crop(box)
    from PIL import ImageChops, Image
    bg = Image.new(crop.mode, crop.size, (255, 255, 255))
    content = ImageChops.difference(crop.convert("RGB"), bg.convert("RGB")).convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    if content:                       # trim white margins
        crop = crop.crop((max(0, content[0] - 6), max(0, content[1] - 6), min(crop.width, content[2] + 6),
                          min(crop.height, content[3] + 6)))
    crop.save(path)
    return str(path.relative_to(out))


def table_grid(page, region) -> str:
    """Rows and cells of a table region as text: 'cell | cell | cell' per line (the model reads exact numbers here)."""
    x0, top, x1, bottom = region
    try:
        words = page.within_bbox((max(x0 - 2, 0), max(top - 2, 0), min(x1 + 2, page.width),
                                  min(bottom + 2, page.height))).extract_words(x_tolerance=1.5, y_tolerance=2,
                                                                               extra_attrs=["size"])
    except Exception:
        return ""
    # superscripts (224^2) are smaller and raised: glue them to the word on their left with '^'
    words.sort(key=lambda w: (round(w["top"]), w["x0"]))
    body = [w for w in words if w["size"] >= 0.8 * max(x["size"] for x in words)]
    for w in [w for w in words if w["size"] < 0.8 * max(x["size"] for x in words)]:
        left = [b for b in body if 0 <= w["x0"] - b["x1"] < 1.5 and b["top"] - 2 < w["top"] < b["bottom"]]
        if left:
            left[0]["text"] += "^" + w["text"]
        else:
            body.append(w)
    words = sorted(body, key=lambda w: w["bottom"])
    rows, cur = [], []
    for w in words:
        if cur and abs(w["bottom"] - cur[0]["bottom"]) > 2.5:
            rows.append(cur)
            cur = []
        cur.append(w)
    if cur:
        rows.append(cur)
    lines = []
    for r in rows:
        r.sort(key=lambda w: w["x0"])
        cells, cell = [], [r[0]]
        for a, b in zip(r, r[1:]):
            if b["x0"] - a["x1"] > 5:
                cells.append(" ".join(w["text"] for w in cell))
                cell = []
            cell.append(b)
        cells.append(" ".join(w["text"] for w in cell))
        lines.append(" | ".join(cells))
    return "\n".join(lines)[:4000]


def _is_math_line(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]{3,}", text)
    sym = len(re.findall(r"[=+\-−∑∏∫√≤≥≈∈×·/^_()|]", text))
    return sym >= 2 and len(letters) <= 4


def parse_pdfplumber(pdf: Path, out: Path) -> dict:
    import pdfplumber
    import pypdfium2 as pdfium

    pdf_doc = pdfium.PdfDocument(str(pdf))
    blocks = []
    with pdfplumber.open(str(pdf)) as doc:
        pages = [(p, *_lines(p)) for p in doc.pages]
        sizes = Counter(round(l["size"]) for _, ls, _ in pages for l in ls for _ in range(len(l["text"])))
        body = sizes.most_common(1)[0][0] if sizes else 10
        seen_labels: set[str] = set()
        for pno, (page, lines, two_col) in enumerate(pages, start=1):
            W, H = page.width, page.height
            mid = W / 2
            first_col_top = min([l["top"] for l in lines if l["col"] != 0] or [H])
            order = sorted(lines, key=lambda l: (0 if l["col"] == 0 and l["top"] < first_col_top else
                                                 (3 if l["col"] == 0 else l["col"]), l["top"]))
            col_x = {0: (min([l["x0"] for l in lines] or [0]), max([l["x1"] for l in lines] or [W])),
                     1: (min([l["x0"] for l in lines if l["col"] == 1] or [0]), mid - 4 if two_col else W),
                     2: (mid + 4, max([l["x1"] for l in lines if l["col"] == 2] or [W]))}

            # 1) equations: a line ending in "(n)" plus the short / math / small-font lines right above and below it
            used = set()
            eq_units = []
            for i, l in enumerate(order):
                if EQNUM_RE.search(l["text"]) and l["last_gap"] > 10 and _numeric_ratio(l["text"]) < 0.4 and \
                        (len(l["text"]) < 70 or _is_math_line(l["text"][:-5])):
                    grp = [i]
                    for step in (-1, 1):
                        j = i + step
                        while 0 <= j < len(order):
                            o = order[j]
                            near = abs((o["top"] if step > 0 else o["bottom"]) -
                                       (order[grp[-1] if step > 0 else grp[0]]["bottom" if step > 0 else "top"])) < 14
                            mathy = len(o["text"]) < 25 or _is_math_line(o["text"]) or o["size"] < body * 0.85
                            if o["col"] != l["col"] or not near or not mathy or j in used:
                                break
                            grp = grp + [j] if step > 0 else [j] + grp
                            j += step
                    used.update(grp)
                    ls = [order[k] for k in grp]
                    eq_units.append({"kind": "equation", "text": " ".join(x["text"] for x in ls), "col": l["col"],
                                     "x0": min(x["x0"] for x in ls), "x1": max(x["x1"] for x in ls),
                                     "top": min(x["top"] for x in ls), "bottom": max(x["bottom"] for x in ls),
                                     "num": EQNUM_RE.search(l["text"]).group(1), "n": len(ls), "pos": i})
            # tiny-font fragments (sub/superscripts) elsewhere are noise
            for i, l in enumerate(order):
                if i not in used and l["size"] < body * 0.85 and len(l["text"]) <= 4:
                    used.add(i)

            # 2) the rest -> heading / caption / paragraph units
            units, cur = [], None
            eq_at = {u["pos"]: u for u in eq_units}
            for i, l in enumerate(order):
                if i in eq_at:
                    units.append(eq_at[i])
                    cur = None
                    continue
                if i in used:
                    continue
                t = l["text"]
                is_cap = bool(CAPTION_RE.match(t))
                is_head = not is_cap and len(t) < 90 and HEADING_RE.match(t) is not None and \
                    (l["bold"] or l["size"] > body * 1.3)
                kind = "caption" if is_cap else "heading" if is_head else "line"
                gap_ok = cur is not None and l["col"] == cur["col"] and \
                    -1 <= l["top"] - cur["bottom"] < 0.9 * max(l["bottom"] - l["top"], 6)
                if cur is not None and cur["kind"] == "caption" and cur["text"].rstrip().endswith(".") and \
                        (abs(l["x0"] - cur["x0"]) > 4 or l["x1"] < cur["x1"] - 40):
                    gap_ok = False          # caption finished; the next line is table/figure content
                if kind == "line" and cur is not None and cur["kind"] in ("paragraph", "caption") and gap_ok:
                    cur["text"] = (cur["text"][:-1] if cur["text"].endswith("-") else cur["text"] + " ") + t
                    cur.update(x0=min(cur["x0"], l["x0"]), x1=max(cur["x1"], l["x1"]), bottom=l["bottom"])
                    cur["n"] += 1
                    continue
                cur = {"kind": "paragraph" if kind == "line" else kind, "text": t, "col": l["col"], "x0": l["x0"],
                       "x1": l["x1"], "top": l["top"], "bottom": l["bottom"], "n": 1}
                units.append(cur)
                if kind == "heading":
                    cur = None
            for u in units:   # table rows / axis labels: numeric-heavy short "paragraphs"
                if u["kind"] == "paragraph" and _numeric_ratio(u["text"]) > 0.25 and u["n"] <= 30:
                    u["kind"] = "cells"

            # 3) figure / table region = free space between its caption and the nearest text block in the column
            for u in units:   # a label seen before (e.g. on an earlier page) -> body text, not a second caption
                if u["kind"] == "caption":
                    m = CAPTION_RE.match(u["text"])
                    lab = norm_label(m.group(1), m.group(2))
                    if lab in seen_labels:
                        u["kind"] = "paragraph"
                    seen_labels.add(lab)
            heads = [u["top"] for u in units if u["kind"] == "heading" and
                     re.search(r"abstract|introduction", u["text"], re.I)]
            title_block_end = min(heads) if pno == 1 and heads else -1
            def body_like(u):   # real body text bounds a figure; short labels inside a figure do not
                cw = col_x[u["col"]][1] - col_x[u["col"]][0]
                if u["kind"] != "paragraph":
                    return True
                if u["col"] == 0 and u["bottom"] < first_col_top + 2:   # title / author block above the columns
                    return True
                if u["bottom"] <= title_block_end:                      # page-1 author / affiliation lines
                    return True
                toks = re.findall(r"[A-Za-z]+", u["text"])
                prose = sum(t.islower() for t in toks) / max(len(toks), 1) > 0.5   # figure labels are Capitalised
                return (u["x1"] - u["x0"]) > 0.6 * cw and len(u["text"]) > 80 and prose
            text_units = [u for u in units if (u["kind"] in ("paragraph", "heading", "equation", "caption") and body_like(u))
                          or (u["kind"] == "cells" and u["col"] == 0 and u["bottom"] < first_col_top + 2)]
            for u in units:
                if u["kind"] != "caption":
                    continue
                m = CAPTION_RE.match(u["text"])
                label = norm_label(m.group(1), m.group(2))
                cx0, cx1 = col_x[u["col"]]
                same = [t for t in text_units if t is not u and (t["col"] == u["col"] or t["col"] == 0 or u["col"] == 0)]
                above = max([t["bottom"] for t in same if t["bottom"] <= u["top"] + 1] or [36])
                below = min([t["top"] for t in same if t["top"] >= u["bottom"] - 1] or [H - 36])
                up = (cx0, above + 2, cx1, u["top"] - 2)
                down = (cx0, u["bottom"] + 2, cx1, below - 2)
                prefer = [up, down] if label.startswith("Figure") else [down, up]
                region = next((r for r in prefer if r[3] - r[1] > 25), prefer[0])
                u.update(kind="figure" if label.startswith("Figure") else "table", label=label, caption=u["text"],
                         region=region)
                for t in units:   # text inside the region = table cells / figure labels
                    if t["kind"] in ("paragraph", "cells") and t["top"] >= region[1] - 5 and t["bottom"] <= region[3] + 5 \
                            and t["x0"] >= region[0] - 2 and t["x1"] <= region[2] + 2:
                        t["kind"] = "inside"
                        u.setdefault("inner_text", []).append(t["text"])

            for u in units:
                if u["kind"] in ("inside", "cells"):
                    continue
                b = {"page": pno, "bbox": [round(u["x0"], 1), round(u["top"], 1), round(u["x1"], 1), round(u["bottom"], 1)],
                     "kind": u["kind"], "text": u["text"]}
                if u["kind"] in ("figure", "table"):
                    b.update(label=u["label"], caption=u["caption"], text=" ".join(u.get("inner_text", [])),
                             bbox=[round(v, 1) for v in u["region"]])
                    b["image"] = _crop(pdf_doc, pno, u["region"], out, f"p{pno}_{u['label'].replace(' ', '')}")
                    if u["kind"] == "table":
                        b["table_text"] = table_grid(page, u["region"])
                elif u["kind"] == "equation":
                    b.update(label=f"Equation {u['num']}", text=re.sub(rf"\(\s*{u['num']}\s*\)", "", u["text"]).strip())
                    b["image"] = _crop(pdf_doc, pno, (u["x0"], u["top"] - 2, u["x1"] - 24, u["bottom"] + 2), out,
                                       f"p{pno}_eq{u['num']}")
                blocks.append(b)
    return {"backend": "pdfplumber", "n_pages": len(pdf_doc), "blocks": blocks}


# =========================================================================== common post-processing
# Page-header lines that are not the title, e.g. "Published as a conference paper at ICLR 2021"
RUNNING_HEADER_RE = re.compile(r"^(published as|under review|accepted (at|to|as)|preprint|arxiv:|proceedings of|"
                               r"in proceedings|workshop|to appear)", re.I)


def finalize(doc: dict, pdf: Path) -> dict:
    blocks = doc["blocks"]
    section, title = "", None
    eq_count = 0
    for i, b in enumerate(blocks):
        b["id"] = f"b{i:04d}"
        if b["kind"] == "heading":
            if title is None and b["page"] == 1 and i < 5 and not RUNNING_HEADER_RE.match(b["text"]):
                title = b["text"]
            section = re.sub(r"^\d+(\.\d+)*\.?\s*", "", b["text"]).strip()
        b["section"] = section
        if b["kind"] == "paragraph" and re.match(r"^(references|bibliography|literature cited)$", section, re.I):
            b["kind"] = "reference"      # not searched, no entities: conference names would pollute everything
        if b["kind"] in ("figure", "table") and not b.get("label"):
            m = CAPTION_RE.match(b.get("caption") or "")
            if m:
                b["label"] = norm_label(m.group(1), m.group(2))
        if b["kind"] == "equation" and not b.get("label"):
            eq_count += 1
            b["label"] = f"Equation {eq_count}"
        if b["kind"] == "paragraph":
            b["mentions"] = sorted({norm_label(k, n) for k, n in MENTION_RE.findall(b["text"])})
        if b["kind"] == "paragraph" and section.lower().startswith("abstract") is False and b["text"].lower().startswith("abstract"):
            b["section"] = "Abstract"
    # equation context: the paragraph just before and after (where its variables are usually explained)
    paras = [i for i, b in enumerate(blocks) if b["kind"] == "paragraph"]
    for i, b in enumerate(blocks):
        if b["kind"] == "equation":
            before = [blocks[j]["text"] for j in paras if j < i][-1:]
            after = [blocks[j]["text"] for j in paras if j > i][:1]
            b["context"] = " ".join(before + after)[:1200]
    title = title or next((b["text"] for b in blocks if b["kind"] == "paragraph"), pdf.stem)
    return {"paper_id": paper_id_of(pdf), "file": pdf.name, "title": title, **doc}


def parse(pdf: str | Path, out_dir: str | Path, backend: str = "pdfplumber") -> dict:
    pdf, out = Path(pdf), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = parse_pdfplumber(pdf, out)
    doc = finalize(doc, pdf)
    (out / "doc.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    counts = Counter(b["kind"] for b in doc["blocks"])
    print(f"{pdf.name}: {doc['n_pages']} pages, backend {doc['backend']}: {dict(counts)}")
    return doc


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    parse(a.pdf, a.out_dir)
