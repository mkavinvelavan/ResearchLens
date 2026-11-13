"""Step 5 of Part 2: Qwen3-VL reads the retrieved evidence (text AND figure/table/equation images) and answers,
citing evidence ids. The Evidence Trace maps every cited id back to paper, page, section, block and why it was retrieved.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from researchlens.pipeline.retrieve import describe

SYSTEM = (
    "You are ResearchLens, an assistant that helps students understand scientific papers. "
    "Use ONLY the numbered evidence provided (text passages, figure/table images and captions, equations). "
    "After every claim, cite the evidence it comes from in square brackets, e.g. [E2] or [E1][E4]. "
    "Read values from figure and table images carefully. If the evidence does not contain the answer, say so plainly. "
    "Do not invent numbers, datasets, or results."
)
CITE_RE = re.compile(r"\[(E\d+)\]")


def _image(path: str | None, max_side: int = 640):
    from PIL import Image

    if not path or not Path(path).exists():
        return None
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def build_messages(evidence: list[dict], instruction: str, max_images: int = 3) -> list[dict]:
    content, n_img = [], 0
    for it in evidence:
        img = _image(it.get("image")) if it["kind"] in ("figure", "table", "equation") and n_img < max_images else None
        if img is not None:
            content.append({"type": "text", "text": f"Image of [{it['eid']}] ({it.get('label') or it['kind']}):"})
            content.append({"type": "image", "image": img})
            n_img += 1
    ev_text = "\n\n".join(describe(it) for it in evidence)
    content.append({"type": "text", "text": f"EVIDENCE\n{ev_text}\n\nTASK\n{instruction}"})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content}]


def generate(bank, messages: list[dict], max_new_tokens: int = 700, temperature: float = 0.0) -> str:
    import torch

    proc, model = bank.vlm
    inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(model.device)
    kw = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    del inputs, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


def evidence_trace(answer: str, evidence: list[dict]) -> list[dict]:
    cited = list(dict.fromkeys(CITE_RE.findall(answer)))
    by_id = {e["eid"]: e for e in evidence}
    trace = []
    for eid in cited:
        e = by_id.get(eid)
        if not e:
            continue
        trace.append({
            "eid": eid, "paper": e.get("paper_title"), "paper_id": e["paper_id"], "page": e["page"],
            "section": e.get("section"), "kind": e["kind"], "label": e.get("label"), "block_id": e["block_id"],
            "snippet": (e.get("caption") or e.get("latex") or e.get("text") or "")[:300], "image": e.get("image"),
            "why": e.get("via") + (f", evidence-finder score {e['evidence_score']:.2f}" if e.get("evidence_score") is not None
                                   else ""),
        })
    return trace


def ask(bank, question: str, paper_ids: list[str], qc=None, **retrieve_kw) -> dict:
    from researchlens.pipeline.retrieve import retrieve

    ev = retrieve(bank, question, paper_ids, qc=qc, **retrieve_kw)
    instr = (f"Question: {question}\nAnswer clearly for a student in a few short paragraphs or bullet points, "
             "citing evidence ids after each claim.")
    ans = generate(bank, build_messages(ev, instr))
    return {"question": question, "answer": ans, "trace": evidence_trace(ans, ev), "evidence": ev,
            "uncited": [e["eid"] for e in ev if e["eid"] not in CITE_RE.findall(ans)]}


def parse_json(text: str):
    """Pull the first JSON object/array out of a model reply (```json fences allowed)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    start = min([i for i in (s.find("{"), s.find("[")) if i >= 0] or [-1])
    if start < 0:
        return None
    for end in range(len(s), start, -1):
        try:
            return json.loads(s[start:end])
        except Exception:
            continue
    return None
