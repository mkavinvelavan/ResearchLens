"""Step 5 of Part 2: Gemini (Google AI Studio API, free tier) reads the retrieved evidence (text AND figure/table/equation images) and answers,
citing evidence ids. The Evidence Trace maps every cited id back to paper, page, section, block and why it was retrieved.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from researchlens.pipeline.retrieve import describe

SYSTEM = (
    "You are ResearchLens, an assistant that helps students understand scientific papers. "
    "Use ONLY the numbered evidence provided (text passages, figure/table images and captions, equations). "
    "After every claim, cite the evidence it comes from in square brackets, e.g. [E2] or [E1][E4]. "
    "For any number taken from a table, use the TABLE CONTENT text (not only the image) and state the exact row and "
    "the table section it comes from, e.g. 'Swin-B, 384^2, (b) ImageNet-22K pre-trained: 86.4 [E3]'. Check that the "
    "row matches every condition in the question (model, setting, resolution, pre-training). "
    "If the evidence does not contain the answer, say so plainly. "
    "Do not invent numbers, datasets, or results."
)
CITE_RE = re.compile(r"\[(E\d+)\]")


def _image(path: str | None, max_side: int = 512):
    from PIL import Image

    if not path or not Path(path).exists():
        return None
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def build_messages(evidence: list[dict], instruction: str, max_images: int = 2) -> list[dict]:
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


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"   # Gemini's OpenAI-compatible endpoint
# Flash-Lite first: fastest and, in our tests, least often busy; the bigger Flash models are the fallback.
GEMINI_MODELS = os.environ.get("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash,gemini-3.8-flash,"
                               "gemini-3.1-flash-lite")
LAST_MODEL = {"name": None}          # which Gemini model produced the last answer (shown in the app)


def gemini_models() -> list[str]:
    return [m.strip() for m in GEMINI_MODELS.split(",") if m.strip()]


def _to_openai(messages: list[dict]) -> list[dict]:
    """Our chat format (PIL images inside) -> OpenAI format (images as base64 data URLs)."""
    import base64
    import io

    out = []
    for m in messages:
        parts = []
        for c in m["content"]:
            if c["type"] == "text":
                parts.append({"type": "text", "text": c["text"]})
            else:
                buf = io.BytesIO()
                c["image"].save(buf, format="JPEG", quality=90)
                url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
                parts.append({"type": "image_url", "image_url": {"url": url}})
        out.append({"role": m["role"], "content": parts})
    return out


class _Retry(Exception):
    pass


def _call(key, model, body, on_text):
    import requests

    try:
        r = requests.post(f"{GEMINI_URL}/chat/completions", json={**body, "model": model}, stream=on_text is not None,
                          timeout=180, headers={"Authorization": f"Bearer {key}"})
    except requests.RequestException as e:
        raise _Retry(f"{model}: {e}")
    if r.status_code in (404, 408, 429) or r.status_code >= 500:     # removed / busy / overloaded -> next model
        raise _Retry(f"{model}: HTTP {r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        raise RuntimeError(f"Gemini API error {r.status_code} ({model}): {r.text[:400]}")
    if on_text is None:
        try:
            content = r.json()["choices"][0]["message"].get("content") or ""
        except Exception as e:
            raise _Retry(f"{model}: unreadable reply ({type(e).__name__})")
        if not content.strip():          # e.g. a thinking model used all tokens before writing -> next model
            raise _Retry(f"{model}: empty answer")
        return content.strip()
    text = ""
    for line in r.iter_lines():
        line = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
        except Exception:
            continue
        if delta:
            text += delta
            on_text(text)
    if not text.strip():
        raise _Retry(f"{model}: empty answer")
    return text.strip()


def generate(bank, messages: list[dict], max_new_tokens: int = 900, temperature: float = 0.0, on_text=None) -> str:
    """Gemini reads the evidence (text + images) and answers. on_text(partial_text) is called while tokens arrive.
    Free models are sometimes busy, so each model in GEMINI_MODELS is tried in turn. `bank` is unused (kept so all
    callers stay the same). Needs the GEMINI_API_KEY secret (Google AI Studio)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set: create a free key at https://aistudio.google.com/apikey and "
                           "add it as a secret.")
    body = {"messages": _to_openai(messages), "max_tokens": max_new_tokens, "temperature": temperature,
            "stream": on_text is not None}
    errors = []
    t0 = time.time()
    for model in gemini_models():
        try:
            text = _call(key, model, body, on_text)
            LAST_MODEL["name"] = model
            print(f"[Gemini] answered by {model} in {time.time() - t0:.0f}s", flush=True)
            return text
        except _Retry as e:
            errors.append(str(e))
            print(f"[Gemini] {e} -> trying next", flush=True)
    raise RuntimeError("All free Gemini models are busy right now. Please try again in a minute.\n" + "\n".join(errors))


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
            "why": (e.get("via") or "retrieved") + (f", evidence-finder score {e['evidence_score']:.2f}" if e.get("evidence_score") is not None
                                   else ""),
        })
    return trace


def ask(bank, question: str, paper_ids: list[str], qc=None, on_text=None, **retrieve_kw) -> dict:
    from researchlens.pipeline.retrieve import retrieve

    ev = retrieve(bank, question, paper_ids, qc=qc, **retrieve_kw)
    instr = (f"Question: {question}\nAnswer clearly for a student in a few short paragraphs or bullet points, "
             "citing evidence ids after each claim.")
    ans = generate(bank, build_messages(ev, instr), max_new_tokens=900, on_text=on_text)
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
