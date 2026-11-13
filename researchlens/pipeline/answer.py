"""Step 5 of Part 2: Qwen3-VL reads the retrieved evidence (text AND figure/table/equation images) and answers,
citing evidence ids. The Evidence Trace maps every cited id back to paper, page, section, block and why it was retrieved.
"""
from __future__ import annotations

import json
import os
import re
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


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
GEMINI_MODELS = os.environ.get("GEMINI_MODELS", "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite,"
                               "gemini-3.1-flash-lite")
OPENROUTER_MODELS = os.environ.get("OPENROUTER_MODELS", "google/gemma-4-31b-it:free,nvidia/nemotron-3.5-super-vl:free,"
                                   "google/gemma-4-26b-a4b-it:free")
LAST_MODEL = {"name": None}          # which provider/model produced the last answer (shown in the app)


def api_providers() -> list[tuple[str, str, str]]:
    """(base_url, key, model) to try in order. Free tiers are often busy, so the app falls back to the next one.
    Secrets: GEMINI_API_KEY (Google AI Studio, free) and/or OPENROUTER_API_KEY (free models; QWEN_API_KEY also accepted).
    A fully custom endpoint: VLM_API_URL + VLM_API_KEY + VLM_API_MODEL (e.g. a paid Qwen3-VL)."""
    out = []
    if os.environ.get("VLM_API_KEY") and os.environ.get("VLM_API_URL"):
        out.append((os.environ["VLM_API_URL"], os.environ["VLM_API_KEY"], os.environ.get("VLM_API_MODEL", "")))
    if os.environ.get("GEMINI_API_KEY"):
        out += [(GEMINI_URL, os.environ["GEMINI_API_KEY"], m.strip()) for m in GEMINI_MODELS.split(",") if m.strip()]
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("QWEN_API_KEY")
    if key:
        out += [(OPENROUTER_URL, key, m.strip()) for m in OPENROUTER_MODELS.split(",") if m.strip()]
    return out


def use_api() -> bool:
    """RL_VLM_MODE=api|local. Default: api when any API key is set (deployed website), otherwise local Qwen3-VL on GPU."""
    mode = os.environ.get("RL_VLM_MODE", "auto")
    return mode == "api" or (mode == "auto" and bool(api_providers()))


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


def _call(url, key, model, body, on_text):
    import requests

    try:
        r = requests.post(f"{url.rstrip('/')}/chat/completions", json={**body, "model": model}, stream=on_text is not None,
                          timeout=180, headers={"Authorization": f"Bearer {key}", "X-Title": "ResearchLens"})
    except requests.RequestException as e:
        raise _Retry(f"{model}: {e}")
    if r.status_code in (404, 408, 429) or r.status_code >= 500:
        raise _Retry(f"{model}: HTTP {r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        raise RuntimeError(f"Vision-language API error {r.status_code} ({model}): {r.text[:400]}")
    if on_text is None:
        return r.json()["choices"][0]["message"]["content"].strip()
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


def generate_api(messages: list[dict], max_new_tokens: int = 900, temperature: float = 0.0, on_text=None) -> str:
    """Any OpenAI-compatible endpoint (Google AI Studio Gemini, OpenRouter, a paid Qwen3-VL ...). Tries each
    provider/model in api_providers() order; a busy (429), removed (404) or failing (5xx) one moves on to the next."""
    providers = api_providers()
    if not providers:
        raise RuntimeError("No API key set: add GEMINI_API_KEY and/or OPENROUTER_API_KEY as a secret.")
    body = {"messages": _to_openai(messages), "max_tokens": max_new_tokens, "temperature": temperature,
            "stream": on_text is not None}
    errors = []
    for url, key, model in providers:
        try:
            text = _call(url, key, model, body, on_text)
            LAST_MODEL["name"] = model
            print(f"[API] answered by {model}", flush=True)
            return text
        except _Retry as e:
            errors.append(str(e))
            print(f"[API] {e} -> trying next", flush=True)
    raise RuntimeError("All free vision-language APIs are busy or unavailable right now. Try again in a minute.\n"
                       + "\n".join(errors))


def generate(bank, messages: list[dict], max_new_tokens: int = 900, temperature: float = 0.0, on_text=None) -> str:
    """on_text(partial_text) is called while tokens arrive (the app shows the answer as it is written)."""
    if use_api():
        return generate_api(messages, max_new_tokens, temperature, on_text)
    import threading

    import torch
    from transformers import TextIteratorStreamer

    proc, model = bank.vlm
    inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(model.device)
    print(f"[Qwen3-VL] prompt tokens: {inputs['input_ids'].shape[1]}", flush=True)
    kw = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "repetition_penalty": 1.05}
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.9)
    if on_text is None:
        with torch.no_grad():
            out = model.generate(**inputs, **kw)
        text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    else:
        streamer = TextIteratorStreamer(proc.tokenizer, skip_prompt=True, skip_special_tokens=True)
        err = []
        def run():
            try:
                with torch.no_grad():
                    model.generate(**inputs, streamer=streamer, **kw)
            except Exception as e:   # surface errors (e.g. GPU out of memory) instead of hanging the app
                err.append(e)
                streamer.end()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        text = ""
        for piece in streamer:
            text += piece
            on_text(text)
        th.join()
        if err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise err[0]
        text = text.strip()
        out = None
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
