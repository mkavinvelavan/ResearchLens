"""System-level evaluation of ResearchLens (the whole RAG pipeline, not single models).

Part A - QASPER (text questions on NLP papers, test split; the evidence finder never saw it)
  Retrieval  : BM25 word overlap  vs  bge search  vs  bge + our SciBERT evidence finder (= the app)
               Recall@1/3/5 and MRR of the expert-marked evidence passages.
  Answers    : Gemini with NO retrieval (title + abstract)  vs  RAG with bge only  vs  full system (bge + evidence finder)
               QASPER answer F1 (official token F1, best over annotators, also per answer type),
               evidence F1 of the passages the answer cites, citation precision, % answers with a citation.
Part B - SPIQA test-A (questions about figures and tables of CS papers)
  Retrieval  : caption search  vs  caption search + re-rank  vs  image search (SigLIP)  vs  full system
               hit@1 / hit@3 of the reference figure.
  Answers    : Gemini with captions only  vs  with the figure images (= the app), judged correct / incorrect by Gemini
               against the gold answer (LLM-as-judge, as in SPIQA's own L3Score) + token F1.
All metrics with bootstrap 95% confidence intervals. Per-question records are written as they are produced
(jsonl), so a stopped run continues where it left off.

    python -m researchlens.evaluation.system_eval --out_dir results/system_eval --n_qasper 100 --n_spiqa 50
Needs GEMINI_API_KEY and (for SPIQA download) internet. A GPU makes retrieval fast; it also runs on CPU.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import string
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CITE_RE = re.compile(r"\[(E\d+)\]")


# =============================================================== scoring helpers
def normalize_answer(s: str) -> str:
    """QASPER / SQuAD normalisation: lower case, no punctuation, no articles, single spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def set_f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def bootstrap_ci(values, n: int = 1000, seed: int = 0):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[E\d+\]", "", text).strip()


# =============================================================== Gemini with pacing + retry
class Pacer:
    """Free Gemini tier: keep below ~12 requests per minute and retry once after a pause."""

    def __init__(self, min_gap: float = 5.0):
        self.min_gap, self.last = min_gap, 0.0

    def call(self, fn, *a, **kw):
        for attempt in range(3):
            wait = self.min_gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                return fn(*a, **kw)
            except Exception as e:      # all Gemini models busy / rate limited
                print(f"  Gemini failed ({str(e).splitlines()[0][:120]}); waiting {30 * (attempt + 1)}s", flush=True)
                time.sleep(30 * (attempt + 1))
        return None


# =============================================================== shared: write a paper folder + index it
def write_doc(out_dir: Path, paper_id: str, title: str, blocks: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(blocks):
        b.setdefault("id", f"b{i:04d}")
        b.setdefault("page", 0)
        b.setdefault("section", "")
    (out_dir / "doc.json").write_text(json.dumps({"paper_id": paper_id, "title": title, "enriched": True,
                                                  "blocks": blocks}, ensure_ascii=False), encoding="utf-8")
    return out_dir


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================== Part A: QASPER
def qasper_answers(qa: dict) -> list[dict]:
    """[{type, text}] one per annotator, as in the official evaluator."""
    out = []
    for a in qa.get("answers") or []:
        ans = a.get("answer", a)
        if ans.get("unanswerable"):
            out.append({"type": "none", "text": "Unanswerable"})
        elif ans.get("extractive_spans"):
            out.append({"type": "extractive", "text": ", ".join(ans["extractive_spans"])})
        elif ans.get("yes_no") is not None:
            out.append({"type": "boolean", "text": "Yes" if ans["yes_no"] else "No"})
        elif ans.get("free_form_answer"):
            out.append({"type": "abstractive", "text": ans["free_form_answer"]})
    return out


def answer_f1(pred: str, golds: list[dict]) -> tuple[float, str]:
    """Best token F1 over annotators; yes/no questions are judged on the leading Yes / No."""
    best, typ = 0.0, golds[0]["type"] if golds else "none"
    for g in golds:
        p = pred
        if g["type"] == "boolean":
            m = re.match(r"\W*(yes|no)\b", pred.lower())
            p = m.group(1) if m else pred
        s = token_f1(p, g["text"])
        if s >= best:
            best, typ = s, g["type"]
    return best, typ


def load_qasper_test(data_dir: Path, n_questions: int, seed: int) -> list[dict]:
    from researchlens.nlp.train_evidence import FILES, download_qasper, passages, questions

    folder = download_qasper(data_dir)
    raw = json.loads((folder / FILES["test"]).read_text(encoding="utf-8"))
    papers = []
    for pid, paper in raw.items():
        cands = passages(paper)
        qs = questions(paper, cands)
        byq = {qa.get("question_id"): qa for qa in paper.get("qas") or []}
        keep = []
        for q in qs:
            golds = [g for g in qasper_answers(byq.get(q["qid"], {})) if g["type"] != "none"]
            if golds:                      # answerable questions with evidence
                keep.append({**q, "golds": golds})
        if keep:
            papers.append({"id": pid, "title": paper.get("title", ""), "abstract": paper.get("abstract", ""),
                           "passages": cands, "questions": keep})
    rng = random.Random(seed)
    rng.shuffle(papers)
    chosen, n = [], 0
    for p in papers:                       # at most 2 questions per paper -> many different papers
        qs = p["questions"][:]
        rng.shuffle(qs)
        p["questions"] = qs[:2]
        chosen.append(p)
        n += len(p["questions"])
        if n >= n_questions:
            break
    print(f"QASPER test: {len(chosen)} papers, {n} answerable questions with evidence", flush=True)
    return chosen


def run_qasper(bank, qc, out: Path, args, pacer: Pacer) -> None:
    from researchlens.nlp.train_evidence import overlap_scores
    from researchlens.pipeline import answer as A
    from researchlens.pipeline.answer import SYSTEM, build_messages, generate
    from researchlens.pipeline.index import index_paper, paper_points
    from researchlens.pipeline.retrieve import retrieve

    rec_path = out / "qasper_records.jsonl"
    done = {r["qid"] for r in read_jsonl(rec_path)}
    papers = load_qasper_test(Path(args.data_dir) / "qasper", args.n_qasper, args.seed)
    instr_tpl = ("Question: {q}\nAnswer in as few words as possible: a short phrase or list; for a yes/no question "
                 "start with Yes or No. Cite the evidence ids you used, e.g. [E2]. If the evidence does not contain "
                 "the answer, say 'Unanswerable'.")
    for p in papers:
        pid = f"qasper_{p['id']}"
        blocks = []
        for c in p["passages"]:
            if c["kind"] == "paragraph":
                blocks.append({"kind": "paragraph", "text": c["text"], "section": c["section"]})
            else:
                m = re.match(r"\s*(fig(?:ure)?|table)\s*(\d+)", c["text"], re.I)
                lab = (("Figure" if m.group(1).lower().startswith("fig") else "Table") + " " + m.group(2)) if m else None
                blocks.append({"kind": c["kind"], "text": "", "caption": c["text"], "label": lab})
        folder = write_doc(out / "papers" / pid, pid, p["title"], blocks)
        if paper_points(pid, qc) == 0:
            index_paper(folder, bank, qc)
        # block id -> passage index (paragraphs <= 40 chars are not indexed, so they can never be retrieved)
        bid2idx = {f"b{i:04d}": i for i in range(len(blocks))}
        for q in p["questions"]:
            if q["qid"] in done:
                continue
            gold = set(q["gold"])
            rec = {"qid": q["qid"], "paper": p["id"], "question": q["question"], "gold_evidence": sorted(gold),
                   "gold_answers": q["golds"]}
            # ---------- retrieval rankings
            bm = overlap_scores(q["question"], p["passages"])
            rank_bm25 = [int(i) for i in np.argsort(-bm)]
            r_bge = retrieve(bank, q["question"], [pid], qc=qc, k_candidates=24, k_text=24, k_image=0, rerank=False,
                             kinds=["paragraph", "figure", "table"])
            r_ours = retrieve(bank, q["question"], [pid], qc=qc, k_candidates=24, k_text=24, k_image=0, rerank=True,
                              kinds=["paragraph", "figure", "table"])
            rank = {"bm25": rank_bm25,
                    "bge": [bid2idx[x["block_id"]] for x in r_bge if x["block_id"] in bid2idx],
                    "bge_rerank": [bid2idx[x["block_id"]] for x in r_ours if x["block_id"] in bid2idx]}
            rec["retrieval"] = {}
            for name, rk in rank.items():
                first = next((i for i, x in enumerate(rk) if x in gold), None)
                rec["retrieval"][name] = {f"recall@{k}": float(any(x in gold for x in rk[:k])) for k in (1, 3, 5)}
                rec["retrieval"][name]["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
            # ---------- answers
            rec["answers"] = {}
            settings = {"bge": r_bge[:5], "full": r_ours[:5]}
            for name, ev in settings.items():
                ev = [dict(x, eid=f"E{i}") for i, x in enumerate(ev, 1)]
                text = pacer.call(generate, bank, build_messages(ev, instr_tpl.format(q=q["question"]), max_images=0),
                                  max_new_tokens=200)
                if text is None:
                    rec["answers"][name] = None
                    continue
                cited = [e for e in dict.fromkeys(CITE_RE.findall(text))]
                by = {e["eid"]: e for e in ev}
                cited_idx = {bid2idx[by[c]["block_id"]] for c in cited if c in by and by[c]["block_id"] in bid2idx}
                f1, typ = answer_f1(strip_citations(text), q["golds"])
                ev_f1 = max(set_f1(cited_idx, set(a)) for a in q["annotators"])
                rec["answers"][name] = {"answer": text, "answer_f1": f1, "type": typ, "evidence_f1": ev_f1,
                                        "cited": len(cited_idx), "citation_precision":
                                        (len(cited_idx & gold) / len(cited_idx)) if cited_idx else None}
            # closed book: title + abstract only, no retrieval
            msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "text", "text":
                     f"EVIDENCE\n[E1] Title: {p['title']}\nAbstract: {p['abstract'][:2500]}\n\nTASK\n"
                     + instr_tpl.format(q=q["question"])}]}]
            text = pacer.call(generate, bank, msgs, max_new_tokens=200)
            if text is not None:
                f1, typ = answer_f1(strip_citations(text), q["golds"])
                rec["answers"]["no_retrieval"] = {"answer": text, "answer_f1": f1, "type": typ}
            rec["answer_model"] = A.LAST_MODEL["name"]
            append_jsonl(rec_path, rec)
            full = rec["answers"].get("full")
            print(f"[QASPER {len(read_jsonl(rec_path))}] answer F1 (full system): "
                  f"{full['answer_f1']:.2f}" if full else f"[QASPER] {q['qid']}: no answer", flush=True)


def summarize_qasper(out: Path) -> dict:
    recs = read_jsonl(out / "qasper_records.jsonl")
    res = {"n_questions": len(recs), "n_papers": len({r["paper"] for r in recs}), "retrieval": {}, "answers": {}}
    for name in ("bm25", "bge", "bge_rerank"):
        res["retrieval"][name] = {m: bootstrap_ci([r["retrieval"][name][m] for r in recs])
                                  for m in ("recall@1", "recall@3", "recall@5", "mrr")}
    for name in ("no_retrieval", "bge", "full"):
        rows = [r["answers"].get(name) for r in recs if r["answers"].get(name)]
        d = {"n": len(rows), "answer_f1": bootstrap_ci([x["answer_f1"] for x in rows])}
        by_type = defaultdict(list)
        for x in rows:
            by_type[x["type"]].append(x["answer_f1"])
        d["answer_f1_by_type"] = {t: {"n": len(v), "mean": float(np.mean(v))} for t, v in by_type.items()}
        if name != "no_retrieval":
            d["evidence_f1"] = bootstrap_ci([x["evidence_f1"] for x in rows])
            d["citation_precision"] = bootstrap_ci([x["citation_precision"] for x in rows])
            d["answers_with_citation"] = float(np.mean([x["cited"] > 0 for x in rows])) if rows else None
        res["answers"][name] = d
    return res


# =============================================================== Part B: SPIQA (figures)
def load_spiqa_test(cache: Path, out: Path, n_questions: int, seed: int) -> list[dict]:
    from PIL import Image

    from researchlens.vision.prepare_spiqa_figures import FILES, download, iter_figures

    jfile, zfile = FILES["test"]
    data = json.loads(download(jfile, cache).read_text(encoding="utf-8"))
    papers = []
    for pid, paper in data.items():
        if not isinstance(paper, dict):
            continue
        qas = paper.get("qa") or paper.get("question_answers") or paper.get("qas") or []
        figs = list(iter_figures({pid: paper}))
        good = [qa for qa in qas if isinstance(qa, dict) and qa.get("question") and qa.get("answer") and
                (qa.get("reference") or qa.get("figure"))]
        if good and len(figs) >= 2:
            papers.append((pid, paper, figs, good))
    if not papers:
        first = next(iter(data.values()))
        raise RuntimeError(f"No SPIQA questions found; keys of a paper: {list(first)[:20]}")
    rng = random.Random(seed)
    rng.shuffle(papers)
    chosen, n = [], 0
    zpath = download(zfile, cache)
    with zipfile.ZipFile(zpath) as zf:
        members = {Path(m).name: m for m in zf.namelist() if m.lower().endswith((".png", ".jpg", ".jpeg"))}
        for pid, paper, figs, qas in papers:
            folder = out / "papers" / f"spiqa_{pid}"
            blocks, names = [], []
            for _, name, meta in figs:
                base = Path(name).name
                if base not in members:
                    continue
                dst = folder / "img" / (Path(base).stem + ".png")
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        img = Image.open(io.BytesIO(zf.read(members[base]))).convert("RGB")
                    except Exception:
                        continue
                    img.thumbnail((768, 768))
                    img.save(dst)
                content = str(meta.get("content_type", meta.get("type", ""))).lower()
                kind = "table" if "table" in content else "figure"
                cap = (meta.get("caption") or "").strip()
                m = re.match(r"\s*(fig(?:ure)?\.?|table)\s*(\d+)", cap, re.I)
                lab = (("Figure" if m.group(1).lower().startswith("fig") else "Table") + " " + m.group(2)) if m else None
                blocks.append({"kind": kind, "text": "", "caption": cap, "label": lab,
                               "image": str(dst.relative_to(folder)), "file": base})
                names.append(base)
            if len(blocks) < 2:
                continue
            qa = rng.choice(qas)
            ref = Path(qa.get("reference") or qa.get("figure")).name
            if ref not in names:
                continue
            chosen.append({"id": pid, "title": paper.get("title") or paper.get("paper_title") or pid,
                           "folder": folder, "blocks": blocks, "question": qa["question"], "answer": qa["answer"],
                           "reference": ref})
            n += 1
            if n >= n_questions:
                break
    print(f"SPIQA test-A: {len(chosen)} questions (one per paper)", flush=True)
    return chosen


def run_spiqa(bank, qc, out: Path, args, pacer: Pacer) -> None:
    from researchlens.pipeline import answer as A
    from researchlens.pipeline.answer import build_messages, generate
    from researchlens.pipeline.index import embed_siglip_text, embed_texts, index_paper, paper_points
    from researchlens.pipeline.retrieve import _filter, _search, retrieve

    rec_path = out / "spiqa_records.jsonl"
    done = {r["paper"] for r in read_jsonl(rec_path)}
    items = load_spiqa_test(Path(args.data_dir) / "spiqa_raw", out, args.n_spiqa, args.seed)
    judge_tpl = ("Question: {q}\nReference answer: {g}\nCandidate answer: {c}\n\nIs the candidate answer correct, i.e. "
                 "does it agree with the reference answer on the facts that matter? Reply with exactly one word: "
                 "YES or NO.")
    for it in items:
        if it["id"] in done:
            continue
        pid = f"spiqa_{it['id']}"
        write_doc(it["folder"], pid, it["title"], it["blocks"])
        if paper_points(pid, qc) == 0:
            index_paper(it["folder"], bank, qc)
        bid2file = {f"b{i:04d}": b["file"] for i, b in enumerate(it["blocks"])}
        ref, q = it["reference"], it["question"]
        rec = {"paper": it["id"], "question": q, "gold_answer": it["answer"], "reference": ref, "retrieval": {}}
        kinds = ["figure", "table"]
        qv = embed_texts(bank, [q], query=True)[0]
        cap_rank = [bid2file[p["block_id"]] for _, p in _search(qc, qv, "text", _filter([pid], kinds), 24)]
        rr = retrieve(bank, q, [pid], qc=qc, k_candidates=24, k_text=24, k_image=0, rerank=True, kinds=kinds)
        rerank_rank = [bid2file[x["block_id"]] for x in rr]
        sv = embed_siglip_text(bank, [q])[0]
        img_rank = [bid2file[p["block_id"]] for _, p in _search(qc, sv, "image", _filter([pid], kinds), 24)]
        full = retrieve(bank, q, [pid], qc=qc, k_text=2, k_image=2, rerank=True, kinds=kinds)   # as in the app
        full_set = [bid2file[x["block_id"]] for x in full]
        for name, rk in (("caption_search", cap_rank), ("caption_rerank", rerank_rank), ("image_search", img_rank)):
            rec["retrieval"][name] = {"hit@1": float(ref in rk[:1]), "hit@3": float(ref in rk[:3])}
        rec["retrieval"]["full_system"] = {"hit@1": float(ref in full_set[:1]), "in_evidence": float(ref in full_set),
                                           "n_evidence": len(full_set)}
        rec["answers"] = {}
        instr = f"Question: {q}\nAnswer in one or two sentences, citing evidence ids."
        for name, n_img in (("captions_only", 0), ("with_images", 2)):
            text = pacer.call(generate, bank, build_messages(full, instr, max_images=n_img), max_new_tokens=250)
            if text is None:
                continue
            ans = strip_citations(text)
            jmsg = [{"role": "user", "content": [{"type": "text", "text": judge_tpl.format(q=q, g=it["answer"], c=ans)}]}]
            verdict = pacer.call(generate, bank, jmsg, max_new_tokens=50)
            rec["answers"][name] = {"answer": ans, "token_f1": token_f1(ans, it["answer"]),
                                    "judge_correct": (1.0 if verdict and verdict.strip().upper().startswith("YES") else 0.0)
                                    if verdict is not None else None}
        rec["answer_model"] = A.LAST_MODEL["name"]
        append_jsonl(rec_path, rec)
        print(f"[SPIQA {len(read_jsonl(rec_path))}] ref in evidence={rec['retrieval']['full_system']['in_evidence']:.0f}",
              flush=True)


def summarize_spiqa(out: Path) -> dict:
    recs = read_jsonl(out / "spiqa_records.jsonl")
    res = {"n_questions": len(recs), "retrieval": {}, "answers": {}}
    for name in ("caption_search", "caption_rerank", "image_search"):
        res["retrieval"][name] = {m: bootstrap_ci([r["retrieval"][name][m] for r in recs]) for m in ("hit@1", "hit@3")}
    res["retrieval"]["full_system"] = {
        "hit@1": bootstrap_ci([r["retrieval"]["full_system"]["hit@1"] for r in recs]),
        "in_evidence": bootstrap_ci([r["retrieval"]["full_system"]["in_evidence"] for r in recs]),
        "mean_evidence_items": float(np.mean([r["retrieval"]["full_system"]["n_evidence"] for r in recs])) if recs else None}
    for name in ("captions_only", "with_images"):
        rows = [r["answers"][name] for r in recs if r["answers"].get(name)]
        res["answers"][name] = {"n": len(rows), "judge_accuracy": bootstrap_ci([x["judge_correct"] for x in rows]),
                                "token_f1": bootstrap_ci([x["token_f1"] for x in rows])}
    return res


# =============================================================== report
def pct(t) -> str:
    if not t or t[0] is None:
        return "–"
    m, lo, hi = t
    return f"{100 * m:.1f} [{100 * lo:.1f}–{100 * hi:.1f}]"


def write_report(out: Path, qa: dict, sp: dict, model_note: str) -> str:
    L = ["# ResearchLens system evaluation", "",
         f"Numbers are % with bootstrap 95% confidence intervals in brackets. Answer model: {model_note}.", ""]
    if qa.get("n_questions"):
        L += [f"## A. QASPER test (text): {qa['n_questions']} answerable questions from {qa['n_papers']} papers", "",
              "### Retrieval of the expert-marked evidence", "", "| Method | Recall@1 | Recall@3 | Recall@5 | MRR |",
              "|---|---|---|---|---|"]
        names = {"bm25": "BM25 word overlap (baseline)", "bge": "bge-small vector search",
                 "bge_rerank": "bge + SciBERT evidence finder (ours, used in app)"}
        for k, v in qa["retrieval"].items():
            L.append(f"| {names[k]} | {pct(v['recall@1'])} | {pct(v['recall@3'])} | {pct(v['recall@5'])} | {pct(v['mrr'])} |")
        L += ["", "### Answers (Gemini)", "", "| Setting | Answer F1 | Evidence F1 (cited) | Citation precision | Answers with a citation |",
              "|---|---|---|---|---|"]
        names = {"no_retrieval": "No retrieval (title + abstract only)", "bge": "RAG, bge search only",
                 "full": "RAG, bge + evidence finder (full system)"}
        for k, v in qa["answers"].items():
            wc = v.get("answers_with_citation")
            wc = "–" if wc is None else f"{100 * wc:.0f}"
            L.append(f"| {names[k]} (n={v['n']}) | {pct(v['answer_f1'])} | {pct(v.get('evidence_f1'))} | "
                     f"{pct(v.get('citation_precision'))} | {wc} |")
        L += ["", "Answer F1 by answer type (full system): " + ", ".join(
            f"{t} {100 * d['mean']:.1f} (n={d['n']})" for t, d in qa["answers"]["full"]["answer_f1_by_type"].items()), ""]
    if sp.get("n_questions"):
        L += [f"## B. SPIQA test-A (figures and tables): {sp['n_questions']} questions", "",
              "### Finding the reference figure / table", "", "| Method | Hit@1 | Hit@3 |", "|---|---|---|"]
        names = {"caption_search": "Caption text search (bge)", "caption_rerank": "Caption search + evidence finder",
                 "image_search": "Image search (SigLIP)"}
        for k in ("caption_search", "caption_rerank", "image_search"):
            v = sp["retrieval"][k]
            L.append(f"| {names[k]} | {pct(v['hit@1'])} | {pct(v['hit@3'])} |")
        fs = sp["retrieval"]["full_system"]
        L.append(f"| Full system (2 text + 2 image, as in app; ~{fs['mean_evidence_items']:.1f} items) | {pct(fs['hit@1'])} | "
                 f"in evidence: {pct(fs['in_evidence'])} |")
        L += ["", "### Answers (Gemini, judged against the gold answer)", "", "| Setting | Judged correct | Token F1 |",
              "|---|---|---|"]
        for k, lab in (("captions_only", "Captions only (no images)"), ("with_images", "With figure images (as in app)")):
            v = sp["answers"][k]
            L.append(f"| {lab} (n={v['n']}) | {pct(v['judge_accuracy'])} | {pct(v['token_f1'])} |")
    txt = "\n".join(L) + "\n"
    (out / "report.md").write_text(txt, encoding="utf-8")
    return txt


def plot(out: Path, qa: dict, sp: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    if qa.get("n_questions"):
        names = [("bm25", "BM25"), ("bge", "bge"), ("bge_rerank", "bge + evidence\nfinder (ours)")]
        for i, k in enumerate(("recall@1", "recall@5", "mrr")):
            vals = [qa["retrieval"][n][k] for n, _ in names]
            axes[0].bar(np.arange(3) + i * 0.27, [100 * v[0] for v in vals], 0.25, label=k,
                        yerr=[[100 * (v[0] - v[1]) for v in vals], [100 * (v[2] - v[0]) for v in vals]], capsize=3)
        axes[0].set_xticks(np.arange(3) + 0.27, [l for _, l in names]); axes[0].set_title("QASPER evidence retrieval (%)")
        axes[0].legend(frameon=False)
        names = [("no_retrieval", "no retrieval"), ("bge", "RAG bge"), ("full", "RAG full")]
        vals = [qa["answers"][n]["answer_f1"] for n, _ in names]
        axes[1].bar(range(3), [100 * v[0] for v in vals], yerr=[[100 * (v[0] - v[1]) for v in vals],
                                                                [100 * (v[2] - v[0]) for v in vals]], capsize=4)
        axes[1].set_xticks(range(3), [l for _, l in names]); axes[1].set_title("QASPER answer F1 (%)")
    if sp.get("n_questions"):
        names = [("caption_search", "caption"), ("caption_rerank", "caption +\nre-rank"), ("image_search", "image\n(SigLIP)")]
        vals = [sp["retrieval"][n]["hit@1"] for n, _ in names]
        axes[2].bar(range(3), [100 * v[0] for v in vals], yerr=[[100 * (v[0] - v[1]) for v in vals],
                                                                [100 * (v[2] - v[0]) for v in vals]], capsize=4)
        axes[2].set_xticks(range(3), [l for _, l in names]); axes[2].set_title("SPIQA: reference figure ranked first (%)")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "system_eval.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/system_eval")
    ap.add_argument("--data_dir", default="/kaggle/working/eval_data")
    ap.add_argument("--n_qasper", type=int, default=100)
    ap.add_argument("--n_spiqa", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_gap", type=float, default=5.0, help="seconds between Gemini calls (free tier)")
    ap.add_argument("--parts", default="qasper,spiqa")
    args = ap.parse_args()

    from researchlens.pipeline import answer as A
    from researchlens.pipeline.index import client
    from researchlens.pipeline.models import ModelBank

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank, qc = ModelBank(), client(out / "qdrant")
    pacer = Pacer(args.min_gap)
    t0 = time.time()
    if "qasper" in args.parts:
        run_qasper(bank, qc, out, args, pacer)
    if "spiqa" in args.parts:
        run_spiqa(bank, qc, out, args, pacer)
    qa, sp = summarize_qasper(out), summarize_spiqa(out)
    note = "Gemini (Google AI Studio free tier; " + ", ".join(A.gemini_models()) + " in fallback order)"
    metrics = {"qasper": qa, "spiqa": sp, "answer_model": note, "seed": args.seed,
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(write_report(out, qa, sp, note))
    plot(out, qa, sp)


if __name__ == "__main__":
    main()
