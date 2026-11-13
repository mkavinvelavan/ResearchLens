"""ResearchLens - Streamlit app (Part 3).

    streamlit run app/streamlit_app.py

Needs a GPU for Qwen3-VL. HF_TOKEN (to load our private models) and optionally QDRANT_URL / QDRANT_API_KEY
come from environment variables or Streamlit secrets.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for k in ("HF_TOKEN", "QDRANT_URL", "QDRANT_API_KEY", "RL_HF_USER"):
    try:
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = st.secrets[k]
    except Exception:
        pass

st.set_page_config(page_title="ResearchLens", page_icon="🔎", layout="wide")
WORK = Path(os.environ.get("RL_WORK_DIR", Path(tempfile.gettempdir()) / "researchlens_papers"))
WORK.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner="Loading models (first time only)...")
def get_bank():
    from researchlens.pipeline.models import ModelBank

    return ModelBank()


@st.cache_resource
def get_qdrant():
    from researchlens.pipeline.index import client

    return client(WORK / "qdrant")


ss = st.session_state
ss.setdefault("docs", {})          # paper_id -> doc
ss.setdefault("overviews", {})     # paper_id -> overview result
ss.setdefault("chat", [])
ss.setdefault("comparison", None)
ss.setdefault("gaps", None)
ss.setdefault("ideas", None)


# ------------------------------------------------------------------ helpers
def show_trace(trace: list[dict], title: str = "Evidence Trace"):
    if not trace:
        st.caption("No evidence cited.")
        return
    with st.expander(f"🔍 {title} ({len(trace)} sources)", expanded=False):
        for t in trace:
            c1, c2 = st.columns([3, 1]) if t.get("image") else (st.container(), None)
            with c1:
                where = f"**[{t['eid']}]** {t.get('paper', '')[:60]} · page {t['page']} · {t.get('section') or '-'}"
                if t.get("label"):
                    where += f" · **{t['label']}**"
                st.markdown(where)
                st.caption(f"{t['kind']} — retrieved by {t['why']}")
                st.markdown(f"> {t['snippet']}")
            if c2 is not None and t.get("image") and Path(t["image"]).exists():
                with c2:
                    st.image(t["image"], use_container_width=True)


def paper_picker(label="Paper", key=None):
    if not ss.docs:
        st.info("Upload a paper first (Upload page).")
        st.stop()
    opts = {d.get("title", pid)[:90]: pid for pid, d in ss.docs.items()}
    return opts[st.selectbox(label, list(opts), key=key)]


# ------------------------------------------------------------------ pages
def page_upload():
    st.header("Upload papers")
    st.write("Upload one paper for **Single Paper Analysis** or two for **Two-Paper Comparison**. Each paper is split "
             "into text, figures, tables and formulas, processed by our fine-tuned models, and stored in Qdrant.")
    files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Process", type="primary"):
        from researchlens.pipeline.run import process_paper

        bank, qc = get_bank(), get_qdrant()
        for f in files[:2]:
            path = WORK / f.name
            path.write_bytes(f.getbuffer())
            with st.status(f"Processing {f.name}", expanded=True) as s:
                doc = process_paper(path, WORK, bank, qc, log=s.write)
                ss.docs[doc["paper_id"]] = doc
                s.update(label=f"✅ {doc.get('title', f.name)[:80]}", state="complete")
    for pid, d in ss.docs.items():
        kinds = {}
        for b in d["blocks"]:
            kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
        st.markdown(f"**{d.get('title')}** — {d['n_pages']} pages · " +
                    " · ".join(f"{v} {k}s" for k, v in kinds.items()) +
                    f" · {len(d.get('entities', []))} entities · {len(d.get('relations', []))} relations")


def page_overview():
    from researchlens.pipeline.analysis import overview

    st.header("Paper overview")
    pid = paper_picker()
    doc = ss.docs[pid]
    if pid not in ss.overviews or st.button("Regenerate"):
        box = st.empty()
        with st.spinner("Retrieving evidence and writing the overview..."):
            ss.overviews[pid] = overview(get_bank(), doc, qc=get_qdrant(), on_text=lambda t: box.markdown(t + " ▌"))
        box.empty()
    res = ss.overviews[pid]
    st.markdown(res["overview"])
    show_trace(res["trace"])


def page_ask():
    from researchlens.pipeline.answer import ask

    st.header("Ask the paper(s)")
    ids = st.multiselect("Search in", list(ss.docs), default=list(ss.docs),
                         format_func=lambda p: ss.docs[p].get("title", p)[:80])
    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                show_trace(m["trace"])
    q = st.chat_input("e.g. Which datasets are used? What does Figure 2 show? Explain Equation 1.")
    if q and ids:
        ss.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            box = st.empty()
            box.markdown("_Retrieving evidence..._")
            r = ask(get_bank(), q, ids, qc=get_qdrant(), on_text=lambda t: box.markdown(t + " ▌"))
            box.markdown(r["answer"])
            show_trace(r["trace"])
        ss.chat.append({"role": "assistant", "content": r["answer"], "trace": r["trace"]})


def page_explain():
    from researchlens.pipeline.analysis import explain_item

    st.header("Explain a figure, table or formula")
    pid = paper_picker()
    doc = ss.docs[pid]
    items = [b for b in doc["blocks"] if b["kind"] in ("figure", "table", "equation") and b.get("label")]
    if not items:
        st.info("No figures, tables or equations were found in this paper.")
        return
    lab = st.selectbox("Item", [b["label"] for b in items])
    b = next(x for x in items if x["label"] == lab)
    c1, c2 = st.columns([1, 2])
    with c1:
        if b.get("image"):
            st.image(str(Path(doc["out_dir"]) / b["image"]), use_container_width=True)
        if b["kind"] == "equation":
            if b.get("latex_ok", True) and b.get("latex"):
                lx = b["latex"]
                try:
                    st.latex(r"\begin{gathered}" + lx + r"\end{gathered}" if b.get("latex_lines", 1) > 1 else lx)
                except Exception:
                    st.code(lx)
                st.caption(f"Read by our TrOCR model ({b.get('latex_lines', 1)} line(s)).")
            else:
                st.caption("Our TrOCR model could not read this equation reliably, so its output is hidden; "
                           "Qwen3-VL reads the equation directly from the image instead.")
        else:
            st.caption(b.get("caption", ""))
            if b.get("figure_type"):
                src = b.get("figure_type_sources", {})
                st.caption(f"Figure type: **{b['figure_type']}** ({b['figure_type_conf']:.0%}) — image model: "
                           f"{src.get('image')}, caption model: {src.get('caption')}")
    with c2:
        if st.button("Explain", type="primary"):
            box = st.empty()
            box.markdown("_Retrieving evidence..._")
            r = explain_item(get_bank(), doc, lab, qc=get_qdrant(), on_text=lambda t: box.markdown(t + " ▌"))
            box.markdown(r["answer"])
            show_trace(r["trace"])


def page_compare():
    from researchlens.pipeline.analysis import compare

    st.header("Two-paper comparison")
    if len(ss.docs) < 2:
        st.info("Upload two papers to compare them.")
        return
    a = paper_picker("Paper A", "cmp_a")
    b = paper_picker("Paper B", "cmp_b")
    if a == b:
        st.warning("Pick two different papers.")
        return
    if st.button("Compare", type="primary"):
        box = st.empty()
        with st.spinner("Comparing (this takes a minute)..."):
            ss.comparison = compare(get_bank(), ss.docs[a], ss.docs[b], qc=get_qdrant(),
                                    on_text=lambda t: box.markdown(t + " ▌"))
        box.empty()
    r = ss.comparison
    if r:
        st.markdown(r["comparison"])
        with st.expander("Entity overlap found by our NER model"):
            for ty, v in r["entity_overlap"].items():
                st.markdown(f"**{ty}** — shared: {', '.join(v['shared']) or '–'}  \nonly A: "
                            f"{', '.join(v['only_a'][:10]) or '–'}  \nonly B: {', '.join(v['only_b'][:10]) or '–'}")
        show_trace(r["trace"])


def page_gaps():
    from researchlens.pipeline.analysis import gap_map, ideas

    st.header("Research Gap Map & potential directions")
    if len(ss.docs) < 2:
        st.info("Upload two papers first.")
        return
    a = paper_picker("Paper A", "gap_a")
    b = paper_picker("Paper B", "gap_b")
    if st.button("Find gaps", type="primary") and a != b:
        with st.spinner("Reading limitations, datasets, evaluation and future work..."):
            ss.gaps = gap_map(get_bank(), ss.docs[a], ss.docs[b], qc=get_qdrant())
            ss.ideas = None
    g = ss.gaps
    if not g:
        return
    if not g["gaps"]:
        st.warning("The model did not return gaps in the expected format.")
        st.text(g["raw"])
        return
    ev = {e["eid"]: e for e in g["evidence"]}
    for gap in g["gaps"]:
        badge = "✅ supported" if gap["supported"] else "⚠️ no valid evidence"
        with st.container(border=True):
            st.markdown(f"### {gap.get('title')}\n`{gap.get('type')}` · papers: {', '.join(gap.get('papers', []))} · {badge}")
            st.write(gap.get("description", ""))
            for eid in gap["evidence"]:
                e = ev[eid]
                st.caption(f"[{eid}] {e.get('paper_title', '')[:50]} · p.{e['page']} · {e.get('section')}: "
                           f"{(e.get('caption') or e.get('text') or '')[:200]}")
    st.divider()
    st.subheader("Potential research directions")
    st.caption("Suggestions to explore, derived from the gaps above — not guaranteed or proven solutions.")
    if st.button("Generate ideas"):
        with st.spinner("Thinking about directions..."):
            ss.ideas = ideas(get_bank(), [x for x in g["gaps"] if x["supported"]], ss.docs[a], ss.docs[b])
    if ss.ideas:
        for idea in ss.ideas["ideas"]:
            with st.container(border=True):
                st.markdown(f"**{idea.get('idea')}**  \n*Addresses gap:* {idea.get('gap')}")
                for k in ("why_relevant", "methodology", "datasets", "evaluation", "risk"):
                    if idea.get(k):
                        st.markdown(f"- **{k.replace('_', ' ').capitalize()}:** {idea[k]}")


def page_entities():
    st.header("Entities & relations (our SciBERT models)")
    pid = paper_picker()
    doc = ss.docs[pid]
    import pandas as pd

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Entities")
        st.dataframe(pd.DataFrame(doc.get("entities", [])), use_container_width=True, height=420)
    with c2:
        st.subheader("Relations")
        rel = pd.DataFrame(doc.get("relations", []))
        if not rel.empty:
            rel = rel[["head", "relation_text", "tail", "conf", "page"]]
        st.dataframe(rel, use_container_width=True, height=420)
    figs = [b for b in doc["blocks"] if b["kind"] in ("figure", "table")]
    if figs:
        st.subheader("Figures & tables")
        cols = st.columns(3)
        for i, b in enumerate(figs):
            with cols[i % 3]:
                if b.get("image"):
                    st.image(str(Path(doc["out_dir"]) / b["image"]), use_container_width=True)
                st.caption(f"**{b.get('label')}** · {b.get('figure_type', '?')} · p.{b['page']} · mentioned in "
                           f"{len(b.get('mentioned_in', []))} paragraph(s)")


PAGES = {"📄 Upload": page_upload, "🧭 Overview": page_overview, "💬 Ask": page_ask, "🖼️ Explain figure / formula":
         page_explain, "⚖️ Compare": page_compare, "🗺️ Gap Map & Ideas": page_gaps, "🔗 Entities": page_entities}
with st.sidebar:
    st.title("🔎 ResearchLens")
    st.caption("Evidence-grounded multimodal research assistant")
    choice = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    for pid, d in ss.docs.items():
        st.caption(f"• {d.get('title', pid)[:60]}")
PAGES[choice]()
