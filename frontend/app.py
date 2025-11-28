# app.py — ULTRA-GOD Hybrid UI (Hybrid layout C)
# Obsidian liquid-glass + Apple×NVIDIA palette.
# Place in same folder as rag_backend.py

import streamlit as st
from pathlib import Path
import time
import io
import traceback

# Backend imports (must exist in rag_backend.py)
from rag_backend import (
    process_pdf,
    summarize_document_hierarchical,
    extract_takeaways,
    answer_question,
    save_summary_txt,
    save_summary_pdf,
)
# optional - safe import
try:
    from rag_backend import save_index_and_chunks
    _SAVE_INDEX_AVAILABLE = True
except Exception:
    _SAVE_INDEX_AVAILABLE = False

# ---------------- page config ----------------
st.set_page_config(
    page_title="LLM RAG — Ultra God Hybrid",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- palette ----------------
ACCENT_RAD = "#2CFFB2"   # radium mint
ACCENT_CYAN = "#4FD8FF"  # cyan
ACCENT_APPLE = "#6EA8FF" # apple blue
BG = "#050607"           # almost black
CARD_BG = "rgba(255,255,255,0.03)"
TEXT = "#E6F0F8"
MUTED = "#9fb0bf"

# ---------------- CSS (liquid-glass + neon glow, dense layout) ----------------
st.markdown(f"""
<style>
:root {{
  --bg: {BG};
  --card: rgba(10,12,15,0.6);
  --glass: rgba(255,255,255,0.03);
  --accent: {ACCENT_RAD};
  --cyan: {ACCENT_CYAN};
  --apple: {ACCENT_APPLE};
  --muted: {MUTED};
  --text: {TEXT};
  --radius: 12px;
}}

html, body, .reportview-container, .main {{
  background: linear-gradient(180deg, #030405 0%, #06080a 100%);
  color: var(--text);
}}

.block-container {{
  padding: 10px 16px 6px 16px;
}}

.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.03);
  backdrop-filter: blur(8px) saturate(1.1);
  -webkit-backdrop-filter: blur(8px) saturate(1.1);
  padding: 14px;
  border-radius: var(--radius);
  box-shadow: 0 10px 30px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
  margin-bottom: 12px;
}}

.header-title {{
  font-size: 20px;
  font-weight: 800;
  color: var(--apple);
  margin-bottom: 6px;
}}

.small-muted {{
  color: var(--muted);
  font-size: 13px;
}}

.kv {{
  font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,"Roboto Mono",monospace;
  color: #cfeffb;
  font-size: 13px;
}}

.neon-badge {{
  display:inline-block;
  padding:6px 10px;
  background: linear-gradient(90deg, var(--accent), var(--cyan));
  color: #021213;
  font-weight:700;
  border-radius:8px;
  box-shadow: 0 6px 18px rgba(44,255,178,0.15), 0 0 28px rgba(79,216,255,0.04);
}}

.chunk {{
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.03);
  padding: 8px;
  border-radius: 8px;
  font-family: ui-monospace, monospace;
  font-size:13px;
  color: #d8eef6;
  margin-bottom:8px;
}}

.btn-accent {{
  background: linear-gradient(90deg, var(--accent), var(--apple));
  color: #021213 !important;
  font-weight:800;
  padding:8px 12px;
  border-radius:8px;
  border: none;
}}

.section-title {{
  color: var(--apple);
  font-weight: 800;
  font-size:16px;
  margin-bottom:6px;
}}

.footer {{
  color: var(--muted);
  font-size:12px;
  text-align:center;
  padding-top:8px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar: MODEL INFO ONLY ----------------
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='header-title'>Model / Env</div>", unsafe_allow_html=True)
    # try to discover backend model names (if rag_backend exposes them)
    model_info_lines = []
    try:
        # attempt to read attributes if backend provides them
        info_vars = []
        for nm in ["_LLM_NAME", "EMBED_MODEL", "PREFERRED_LLM", "FALLBACK_LLM", "CHUNK_SIZE", "CHUNK_OVERLAP"]:
            if hasattr(__import__("rag_backend"), nm):
                val = getattr(__import__("rag_backend"), nm)
                model_info_lines.append(f"{nm}: {val}")
    except Exception:
        pass

    # display general fallback info
    st.markdown("<div class='small-muted'>LLM & Embed Info (read-only)</div>", unsafe_allow_html=True)
    if model_info_lines:
        for l in model_info_lines:
            st.markdown(f"<div class='kv'>{l}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='kv'>Embedding model: all-MiniLM-L6-v2</div>", unsafe_allow_html=True)
        st.markdown("<div class='kv'>Preferred LLM: flan-t5-large (fallback: flan-t5-base)</div>", unsafe_allow_html=True)
        st.markdown("<div class='kv'>Chunk size / overlap: backend defaults</div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Controls</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
    summary_target = st.slider("Summary words", min_value=100, max_value=180, value=130, step=10)
    strict_mode_default = st.checkbox("Strict QA fallback", value=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("Clear cached doc"):
        st.session_state.pop("doc", None)
        st.session_state.pop("summary", None)
        st.session_state.pop("tks", None)
        st.session_state.pop("qa", None)
        st.success("Cleared session cache.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Utilities & safe wrappers ----------------
def safe_save_summary_pdf(text):
    try:
        return save_summary_pdf(text)
    except Exception:
        return None

def safe_answer_question(doc, question, strict=True):
    """
    Try to call answer_question with strict kwarg; if backend doesn't accept it, fallback.
    Also ensure result dict has expected keys.
    """
    try:
        res = answer_question(doc, question, strict=strict)
    except TypeError:
        try:
            res = answer_question(doc, question)
        except Exception as e:
            res = {"type":"qa","answer":"Error while answering: "+str(e),"sources":[],"confidence":0.0}
    except Exception as e:
        res = {"type":"qa","answer":"Error while answering: "+str(e),"sources":[],"confidence":0.0}

    # normalize
    if not isinstance(res, dict):
        res = {"type":"qa","answer":str(res), "sources": [], "confidence": 0.0}
    if "answer" not in res:
        res["answer"] = str(res.get("text",""))
    if "confidence" not in res:
        res["confidence"] = float(res.get("conf", 0.0) or 0.0)
    return res

# ---------------- Load / process PDF ----------------
if uploaded_file is not None:
    # write to temp file
    tmp = Path("uploaded_for_rag.pdf")
    tmp.write_bytes(uploaded_file.read())
    try:
        doc = process_pdf(str(tmp))
        st.session_state["doc"] = doc
    except Exception as e:
        st.error("Error processing PDF: " + str(e))
        st.stop()

if "doc" not in st.session_state:
    st.info("Upload a technical PDF (LLM_TEST_FINAL.pdf recommended).", icon="📄")
    st.stop()

doc = st.session_state["doc"]

# ---------------- TOP: Full-width summary card ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div><div class='header-title'>Document Intelligence — Overview</div>"
            f"<div class='small-muted'>Path: <span class='kv'>{doc.path}</span></div></div>"
            f"<div><span class='neon-badge'>Hybrid • Ultra</span></div></div>", unsafe_allow_html=True)

# Generate summary button (center)
col_btns = st.columns([1,1,1,6])
with col_btns[0]:
    if st.button("Generate Summary"):
        with st.spinner("Summarizing (hierarchical, 100–150 words)..."):
            summ = summarize_document_hierarchical(doc, target_words=summary_target)
            st.session_state["summary"] = summ
            st.session_state["summary_txt_path"] = save_summary_txt(summ)
            st.session_state["summary_pdf_path"] = safe_save_summary_pdf(summ)
with col_btns[1]:
    if st.button("Extract Takeaways"):
        with st.spinner("Extracting takeaways..."):
            tks = extract_takeaways(doc, n=6)
            if not isinstance(tks, list):
                tks = [str(tks)]
            st.session_state["tks"] = tks
with col_btns[2]:
    if st.button("Save Index & Chunks"):
        if _SAVE_INDEX_AVAILABLE:
            try:
                save_index_and_chunks(doc, "saved_index")
                st.success("Index & chunks saved to saved_index/")
            except Exception as e:
                st.error("Save failed: " + str(e))
        else:
            st.warning("save_index_and_chunks not available in backend.", icon="⚠️")

# show generated summary
if "summary" in st.session_state:
    st.markdown(f"<div style='margin-top:10px'>{st.session_state['summary']}</div>", unsafe_allow_html=True)
else:
    # show short auto summary (one-liner) from small pipeline if available
    try:
        preview = summarize_document_hierarchical(doc, target_words=30)
    except Exception:
        preview = (doc.cleaned_text[:240] + "...") if hasattr(doc, "cleaned_text") else "Document loaded."
    st.markdown(f"<div style='color:{MUTED}; margin-top:6px'>{preview}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MIDDLE: Three equal columns ----------------
c1, c2, c3 = st.columns([1,1,1], gap="small")

# ---------------- Column 1: Summary details & download ----------------
with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📘 Summary</div>", unsafe_allow_html=True)
    if "summary" in st.session_state:
        st.markdown(f"<div style='font-size:14px;line-height:1.45'>{st.session_state['summary']}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        dl1, dl2 = st.columns([1,1])
        with dl1:
            st.download_button("Download TXT", st.session_state["summary"], "summary.txt", mime="text/plain")
        with dl2:
            if st.session_state.get("summary_pdf_path"):
                with open(st.session_state["summary_pdf_path"], "rb") as f:
                    st.download_button("Download PDF", f, "summary.pdf", mime="application/pdf")
            else:
                st.button("Export PDF (install fpdf)", disabled=True)
    else:
        st.markdown("<div class='small-muted'>No summary generated. Click <b>Generate Summary</b>.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Column 2: QA Engine — main interactive ----------------
with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🤖 QA Engine</div>", unsafe_allow_html=True)
    q = st.text_input("Ask (e.g., 'What is a Transformer model?')", key="qa_input")
    qcol1, qcol2 = st.columns([1,1])
    with qcol1:
        if st.button("Ask (Strict)"):
            if not q or q.strip()=="":
                st.warning("Type a question.")
            else:
                res = safe_answer_question(doc, q.strip(), strict=True)
                # fallback: if strict said 'not found' and we allow fallback
                if ("not found" in res.get("answer","").lower()) and strict_mode_default:
                    res_f = safe_answer_question(doc, q.strip(), strict=False)
                    st.session_state["qa"] = res_f
                    st.session_state["qa_mode"] = "Best-Effort (fallback)"
                else:
                    st.session_state["qa"] = res
                    st.session_state["qa_mode"] = "Strict"
    with qcol2:
        if st.button("Ask (Best Effort)"):
            if not q or q.strip()=="":
                st.warning("Type a question.")
            else:
                res = safe_answer_question(doc, q.strip(), strict=False)
                st.session_state["qa"] = res
                st.session_state["qa_mode"] = "Best-Effort"

    # show QA result
    if st.session_state.get("qa"):
        r = st.session_state["qa"]
        st.markdown(f"<div style='font-weight:700;color:{ACCENT_APPLE}'>Mode: {st.session_state.get('qa_mode','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:8px'>{r.get('answer','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Confidence: {r.get('confidence',0):.2f}</div>", unsafe_allow_html=True)
        # sources (if present)
        srcs = r.get("sources", []) or r.get("meta", {}).get("sources", [])
        if srcs:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted'>Retrieved chunks (top):</div>", unsafe_allow_html=True)
            for s in srcs:
                idx = s.get("index", s.get("idx","n/a"))
                wc = s.get("word_count", s.get("wc","n/a"))
                off = s.get("offset", s.get("approx_offset", s.get("approx_offset_words","n/a")))
                st.markdown(f"<div class='chunk'>[chunk {idx}] offset:{off} words:{wc}<div style='margin-top:6px'>{(''.join((doc.chunks[idx].text[:260]+'...') if idx < len(doc.chunks) else ''))}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Column 3: Insights, Chunks, Exports ----------------
with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Insights & Chunks</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>Chunks created: <span class='kv'>{len(doc.chunks)}</span></div>", unsafe_allow_html=True)
    # chunk preview scrollable region
    if st.button("Preview Top Chunks"):
        for c in doc.chunks[:min(12, len(doc.chunks))]:
            st.markdown(f"<div class='chunk'><b>IDX {c.index}</b>  OFF:{c.approx_offset_words}  WC:{c.word_count}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-bottom:12px'>{c.text[:420]}{'...' if len(c.text)>420 else ''}</div>", unsafe_allow_html=True)

    if st.button("Download cleaned text"):
        txt = getattr(doc, "cleaned_text", None) or getattr(doc, "raw_text", "")
        st.download_button("Download cleaned document", txt, file_name="cleaned.txt", mime="text/plain")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BOTTOM TECHNICAL PANELS (full width) ----------------
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>⚙️ Technical — Notes & Next Steps</div>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Project: LLM RAG — Hybrid Demo • Backend: sentence-transformers + FAISS + flan-t5</div>", unsafe_allow_html=True)
st.markdown("<ul class='small-muted'>"
            "<li>Chunking: overlap-based; use backend CHUNK_SIZE/CHUNK_OVERLAP</li>"
            "<li>Summary: hierarchical extractive→abstractive pipeline, target 100–150 words</li>"
            "<li>QA: strict-context LLM responses; fallback to best-effort generation when necessary</li>"
            "</ul>", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
# Takeaways display & download if exists
if st.session_state.get("tks"):
    st.markdown("<div class='section-title'>📌 Key Takeaways</div>", unsafe_allow_html=True)
    for t in st.session_state["tks"]:
        st.markdown(f"<div class='kv'>• {t}</div>", unsafe_allow_html=True)
    st.download_button("Download takeaways", "\n".join(st.session_state["tks"]), file_name="takeaways.txt", mime="text/plain")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("<div class='footer'>Obsidian • Apple×NVIDIA • Hybrid god-mode UI — Built for exam/demo • © 2025</div>", unsafe_allow_html=True)
