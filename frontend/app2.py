import sys
import os
import streamlit as st
# Add backend folder to PYTHONPATH (VERY IMPORTANT FOR STREAMLIT CLOUD)
BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if BACKEND_PATH not in sys.path:
    sys.path.append(BACKEND_PATH)

from rag_backend import (
    process_pdf,
    summarize_document_hierarchical,
    extract_takeaways,
    answer_question,
    save_summary_txt,
)



# ----------------------------------------------------
#                GLOBAL PAGE SETTINGS
# ----------------------------------------------------
st.set_page_config(
    page_title="NovaMind LLM Research Console",
    layout="wide",
    page_icon="🧠"
)

# ----------------------------------------------------
#        CUSTOM CSS — APPLE × NVIDIA GOD UI
# ----------------------------------------------------
st.markdown("""
<style>

body {
    background: radial-gradient(circle at top left, #090d10, #020305, #000000);
    color: #e8f0ff;
}

/* Hide Streamlit defaults */
header, footer {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(12, 18, 24, 0.55);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(0,255,170,0.25);
    box-shadow: inset -4px 0px 15px rgba(0,255,170,0.08);
}

.sidebar-title {
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg,#00ffa2,#00d2ff);
    -webkit-background-clip: text;
    color: transparent;
    padding-bottom: 10px;
}

/* Glass Cards */
.glass-card {
    background: rgba(18, 25, 35, 0.55);
    padding: 22px 28px;
    border-radius: 18px;
    border: 1px solid rgba(0, 255, 170, 0.18);
    backdrop-filter: blur(14px);
    box-shadow: 0 0 25px rgba(0,255,170,0.08);
    transition: 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 35px rgba(0,255,170,0.25);
}

/* Section Titles */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: -10px;
    padding-left: 4px;
    background: linear-gradient(90deg,#00ffa2,#00aaff);
    -webkit-background-clip: text;
    color: transparent;
}

/* Neon Text */
.neon {
    color: #00ffc6;
    text-shadow: 0 0 12px #00ffc6aa;
}

/* Inputs */
textarea, input {
    background: rgba(30,30,40,0.4)!important;
    border: 1px solid rgba(0,255,170,0.3)!important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#00ffa2,#00aaff);
    border: none;
    padding: 0.7rem 1.3rem;
    border-radius: 10px;
    color: black;
    font-weight: 700;
    transition: 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.04);
    box-shadow: 0 0 25px rgba(0,255,200,0.7);
}

/* Containers */
.block-container {
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
#                  SIDEBAR CONTENT
# ----------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>NovaMind LLM Research Console</div>", unsafe_allow_html=True)

st.sidebar.markdown("""
### **Model Specifications**
- **Embedding Engine:** all-MiniLM-L6-v2  
- **LLM Core:** FLAN-T5-Large  
- **Vector Index:** FAISS L2 Optimized  
- **Pipeline Mode:** Hierarchical Semantic RAG  
- **Synthesis Layer:** Extractive → Abstractive Neural Fusion  

---

### **System Architect**
**Harsha Bharadwaj**  
Machine Learning Engineer  
GitHub: **SRIHARSHA-BHARADWAJ**

---

### **System Metadata**
- Build: **NovaMind RC-3.2**  
- Frameworks: PyMuPDF • Sentence-BERT • FAISS • Transformers  
- UI Engine: Liquid-Glass Neon Adaptive Surface  

---
""")

# ----------------------------------------------------
#                 MAIN PAGE HEADER
# ----------------------------------------------------
st.markdown("""
<div style='padding: 10px 0 25px 5px;'>
  <h1 class='neon' style='font-size:48px;'>
    NovaMind LLM Research Console
  </h1>
  <p style='font-size:18px;color:#8b9bb5;margin-top:-10px;'>
      Neural Document Reasoning • Transformer-Based Semantic Intelligence • Retrieval-Augmented Neural Comprehension
  </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
#          UPLOAD PANEL (FULL-WIDTH)
# =====================================================
st.markdown("<div class='section-title'>Document Ingestion</div>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
uploaded_pdf = st.file_uploader("Upload a PDF for Neural Processing", type=["pdf"])
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_pdf:
    st.info("Awaiting document upload… Neural cores idle.")
    st.stop()

# PROCESS PDF
with st.spinner("Initializing Neural Pipeline… Extracting → Chunking → Embedding → Indexing"):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.read())
    doc = process_pdf("temp.pdf")

# =====================================================
#                 SUMMARY (FULL WIDTH)
# =====================================================
st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

summary = summarize_document_hierarchical(doc)
st.write(summary)

txt_path = save_summary_txt(summary, prefix="NovaMind_Summary")
st.download_button("⬇ Download Summary (.txt)", open(txt_path, "rb"), file_name="summary.txt")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
#         THREE-PANEL LAYOUT (QA / TAKEAWAYS / META)
# =====================================================
col1, col2, col3 = st.columns(3)

# ----------- COLUMN 1: QUERY ENGINE ----------
with col1:
    st.markdown("<div class='section-title'>Query Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    question = st.text_input("Enter a technical query:")
    if question:
        res = answer_question(doc, question)
        st.subheader("Answer:")
        st.write(res["answer"])
        st.caption(f"Confidence: {res.get('confidence'):.3f}")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------- COLUMN 2: KEY INSIGHTS ----------
with col2:
    st.markdown("<div class='section-title'>Key Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    takeaways = extract_takeaways(doc, n=6)
    for t in takeaways:
        st.write("• " + t)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------- COLUMN 3: SEMANTIC METADATA ----------
with col3:
    st.markdown("<div class='section-title'>Semantic Matrix</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.write(f"Total Semantic Chunks: **{len(doc.chunks)}**")
    st.write("Indexed using hierarchical vector grouping optimized for Transformer-based retrieval inference.")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
#                     FOOTER
# =====================================================
st.markdown("""
<hr style='border: 1px solid rgba(0,255,170,0.25);'>
<p style='text-align:center;color:#556677;font-size:14px;margin-top:10px;'>
NovaMind LLM Research Console • Neural RAG Intelligence Framework • RC-3.2  
<br>Designed & Engineered by <b>Harsha Bharadwaj</b> — GitHub: <b>SRIHARSHA-BHARADWAJ</b>
</p>
""", unsafe_allow_html=True)
