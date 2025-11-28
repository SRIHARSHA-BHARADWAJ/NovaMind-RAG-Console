import streamlit as st
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
    page_title="AstraVision LLM Intelligence Suite",
    layout="wide",
    page_icon="✨"
)

# ----------------------------------------------------
#        CUSTOM CSS — GOD MODE APPLE × NVIDIA UI
# ----------------------------------------------------
st.markdown("""
<style>

body {
    background: radial-gradient(circle at top left, #0a0f14, #020304, #000000);
    color: #e8f0ff;
}

/* Hide default Streamlit elements */
header, footer {visibility: hidden;}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 20, 0.55);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(0,255,170,0.25);
    box-shadow: inset -2px 0px 15px rgba(0,255,170,0.08);
}
.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    background: linear-gradient(90deg,#00ffa2,#00d2ff);
    -webkit-background-clip: text;
    color: transparent;
}

/* Glass Cards */
.glass-card {
    background: rgba(15, 25, 35, 0.55);
    padding: 22px 28px;
    border-radius: 18px;
    border: 1px solid rgba(0, 255, 170, 0.18);
    backdrop-filter: blur(14px);
    box-shadow: 0 0 25px rgba(0,255,170,0.08);
    transition: 0.25s ease-in-out;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 35px rgba(0,255,170,0.25);
}

/* Titles */
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

/* Input box */
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
    transition: 0.2s;
}
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 25px rgba(0,255,200,0.7);
}

/* Full-width container fix */
.block-container {
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
#                  SIDEBAR CONTENT
# ----------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>AstraVision LLM</div>", unsafe_allow_html=True)

st.sidebar.markdown("""
### **Model Specifications**
- **Embedding:** all-MiniLM-L6-v2  
- **LLM Core:** FLAN-T5-Large  
- **Pipeline:** Hierarchical RAG  
- **Vector Index:** FAISS L2 Optimized  
- **Compression:** Extractive-Abstractive Hybrid  

---
### **System Architect**
**Sriharsha Bharadwaj**  
GitHub: **@SRIHARSHA-BHARADWAJ**

---
""")

# ----------------------------------------------------
#                 MAIN PAGE HEADER
# ----------------------------------------------------
st.markdown("""
<div style='padding: 10px 0 25px 5px;'>
  <h1 class='neon' style='font-size:48px;'>
    AstraVision LLM Intelligence Suite
  </h1>
  <p style='font-size:18px;color:#8b9bb5;margin-top:-10px;'>
      Neural Document Reasoning • Transformer-driven Semantic Extraction • Retrieval-Augmented Comprehension
  </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
#          UPLOAD ZONE (FULL WIDTH GLASS CARD)
# =====================================================
with st.container():
    st.markdown("<div class='section-title'>Document Ingestion</div>", unsafe_allow_html=True)
    with st.container():
        upload_card = st.container()
        with upload_card:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

            pdf_file = st.file_uploader("Upload a PDF for semantic analysis", type=["pdf"])

            st.markdown("</div>", unsafe_allow_html=True)

# If no file uploaded → show placeholder UI
if not pdf_file:
    st.info("Awaiting document upload… Neural pipeline idle.")
    st.stop()

# Process PDF
with st.spinner("Initializing Neural Pipeline… Extracting → Chunking → Embedding → Indexing"):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    doc = process_pdf("temp.pdf")

# =====================================================
#                 SUMMARY (FULL WIDTH)
# =====================================================
st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

summary = summarize_document_hierarchical(doc)
st.write(summary)

summary_path = save_summary_txt(summary, prefix="AstraVision_Summary")
st.download_button("⬇ Download Summary (.txt)", open(summary_path, "rb"), file_name="summary.txt")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
#                 3-COLUMN PANEL (QA / Takeaways / Chunks)
# =====================================================
col1, col2, col3 = st.columns(3)

# ----------- COLUMN 1: QA ENGINE ----------
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

# ----------- COLUMN 2: TAKEAWAYS ----------
with col2:
    st.markdown("<div class='section-title'>Key Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    takeaways = extract_takeaways(doc, n=6)
    for t in takeaways:
        st.write("• " + t)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------- COLUMN 3: CHUNK ANALYSIS ----------
with col3:
    st.markdown("<div class='section-title'>Semantic Chunks</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write(f"Total Chunks: **{len(doc.chunks)}**")
    st.write("The system uses hierarchical semantic grouping optimized for Transformer-based retrieval.")
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
#                FOOTER (NEON)
# =====================================================
st.markdown("""
<hr style='border: 1px solid rgba(0,255,170,0.25);'>
<p style='text-align:center;color:#556677;font-size:14px;margin-top:10px;'>
AstraVision LLM Intelligence Suite • Powered by Neural RAG Architecture • Designed by Sriharsha Bharadwaj
</p>
""", unsafe_allow_html=True)
