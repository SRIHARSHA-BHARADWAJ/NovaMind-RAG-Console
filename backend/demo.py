"""
demo.py — Final CLI runner for rag_backend.py
Works with LLM_TEST_FINAL.pdf
- Generates 100–150 word summary
- Extracts factual takeaways
- Performs strict QA → fallback best-effort QA
"""

import sys
from rag_backend import (
    process_pdf,
    summarize_document_hierarchical,
    extract_takeaways,
    answer_question,
    save_summary_txt,
    save_summary_pdf,
)

# ------------------ Pretty Printers ------------------

def pretty_summary(text):
    print("\n================ SUMMARY ================\n")
    print(text)
    print("\n=========================================\n")

def pretty_takeaways(points):
    print("\n================ TAKEAWAYS ================\n")
    for p in points:
        print(f"• {p}")
    print("\n===========================================\n")

def pretty_qa(label, res):
    print(f"\n=========== QA ({label}) ===========\n")
    print("Answer:", res["answer"])
    print("\nConfidence:", res.get("confidence"))
    print("Sources:", res.get("sources"))
    print("\n============================================\n")

# ------------------ MAIN PIPELINE ------------------

def main(path):
    print("\nProcessing PDF:", path)
    print("Please wait... (extracting, chunking, embedding)\n")

    doc = process_pdf(path)
    print(f"Document Loaded. Chunks created: {len(doc.chunks)}\n")

    # ----------- Summary (100–150 words) -----------
    summary = summarize_document_hierarchical(doc, target_words=130)
    pretty_summary(summary)

    txt_path = save_summary_txt(summary, out_dir="outputs", prefix="summary")
    print("✔ Summary saved as TXT at:", txt_path)

    try:
        pdf_path = save_summary_pdf(summary, out_dir="outputs", prefix="summary")
        print("✔ Summary PDF saved at:", pdf_path)
    except:
        print("✘ PDF export skipped (fpdf not installed). Run: pip install fpdf")

    # ----------- Takeaways -----------
    takeaways = extract_takeaways(doc, n=6)
    pretty_takeaways(takeaways)

    # ----------- QA (Strict + Best-Effort fallback) -----------
    questions = [ "What is a Transformer model in NLP?",
                 "How does the self-attention mechanism work?"]

    for q in questions:
        print("Question:", q)

        # Strict mode first
        strict_result = answer_question(doc, q, strict=True)

        if "not found" not in strict_result["answer"].lower():
            pretty_qa("Strict", strict_result)
        else:
            # Fallback - best effort
            best_effort = answer_question(doc, q, strict=False)
            pretty_qa("Best-Effort", best_effort)

# ------------------ Entry ------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo.py <path_to_pdf>")
    else:
        main(sys.argv[1])
