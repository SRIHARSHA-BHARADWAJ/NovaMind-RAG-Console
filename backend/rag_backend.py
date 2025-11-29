"""
rag_backend.py  -- Improved RAG backend for robust document QA & summarization.

Key improvements compared to previous:
 - smaller chunk_size + overlap for more chunks (better retrieval)
 - sentence-level fallback splitting when doc is very short
 - safe llm.generate usage (no unsupported flags)
 - two-stage QA: strict mode + best-effort fallback when retrieval confidence is low
 - improved hierarchical summarization with longer chunk summaries
 - configurable thresholds
"""

import os
import re
import uuid
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# PDF export (for summary)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# ---------------- CONFIG (tweakable) ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"
PREFERRED_LLM = "google/flan-t5-base"
FALLBACK_LLM = "google/flan-t5-small"

# Aggressive chunking for high-quality retrieval
CHUNK_SIZE = 150
CHUNK_OVERLAP = 50

TOP_K = 12
FINAL_K = 4
SUMMARY_CHUNK_TOP_N = 10
TARGET_SUMMARY_WORDS = 170

# Retrieval confidence threshold for strict answers (0-1)
RETRIEVAL_CONF_THRESHOLD = 0.40

# ---------------- global singletons (lazy load) ----------------
_embedder = SentenceTransformer(EMBED_MODEL)

_tokenizer = None
_llm = None
_LLM_NAME = None

def load_llm(preferred: str = PREFERRED_LLM, fallback: str = FALLBACK_LLM):
    """Lazy load LLM with fallback. Safe for different model types."""
    global _tokenizer, _llm, _LLM_NAME
    if _llm is not None:
        return
    try:
        _tokenizer = AutoTokenizer.from_pretrained(preferred, use_fast=True)
        _llm = AutoModelForSeq2SeqLM.from_pretrained(preferred)
        _LLM_NAME = preferred
        print(f"[rag_backend] Loaded LLM: {preferred}")
    except Exception as e:
        warnings.warn(f"[rag_backend] Could not load preferred LLM {preferred}: {e}. Falling back to {fallback}.")
        _tokenizer = AutoTokenizer.from_pretrained(fallback, use_fast=True)
        _llm = AutoModelForSeq2SeqLM.from_pretrained(fallback)
        _LLM_NAME = fallback
        print(f"[rag_backend] Loaded fallback LLM: {fallback}")

# ---------------- data classes ----------------
@dataclass
class ChunkMeta:
    index: int
    text: str
    word_count: int
    approx_offset_words: int

@dataclass
class Document:
    path: str
    raw_text: str
    cleaned_text: str
    chunks: List[ChunkMeta] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    faiss_index: Optional[faiss.IndexFlatL2] = None

# ---------------- extraction ----------------
def extract_text_pymupdf(pdf_path: str) -> str:
    """Extract text using pymupdf. Returns concatenated page text."""
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        try:
            txt = page.get_text("text")
        except Exception:
            txt = page.get_text()
        if txt:
            parts.append(txt)
    doc.close()
    return "\n\n".join(parts)

def extract_text_with_ocr(pdf_path: str) -> str:
    """Fallback OCR: render each page as an image and run pytesseract."""
    if not OCR_AVAILABLE:
        raise RuntimeError("pytesseract not available. Install 'pytesseract' and pillow to enable OCR.")
    doc = fitz.open(pdf_path)
    pages_text = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        # PIL can open from bytes buffer
        from io import BytesIO
        img = Image.open(BytesIO(img_bytes))
        txt = pytesseract.image_to_string(img)
        pages_text.append(txt)
    doc.close()
    return "\n\n".join(pages_text)

def extract_text_auto(pdf_path: str, ocr_if_needed: bool = True) -> str:
    """Try pymupdf extraction; if result is too small and OCR allowed, run OCR."""
    text = extract_text_pymupdf(pdf_path)
    if (len(text.strip().split()) < 80) and ocr_if_needed and OCR_AVAILABLE:
        print("[rag_backend] Low-extraction output, running OCR fallback...")
        try:
            ocr_text = extract_text_with_ocr(pdf_path)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        except Exception as e:
            warnings.warn(f"[rag_backend] OCR extraction failed: {e}")
    return text

# ---------------- cleaning ----------------
def fix_hyphens_and_linebreaks(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preserve_bullets(text: str) -> str:
    text = text.replace("\u2022", " • ")
    text = re.sub(r"\s*-\s+", " • ", text)
    return text

def clean_report_text(raw: str) -> str:
    t = raw.replace("\r", " ")
    t = preserve_bullets(t)
    t = fix_hyphens_and_linebreaks(t)
    t = re.sub(r"[ ]{2,}", " ", t)
    return t.strip()

# ---------------- chunking ----------------
def chunk_text_with_meta(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[ChunkMeta]:
    words = text.split()
    if len(words) == 0:
        return []
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(ChunkMeta(index=idx, text=chunk, word_count=len(chunk.split()), approx_offset_words=start))
        idx += 1
        if end == len(words):
            break
        start = end - overlap
    # If we still have too few chunks (e.g. doc tiny), create sentence-based chunks
    if len(chunks) < 6:
        sent_chunks = _sentence_fallback_chunks(text, target_chunk_size=round(chunk_size/2))
        if len(sent_chunks) > len(chunks):
            chunks = sent_chunks
    return chunks

def _sentence_fallback_chunks(text: str, target_chunk_size: int = 80) -> List[ChunkMeta]:
    # split into sentences (simple) then group to target size
    sats = re.split(r'(?<=[\.\?\!])\s+', text)
    words = []
    chunks = []
    idx = 0
    cursor = 0
    for s in sats:
        s = s.strip()
        if not s:
            continue
        sw = s.split()
        words.extend(sw)
        # when words exceed target_chunk_size, flush chunk
        if len(words) >= target_chunk_size:
            chunk = " ".join(words)
            chunks.append(ChunkMeta(index=idx, text=chunk, word_count=len(chunk.split()), approx_offset_words=cursor))
            cursor += len(words)
            idx += 1
            words = []
    if words:
        chunks.append(ChunkMeta(index=idx, text=" ".join(words), word_count=len(words), approx_offset_words=cursor))
    return chunks

# ---------------- embeddings & faiss ----------------
def embed_chunks(chunks: List[ChunkMeta]) -> np.ndarray:
    texts = [c.text for c in chunks]
    emb = _embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# ---------------- retrieval + rerank ----------------
def l2_to_cosine(qvec: np.ndarray, vec: np.ndarray) -> float:
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    v = vec / (np.linalg.norm(vec) + 1e-9)
    return float(np.dot(q, v))

def token_overlap_score(query: str, chunk_text: str) -> float:
    qset = set(re.findall(r"\w+", query.lower()))
    if not qset:
        return 0.0
    cset = set(re.findall(r"\w+", chunk_text.lower()))
    return len(qset & cset) / len(qset)

def retrieve_and_rerank(doc: Document, query: str, top_k: int = TOP_K, final_k: int = FINAL_K):
    if doc.faiss_index is None or doc.embeddings is None or len(doc.chunks) == 0:
        return [], []
    qvec = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = doc.faiss_index.search(qvec, top_k)
    retrieved = []
    scores = []
    for idx in I[0]:
        if idx < 0 or idx >= len(doc.chunks):
            continue
        c = doc.chunks[idx]
        emb = doc.embeddings[idx]
        cos = l2_to_cosine(qvec[0], emb)
        overlap = token_overlap_score(query, c.text)
        score = 0.8 * cos + 0.2 * overlap
        retrieved.append(c)
        scores.append(score)
    pairs = list(zip(retrieved, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:final_k]
    top_chunks = [p[0] for p in top_pairs]
    top_scores = [p[1] for p in top_pairs]
    return top_chunks, top_scores

# ---------------- LLM generation helpers ----------------
def llm_generate(prompt: str, max_new_tokens: int = 220) -> str:
    load_llm()
    # safe tokenization: truncate long prompts
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = _llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,   # deterministic
        # do not pass temperature/top_p to models that don't accept them
    )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ---------------- hierarchical summarization ----------------
def summarize_document_hierarchical(doc: Document, target_words: int = TARGET_SUMMARY_WORDS) -> str:
    if not doc.chunks or doc.embeddings is None:
        return "No extractable text found in document."

    # choose representative chunks (closer to centroid = central content)
    centroid = np.mean(doc.embeddings, axis=0, keepdims=True)
    dists = np.linalg.norm(doc.embeddings - centroid, axis=1)
    idxs = np.argsort(dists)[:SUMMARY_CHUNK_TOP_N]
    selected = [doc.chunks[i] for i in idxs]

    # produce a slightly longer summary for each selected chunk (2-3 sentences)
    chunk_summaries = []
    for c in selected:
        prompt = f"Summarize the following excerpt in 2-3 concise sentences, factual and technical:\n\n{c.text}\n\nSummary:"
        s = llm_generate(prompt, max_new_tokens=100)
        if not s:
            s = c.text.split(".")[0].strip()
        chunk_summaries.append(s)

    combined = " ".join(chunk_summaries)
    final_prompt = f"""
You are a technical summarizer. Using only the text below (short excerpt-summaries),
create a single coherent paragraph of approximately {target_words} words summarizing the entire document.
Text:
{combined}

Summary (~{target_words} words):
"""
    final = llm_generate(final_prompt, max_new_tokens=300)
    words = final.split()
    if len(words) < int(target_words*0.6):
        # fallback: summarize the entire cleaned text if final is too short
        fallback_prompt = f"Summarize this document into approximately {target_words} words:\n\n{doc.cleaned_text}\n\nSummary:"
        final = llm_generate(fallback_prompt, max_new_tokens=300)
    # enforce rough upper length
    words = final.split()
    if len(words) > target_words + 50:
        final = " ".join(words[: target_words + 10]) + "..."
    return final

# ---------------- takeaways ----------------
def extract_takeaways(doc: Document, n: int = 6) -> List[str]:
    if not doc.chunks or doc.embeddings is None:
        return []
    centroid = np.mean(doc.embeddings, axis=0, keepdims=True)
    dists = np.linalg.norm(doc.embeddings - centroid, axis=1)
    idxs = np.argsort(dists)[: max(n, SUMMARY_CHUNK_TOP_N)]
    selected = [doc.chunks[i] for i in idxs[:n]]
    combined = "\n\n".join([c.text for c in selected])
    prompt = f"Extract {n} concise takeaway bullet points from the following excerpts. Use short factual sentences:\n\n{combined}\n\nTakeaways:"
    out = llm_generate(prompt, max_new_tokens=200)
    lines = [l.strip("-• \n\r") for l in re.split(r'[\n\r]+', out) if l.strip()]
    if not lines:
        lines = [c.text.split(".")[0].strip() for c in selected[:n]]
    return lines[:n]

# ---------------- intent detection ----------------
def detect_intent(question: str) -> str:
    q = question.lower()
    if re.search(r'\b(summar(y|ise)|overview|abstract|conclude|conclusion)\b', q):
        return "summary"
    if re.search(r'\b(takeaway|key point|key takeaway|highlights|bullets)\b', q):
        return "takeaways"
    if re.search(r'\b(name|email|phone|contact|address)\b', q):
        return "contact"
    return "qa"

def extract_contact(doc: Document) -> Dict[str, str]:
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}", doc.cleaned_text)
    phone = re.findall(r"\+?\d{1,3}\s?\d{5,12}", doc.cleaned_text)
    header = doc.cleaned_text[:200]
    name_match = re.findall(r"([A-Z][a-z]{2,}\s?[A-Z]?[a-z]{0,})", header)
    name = name_match[0].strip() if name_match else "Not found"
    return {"name": name, "email": (emails[0] if emails else "Not found"), "phone": (phone[0] if phone else "Not found")}

# ---------------- QA ----------------
def answer_question(doc: Document, question: str, strict: bool = True) -> Dict:
    """
    strict=True -> only answer if retrieval confidence >= RETRIEVAL_CONF_THRESHOLD (otherwise say 'Answer not found')
    strict=False -> produce a best-effort answer grounded in top retrieved chunks (prefix 'Best-effort:')
    """
    intent = detect_intent(question)
    if intent == "summary":
        return {"type": "summary", "text": summarize_document_hierarchical(doc)}
    if intent == "takeaways":
        return {"type": "takeaways", "points": extract_takeaways(doc)}
    if intent == "contact":
        return {"type": "contact", "contact": extract_contact(doc)}

    top_chunks, top_scores = retrieve_and_rerank(doc, question, top_k=TOP_K, final_k=FINAL_K)
    if not top_chunks:
        return {"type": "qa", "answer": "No relevant content found.", "sources": [], "confidence": 0.0}

    mean_score = float(np.mean(top_scores)) if len(top_scores) else 0.0

    # Build strict-context prompt
    context_parts = [f"[p{i}] {c.text}" for i, c in enumerate(top_chunks)]
    context = "\n\n".join(context_parts)
    strict_prompt = (f"You are an assistant that MUST answer using ONLY the context below.\n"
                     "If the answer is not present, reply exactly: 'Answer not found in the document.'\n\n"
                     f"Context:\n{context}\n\nQuestion:\n{question}\n\n"
                     "Answer succinctly and reference paragraphs like [p0].")
    answer_text = llm_generate(strict_prompt, max_new_tokens=220)

    # If strict mode requested, enforce threshold and LLM's answer check
    if strict:
        # If retrieval confidence low or LLM indicates "not found", return not found
        if mean_score < RETRIEVAL_CONF_THRESHOLD or re.search(r'answer not found|not present|not in the document', answer_text.lower()):
            return {"type": "qa", "answer": "Answer not found in the document", "sources": [{"index": c.index, "word_count": c.word_count, "offset": c.approx_offset_words} for c in top_chunks], "confidence": mean_score}
        # otherwise return LLM answer
        return {"type": "qa", "answer": answer_text, "sources": [{"index": c.index, "word_count": c.word_count, "offset": c.approx_offset_words} for c in top_chunks], "confidence": mean_score}
    else:
        # best-effort mode: produce an answer from context, even if retrieval score low
        best_effort_prompt = (f"Using only the context below, produce a concise factual answer to the question. "
                              f"If uncertain, mark as 'Possibly' and provide the most likely answer. Context:\n{context}\n\nQ: {question}\nA:")
        be_ans = llm_generate(best_effort_prompt, max_new_tokens=220)
        prefixed = "Best-effort: " + be_ans
        return {"type": "qa", "answer": prefixed, "sources": [{"index": c.index, "word_count": c.word_count, "offset": c.approx_offset_words} for c in top_chunks], "confidence": mean_score}

# ---------------- pipeline ----------------
def process_pdf(path: str, use_ocr_if_needed: bool = True) -> Document:
    raw = extract_text_auto(path, ocr_if_needed=use_ocr_if_needed)
    cleaned = clean_report_text(raw)
    chunks = chunk_text_with_meta(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        return Document(path=path, raw_text=raw, cleaned_text=cleaned, chunks=[], embeddings=None, faiss_index=None)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    return Document(path=path, raw_text=raw, cleaned_text=cleaned, chunks=chunks, embeddings=embeddings, faiss_index=index)

# ---------------- save summary text & pdf ----------------
def save_summary_txt(text: str, out_dir: str = "outputs", prefix: str = "summary") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.txt"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf8") as f:
        f.write(text)
    return fpath

def save_summary_pdf(text: str, out_dir: str = "outputs", prefix: str = "summary") -> str:
    if not FPDF_AVAILABLE:
        raise RuntimeError("FPDF not installed. Install 'fpdf' to enable PDF export.")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.pdf"
    fpath = os.path.join(out_dir, fname)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    max_chars = 95
    lines = []
    for paragraph in text.split("\n\n"):
        words = paragraph.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > max_chars:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        lines.append("")
    for ln in lines:
        pdf.cell(0, 8, ln, ln=1)
    pdf.output(fpath)
    return fpath

# ---------------- persistence ----------------
def save_index_and_chunks(doc: Document, basepath: str):
    os.makedirs(basepath, exist_ok=True)
    if doc.faiss_index is not None:
        faiss.write_index(doc.faiss_index, os.path.join(basepath, "index.faiss"))
    if doc.embeddings is not None:
        np.save(os.path.join(basepath, "embeddings.npy"), doc.embeddings)
    with open(os.path.join(basepath, "chunks.txt"), "w", encoding="utf8") as f:
        for c in doc.chunks:
            f.write(f"IDX:{c.index}\tOFF:{c.approx_offset_words}\tWC:{c.word_count}\n")
            f.write(c.text.replace("\n", " ") + "\n<<<END>>>\n")
