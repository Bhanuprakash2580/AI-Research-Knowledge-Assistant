from pathlib import Path
from PyPDF2 import PdfReader
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ..db import update_document
from ..classifier import predict_category


VECTOR_MODEL = None
SENTENCE_MODEL = None


def get_vectorizer():
    global VECTOR_MODEL
    if VECTOR_MODEL is None:
        VECTOR_MODEL = TfidfVectorizer(stop_words='english', max_features=5000)
    return VECTOR_MODEL


def get_sentence_model():
    global SENTENCE_MODEL
    if SENTENCE_MODEL is not None:
        return SENTENCE_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        SENTENCE_MODEL = SentenceTransformer(model_name)
        return SENTENCE_MODEL
    except Exception:
        return None


def embedding_backend():
    configured = os.getenv("EMBEDDING_BACKEND", "auto").lower()
    if configured == "tfidf":
        return "tfidf"
    if get_sentence_model() is not None:
        return "sentence-transformers"
    return "tfidf"


def extract_text_per_page(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)
    return pages


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\x00", " ").split())


def chunk_text(text: str, max_chars=1000, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    idx = 0
    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end]
        chunks.append({"id": idx, "text": chunk, "start": start, "end": end})
        idx += 1
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_pages(doc_id: str, file_name: str, pages: list, max_chars=1000, overlap=150):
    chunks = []
    chunk_idx = 0
    step = max(1, max_chars - overlap)
    for page_number, page_text in enumerate(pages, start=1):
        text = clean_text(page_text)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(
                {
                    "id": chunk_idx,
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page_number": page_number,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                }
            )
            chunk_idx += 1
            if end == len(text):
                break
            start += step
    return chunks


def index_document(doc_id: str, chunks: list, embeddings: np.ndarray, index_dir: str = "index", backend: str = "tfidf"):
    p = Path(index_dir)
    p.mkdir(parents=True, exist_ok=True)
    emb_file = p / f"{doc_id}.npy"
    chunks_file = p / f"{doc_id}_chunks.json"
    np.save(emb_file, embeddings)
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    # register in registry
    registry = p / "registry.json"
    if registry.exists():
        with open(registry, "r", encoding="utf-8") as f:
            reg = json.load(f)
    else:
        reg = {}
    reg[doc_id] = {"emb_file": str(emb_file), "chunks_file": str(chunks_file), "embedding_backend": backend}
    with open(registry, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False)


def process_document(doc_id: str, file_path: str):
    update_document(doc_id, status="processing", file_path=file_path)
    try:
        pages = extract_text_per_page(file_path)
        file_name = Path(file_path).name
        full_text = clean_text("\n\n".join(pages))
        chunks = chunk_pages(doc_id, file_name, pages)
        texts = [c["text"] for c in chunks]
        backend = embedding_backend()
        if texts and backend == "sentence-transformers":
            model = get_sentence_model()
            embeddings = np.asarray(model.encode(texts, normalize_embeddings=True))
        elif texts:
            vectorizer = get_vectorizer()
            embeddings = vectorizer.fit_transform(texts).toarray()
        else:
            embeddings = np.zeros((0, 1))
        classification = predict_category(full_text[:20000])
        index_document(doc_id, chunks, embeddings, backend=backend)
        update_document(
            doc_id,
            total_pages=len(pages),
            total_chunks=len(chunks),
            status="processed",
            category=classification.get("category"),
            classification_confidence=classification.get("confidence"),
            classifier_note=classification.get("note"),
            file_path=file_path,
        )
    except Exception as exc:
        update_document(doc_id, status="failed", classifier_note=str(exc), file_path=file_path)
        raise


def load_registry(index_dir: str = "index"):
    p = Path(index_dir)
    registry = p / "registry.json"
    if not registry.exists():
        return {}
    with open(registry, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_chunks(index_dir: str = "index", doc_ids=None):
    reg = load_registry(index_dir)
    selected = set(doc_ids or [])
    all_chunks = []
    for doc_id, info in reg.items():
        if selected and doc_id not in selected:
            continue
        with open(info["chunks_file"], "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                chunk.setdefault("doc_id", doc_id)
                chunk.setdefault("chunk_id", f"{doc_id}_c{chunk.get('id', 0)}")
                chunk.setdefault("page_number", None)
                all_chunks.append(chunk)
    return all_chunks


def semantic_search_with_sentence_transformers(query: str, index_dir: str, doc_ids=None):
    model = get_sentence_model()
    if model is None:
        return None
    selected = set(doc_ids or [])
    q_vec = np.asarray(model.encode([query], normalize_embeddings=True))[0]
    scored = []
    for doc_id, info in load_registry(index_dir).items():
        if selected and doc_id not in selected:
            continue
        if info.get("embedding_backend") != "sentence-transformers":
            return None
        embeddings = np.load(info["emb_file"])
        if embeddings.size == 0:
            continue
        with open(info["chunks_file"], "r", encoding="utf-8") as f:
            chunks = json.load(f)
        scores = embeddings.dot(q_vec)
        for idx, chunk in enumerate(chunks):
            scored.append((float(scores[idx]), chunk))
    return scored


def keyword_score(query: str, text: str) -> float:
    query_terms = [term for term in clean_text(query).lower().split() if term]
    if not query_terms:
        return 0.0
    lowered = (text or "").lower()
    matches = sum(1 for term in query_terms if term in lowered)
    return matches / len(query_terms)


def search(query: str, top_k: int = 5, index_dir: str = "index", mode: str = "semantic", doc_ids=None):
    st_scored = None
    if mode in ("semantic", "hybrid"):
        st_scored = semantic_search_with_sentence_transformers(query, index_dir, doc_ids=doc_ids)

    if st_scored is not None:
        chunks = [chunk for _, chunk in st_scored]
        semantic_scores = np.array([score for score, _ in st_scored])
    else:
        chunks = load_all_chunks(index_dir, doc_ids=doc_ids)
        if not chunks:
            return []
        texts = [c["text"] for c in chunks]
        semantic_scores = np.zeros(len(chunks))
    texts = [c["text"] for c in chunks]
    if mode in ("semantic", "hybrid") and st_scored is None:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        matrix = vectorizer.fit_transform(texts)
        q_vec = vectorizer.transform([query])
        sims = (matrix @ q_vec.T).toarray().ravel()
        semantic_scores = sims

    keyword_scores = np.array([keyword_score(query, text) for text in texts])
    if mode == "keyword":
        scores = keyword_scores
    elif mode == "hybrid":
        scores = (0.7 * semantic_scores) + (0.3 * keyword_scores)
    else:
        scores = semantic_scores

    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        chunk = chunks[i]
        results.append(
            {
                "doc_id": chunk.get("doc_id"),
                "chunk_id": chunk.get("chunk_id", chunk.get("id")),
                "file_name": chunk.get("file_name"),
                "page_number": chunk.get("page_number"),
                "text": chunk.get("text", ""),
                "score": float(scores[i]),
                "semantic_score": float(semantic_scores[i]),
                "keyword_score": float(keyword_scores[i]),
            }
        )
    results = sorted(results, key=lambda r: -r["score"])[:top_k]
    return results
