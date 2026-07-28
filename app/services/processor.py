from pathlib import Path
import json
import os
import tempfile
from ..db import update_document
from ..classifier import predict_category


def get_default_index_dir(provided: str = None) -> str:
    if provided:
        return provided
    env_dir = os.getenv("INDEX_DIR")
    if env_dir:
        target = Path(env_dir)
    elif os.getenv("VERCEL"):
        target = Path(tempfile.gettempdir()) / "index"
    else:
        target = Path("index")
    try:
        target.mkdir(parents=True, exist_ok=True)
        test_file = target / ".write_test"
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        return str(target)
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "index"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\x00", " ").split())


def extract_text_per_page(pdf_path: str):
    pages = []
    try:
        try:
            from PyPDF2 import PdfReader
        except Exception:
            from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
                pages.append(txt)
            except Exception:
                pages.append("")
    except Exception as exc:
        print(f"PDF extraction error for {pdf_path}: {exc}")
    return pages


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


def index_document(doc_id: str, chunks: list, index_dir: str = None):
    index_dir = get_default_index_dir(index_dir)
    p = Path(index_dir)
    p.mkdir(parents=True, exist_ok=True)
    chunks_file = p / f"{doc_id}_chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    registry = p / "registry.json"
    if registry.exists():
        with open(registry, "r", encoding="utf-8") as f:
            reg = json.load(f)
    else:
        reg = {}
    reg[doc_id] = {"chunks_file": str(chunks_file), "embedding_backend": "tfidf"}
    with open(registry, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False)


def process_document(doc_id: str, file_path: str):
    update_document(doc_id, status="processing", file_path=file_path)
    try:
        pages = extract_text_per_page(file_path)
        file_name = Path(file_path).name
        full_text = clean_text("\n\n".join(pages))
        chunks = chunk_pages(doc_id, file_name, pages)
        if not chunks:
            chunks = [
                {
                    "id": 0,
                    "chunk_id": f"{doc_id}_c0",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page_number": 1,
                    "text": full_text or f"Document {file_name} uploaded. (No selectable text extracted).",
                    "start": 0,
                    "end": len(full_text),
                }
            ]
        classification = predict_category(full_text[:20000])
        index_document(doc_id, chunks, index_dir=get_default_index_dir())
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


def load_registry(index_dir: str = None):
    index_dir = get_default_index_dir(index_dir)
    p = Path(index_dir)
    registry = p / "registry.json"
    if not registry.exists():
        return {}
    with open(registry, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_chunks(index_dir: str = None, doc_ids=None):
    index_dir = get_default_index_dir(index_dir)
    reg = load_registry(index_dir)
    selected = set(doc_ids or [])
    all_chunks = []
    for doc_id, info in reg.items():
        if selected and doc_id not in selected:
            continue
        chunks_file = info.get("chunks_file")
        if not chunks_file:
            continue
        if not Path(chunks_file).exists():
            continue
        with open(chunks_file, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                chunk.setdefault("doc_id", doc_id)
                chunk.setdefault("chunk_id", f"{doc_id}_c{chunk.get('id', 0)}")
                chunk.setdefault("page_number", None)
                all_chunks.append(chunk)
    return all_chunks


def keyword_score(query: str, text: str) -> float:
    query_terms = [term for term in clean_text(query).lower().split() if term]
    if not query_terms:
        return 0.0
    lowered = (text or "").lower()
    matches = sum(1 for term in query_terms if term in lowered)
    return matches / len(query_terms)


def search(query: str, top_k: int = 5, index_dir: str = None, mode: str = "semantic", doc_ids=None):
    index_dir = get_default_index_dir(index_dir)
    chunks = load_all_chunks(index_dir, doc_ids=doc_ids)

    if not chunks:
        return []

    keyword_scores = [keyword_score(query, text) for text in [c["text"] for c in chunks]]
    ranked = []
    for chunk, score in zip(chunks, keyword_scores):
        ranked.append({**chunk, "score": float(score)})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]
