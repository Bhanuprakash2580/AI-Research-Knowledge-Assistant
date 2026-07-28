from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ..services import memory
from ..db import record_query
import os
from typing import List

router = APIRouter(prefix="/search", tags=["search"])


def _get_search_backend():
    try:
        from ..services.processor import search as search_backend
    except Exception as exc:
        raise HTTPException(status_code=501, detail="Search backend unavailable") from exc
    return search_backend


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="semantic", pattern="^(keyword|semantic|hybrid)$")
    doc_ids: List[str] | None = None


class QARequest(SearchRequest):
    session_id: str | None = None


@router.get("/semantic")
def semantic_search(q: str, k: int = 5):
    if not q:
        raise HTTPException(status_code=400, detail="Query required")
    search_backend = _get_search_backend()
    results = search_backend(q, top_k=k, index_dir=os.getenv("INDEX_DIR", "index"), mode="semantic")
    record_query(q, "semantic", referenced_doc_ids=list({r["doc_id"] for r in results}))
    return {"query": q, "results": results}


@router.post("/")
def search_documents(req: SearchRequest):
    search_backend = _get_search_backend()
    results = search_backend(
        req.query,
        top_k=req.k,
        index_dir=os.getenv("INDEX_DIR", "index"),
        mode=req.mode,
        doc_ids=req.doc_ids,
    )
    record_query(req.query, req.mode, referenced_doc_ids=list({r["doc_id"] for r in results}))
    return {"query": req.query, "mode": req.mode, "results": results}


def build_context(results):
    blocks = []
    citations = []
    for result in results:
        citation = {
            "doc_id": result["doc_id"],
            "document": result.get("file_name") or result["doc_id"],
            "page_number": result.get("page_number"),
            "chunk_id": result.get("chunk_id"),
            "score": result.get("score"),
        }
        citations.append(citation)
        blocks.append(
            f"Source: {citation['document']} | page {citation['page_number']} | chunk {citation['chunk_id']}\n{result['text']}"
        )
    return "\n\n---\n\n".join(blocks), citations


def format_history(session_id: str | None):
    if not session_id:
        return ""
    rows = memory.get_conversation(session_id, limit=12)
    return "\n".join([f"{row['role']}: {row['message']}" for row in rows])


@router.post("/qa")
def rag_qa(req: QARequest):
    search_backend = _get_search_backend()
    results = search_backend(
        req.query,
        top_k=req.k,
        index_dir=os.getenv("INDEX_DIR", "index"),
        mode=req.mode,
        doc_ids=req.doc_ids,
    )
    if not results:
        answer = "I cannot determine the answer from the provided documents."
        if req.session_id:
            memory.append_message(req.session_id, "user", req.query)
            memory.append_message(req.session_id, "assistant", answer)
        record_query(req.query, req.mode, req.session_id, False, [])
        return {"answer": answer, "sources": [], "retrieved_context": [], "confidence": 0.0}

    context, sources = build_context(results)
    answer = None
    answer_generated = False
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                "You are an AI Research Assistant. Answer using only the retrieved document context. "
                "If the context is insufficient, say exactly: I cannot determine the answer from the provided documents.\n\n"
                f"Conversation history:\n{format_history(req.session_id)}\n\n"
                f"Retrieved context:\n{context}\n\n"
                f"Question: {req.query}\n\n"
                "Return a direct answer and cite document names, pages, and chunk IDs."
            )
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=700,
            )
            answer = resp.choices[0].message.content
            answer_generated = True
        except Exception as exc:
            answer = f"LLM error: {exc}"

    if answer is None:
        snippets = "\n\n".join(
            f"- {source['document']} page {source['page_number']}: {result['text'][:350]}"
            for source, result in zip(sources, results)
        )
        answer = (
            "LLM not configured; here are the most relevant grounded excerpts:\n"
            f"{snippets}"
        )

    if req.session_id:
        memory.append_message(req.session_id, "user", req.query)
        memory.append_message(req.session_id, "assistant", answer)
    referenced_doc_ids = list({source["doc_id"] for source in sources})
    record_query(req.query, req.mode, req.session_id, answer_generated, referenced_doc_ids)
    confidence = max([r["score"] for r in results]) if results else 0.0
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_context": results,
        "confidence": confidence,
    }


@router.get("/qa")
def rag_qa_get(q: str, k: int = 5, session_id: str | None = None):
    return rag_qa(QARequest(query=q, k=k, mode="semantic", session_id=session_id))
