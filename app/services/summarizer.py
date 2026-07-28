import os
from pathlib import Path
import json
from typing import List
from ..db import get_document

def load_chunks_for_doc(doc_id: str, index_dir: str = None, as_dict: bool = False):
    index_dir = index_dir or os.getenv("INDEX_DIR", "index")
    reg = Path(index_dir) / "registry.json"
    if not reg.exists():
        return []
    with open(reg, "r", encoding="utf-8") as f:
        r = json.load(f)
    if doc_id not in r:
        return []
    info = r[doc_id]
    with open(info["chunks_file"], "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if as_dict:
        return chunks
    return [c["text"] for c in chunks]


def structured_extract_summary(texts: List[str]):
    context = "\n\n".join(texts[:10])
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 40]
    bullets = sentences[:5] or [context[:500]]
    return {
        "executive_summary": " ".join(sentences[:3]) if sentences else context[:800],
        "technical_summary": " ".join(sentences[:7]) if sentences else context[:1200],
        "bullet_point_summary": [f"{item}." for item in bullets],
        "key_takeaways": [f"{item}." for item in bullets[:3]],
    }


def summarize_texts(texts: List[str], summary_type: str = "executive"):
    # join top N texts
    context = "\n\n".join(texts[:8])
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                "Summarize the provided document excerpts using only the excerpts. "
                "Include these labeled sections: Executive Summary, Technical Summary, "
                "Bullet Point Summary, Key Takeaways.\n\n"
                f"Requested summary type: {summary_type}\n\n{context}"
            )
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=900,
            )
            ans = resp.choices[0].message.content
            return {"summary": ans, "source_count": len(texts)}
        except Exception as e:
            return {"summary": structured_extract_summary(texts), "note": f"LLM error: {e}", "source_count": len(texts)}
    return {
        "summary": structured_extract_summary(texts),
        "note": "LLM not configured; returning extractive structured summary",
        "source_count": len(texts),
    }


def summarize_document(doc_id: str, summary_type: str = "executive"):
    texts = load_chunks_for_doc(doc_id)
    if not texts:
        return {"summary": None, "note": "No chunks found for document"}
    result = summarize_texts(texts, summary_type=summary_type)
    result["document"] = get_document(doc_id)
    return result


def compare_documents(doc_ids: List[str], focus: str = "compare"):
    # gather top chunks per document
    docs_context = {}
    for did in doc_ids:
        docs_context[did] = load_chunks_for_doc(did)[:6]

    context_str = "\n\n".join([f"--- Document {d} ---\n" + "\n\n".join(docs_context[d]) for d in doc_ids])
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = f"Compare the following documents with focus on: {focus}. Use only the provided excerpts and cite document IDs.\n\n{context_str}\n\nProvide a structured comparison: Methodologies, Similarities, Differences, Advantages, Disadvantages, Conclusions."
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
            )
            ans = resp.choices[0].message.content
            return {"comparison": ans, "docs": doc_ids}
        except Exception as e:
            return {"comparison": fallback_comparison(docs_context, focus), "note": f"LLM error: {e}", "docs": doc_ids}
    return {
        "comparison": fallback_comparison(docs_context, focus),
        "note": "LLM not configured; returning extractive comparison",
        "docs": doc_ids,
    }


def fallback_comparison(docs_context, focus: str):
    return {
        "focus": focus,
        "methodologies": {doc_id: chunks[:2] for doc_id, chunks in docs_context.items()},
        "similarities": "Configure OPENAI_API_KEY for synthesized similarity analysis; excerpts are grouped per document below.",
        "differences": "Configure OPENAI_API_KEY for synthesized difference analysis; excerpts are grouped per document below.",
        "advantages_disadvantages": "Requires LLM synthesis for reliable advantages/disadvantages.",
        "source_excerpts": docs_context,
    }
