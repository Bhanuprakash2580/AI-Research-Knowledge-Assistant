from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from ..services import summarizer
from ..classifier import predict_category
from ..db import get_document, update_document
from typing import List, Union

router = APIRouter(prefix="/analysis", tags=["analysis"])


class CompareRequest(BaseModel):
    doc_ids: List[str]
    focus: str = "general"


@router.get("/summarize/{doc_id}")
def summarize(doc_id: str, type: str = "executive"):
    res = summarizer.summarize_document(doc_id, summary_type=type)
    if res.get("summary") is None:
        raise HTTPException(status_code=404, detail=res.get("note", "No summary available"))
    return res


@router.post("/compare")
def compare(payload: Union[CompareRequest, List[str]] = Body(...)):
    if isinstance(payload, list):
        doc_ids = payload
        focus = "general"
    else:
        doc_ids = payload.doc_ids
        focus = payload.focus
    if not doc_ids or len(doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two document IDs to compare")
    res = summarizer.compare_documents(doc_ids, focus=focus)
    return res


@router.get("/classify/{doc_id}")
def classify(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = summarizer.load_chunks_for_doc(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="No processed text chunks found for document")
    prediction = predict_category("\n\n".join(chunks[:20]))
    update_document(
        doc_id,
        category=prediction.get("category"),
        classification_confidence=prediction.get("confidence"),
        classifier_note=prediction.get("note"),
    )
    return {"doc_id": doc_id, **prediction}
