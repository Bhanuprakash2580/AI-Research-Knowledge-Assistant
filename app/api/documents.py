from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pathlib import Path
from ..db import (
    create_document,
    get_document,
    list_documents as db_list_documents,
    delete_document as db_delete_document,
    update_document,
)
from ..models import DocumentRead
from ..services import processor
import shutil
import os
from typing import List

router = APIRouter(prefix="/documents", tags=["documents"])

STORAGE = Path(os.getenv("STORAGE_DIR", "storage"))
STORAGE.mkdir(parents=True, exist_ok=True)


def storage_path(doc_id: str, filename: str) -> Path:
    return STORAGE / f"{doc_id}_{filename}"


@router.post("/upload", response_model=DocumentRead)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in this prototype")
    doc = create_document(file.filename)
    doc_id = doc["id"]
    dest = storage_path(doc_id, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    doc = update_document(doc_id, file_path=str(dest), status="uploaded")
    if background_tasks is not None:
        background_tasks.add_task(processor.process_document, doc_id, str(dest))
    else:
        processor.process_document(doc_id, str(dest))
    return doc


@router.get("/", response_model=List[DocumentRead])
def list_documents():
    return db_list_documents()


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    deleted = db_delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    file_path = storage_path(doc_id, doc["name"])
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass
    idx = Path(os.getenv("INDEX_DIR", "index"))
    reg = idx / "registry.json"
    if reg.exists():
        import json
        with open(reg, "r", encoding="utf-8") as f:
            r = json.load(f)
        if doc_id in r:
            info = r.pop(doc_id)
            for v in info.values():
                try:
                    Path(v).unlink()
                except Exception:
                    pass
            with open(reg, "w", encoding="utf-8") as f:
                json.dump(r, f)
    return {"status": "deleted"}


@router.post("/{doc_id}/reprocess")
def reprocess_document(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    file_path = storage_path(doc_id, doc["name"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Original file missing")
    processor.process_document(doc_id, str(file_path))
    return {"status": "reprocessed"}
