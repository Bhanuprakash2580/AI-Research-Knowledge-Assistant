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
import shutil
import os
from typing import List

import tempfile

router = APIRouter(prefix="/documents", tags=["documents"])


def _get_storage_dir() -> Path:
    env_dir = os.getenv("STORAGE_DIR")
    if env_dir:
        p = Path(env_dir)
    elif os.getenv("VERCEL"):
        p = Path(tempfile.gettempdir()) / "storage"
    else:
        p = Path("storage")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        p = Path(tempfile.gettempdir()) / "storage"
        p.mkdir(parents=True, exist_ok=True)
    return p


def storage_path(doc_id: str, filename: str) -> Path:
    return _get_storage_dir() / f"{doc_id}_{filename}"



def validate_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in this prototype")


def _get_processor_backend():
    try:
        from ..services import processor as processor_backend
    except Exception as exc:
        raise HTTPException(status_code=501, detail="Document processing backend unavailable") from exc
    return processor_backend


async def save_and_process(file: UploadFile, background_tasks: BackgroundTasks = None):
    validate_pdf(file)
    doc = create_document(file.filename)
    doc_id = doc["id"]
    dest = storage_path(doc_id, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    doc = update_document(doc_id, file_path=str(dest), status="uploaded")
    processor_backend = _get_processor_backend()
    if background_tasks is not None:
        background_tasks.add_task(processor_backend.process_document, doc_id, str(dest))
    else:
        processor_backend.process_document(doc_id, str(dest))
    return doc


@router.post("/upload", response_model=DocumentRead)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    return await save_and_process(file, background_tasks)


@router.post("/upload-batch", response_model=List[DocumentRead])
async def upload_documents(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required")
    return [await save_and_process(file, background_tasks) for file in files]


@router.get("/", response_model=List[DocumentRead])
def list_documents():
    return db_list_documents()


@router.get("/{doc_id}", response_model=DocumentRead)
def get_document_detail(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


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
    processor_backend = _get_processor_backend()
    processor_backend.process_document(doc_id, str(file_path))
    return {"status": "reprocessed"}
