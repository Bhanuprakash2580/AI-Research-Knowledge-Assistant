from pydantic import BaseModel
from typing import Optional


class DocumentCreate(BaseModel):
    name: str


class DocumentRead(BaseModel):
    id: str
    name: str
    upload_ts: Optional[str]
    total_pages: int
    total_chunks: int
    status: str
    category: Optional[str] = None
    classification_confidence: Optional[float] = None
    classifier_note: Optional[str] = None
    file_path: Optional[str] = None


class ConversationRead(BaseModel):
    role: str
    message: str
    ts: str
