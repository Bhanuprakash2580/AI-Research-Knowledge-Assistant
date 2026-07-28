from fastapi import APIRouter, HTTPException
from ..services import memory
from pydantic import BaseModel

router = APIRouter(prefix="/memory", tags=["memory"])


class MessageIn(BaseModel):
    session_id: str
    role: str
    message: str


@router.post("/append")
def append(msg: MessageIn):
    if msg.role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'assistant'")
    return memory.append_message(msg.session_id, msg.role, msg.message)


@router.get("/get/{session_id}")
def get(session_id: str):
    return memory.get_conversation(session_id)


@router.post("/clear/{session_id}")
def clear(session_id: str):
    return memory.clear_conversation(session_id)
