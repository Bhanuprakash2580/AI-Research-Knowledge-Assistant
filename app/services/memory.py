from ..db import append_conversation, get_conversation as db_get_conversation, clear_conversation as db_clear_conversation


def append_message(session_id: str, role: str, message: str):
    return append_conversation(session_id, role, message)


def get_conversation(session_id: str, limit: int = 50):
    return db_get_conversation(session_id, limit=limit)


def clear_conversation(session_id: str):
    return db_clear_conversation(session_id)

