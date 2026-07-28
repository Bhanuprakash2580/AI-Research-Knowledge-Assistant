import os
import sqlite3
import tempfile
import uuid
from pathlib import Path


def _resolve_db_path() -> str:
    if os.getenv("DATABASE_FILE"):
        return os.getenv("DATABASE_FILE")
    if os.getenv("VERCEL"):
        return os.path.join(tempfile.gettempdir(), "data.db")
    target = Path("data.db")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        test_file = target.parent / ".write_test"
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        return str(target)
    except Exception:
        return os.path.join(tempfile.gettempdir(), "data.db")


DATABASE_PATH = _resolve_db_path()


def get_connection():
    target_dir = Path(DATABASE_PATH).parent
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn



def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            upload_ts TEXT DEFAULT CURRENT_TIMESTAMP,
            total_pages INTEGER DEFAULT 0,
            total_chunks INTEGER DEFAULT 0,
            status TEXT DEFAULT 'uploaded',
            category TEXT,
            classification_confidence REAL,
            classifier_note TEXT,
            file_path TEXT
        )
        """
    )
    existing_columns = {
        row["name"]
        for row in cursor.execute("PRAGMA table_info(documents)").fetchall()
    }
    migrations = {
        "category": "ALTER TABLE documents ADD COLUMN category TEXT",
        "classification_confidence": "ALTER TABLE documents ADD COLUMN classification_confidence REAL",
        "classifier_note": "ALTER TABLE documents ADD COLUMN classifier_note TEXT",
        "file_path": "ALTER TABLE documents ADD COLUMN file_path TEXT",
    }
    for column, statement in migrations.items():
        if column not in existing_columns:
            cursor.execute(statement)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS query_events (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            mode TEXT NOT NULL,
            session_id TEXT,
            answer_generated INTEGER DEFAULT 0,
            referenced_doc_ids TEXT DEFAULT '[]',
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def row_to_dict(row):
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def create_document(name: str, doc_id: str = None, file_path: str = None):
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (id, name, file_path) VALUES (?, ?, ?)",
        (doc_id, name, file_path),
    )
    conn.commit()
    conn.close()
    return get_document(doc_id)


def get_document(doc_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    return row_to_dict(row)


def list_documents():
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM documents ORDER BY upload_ts DESC").fetchall()
    conn.close()
    return [row_to_dict(row) for row in rows]


def delete_document(doc_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def update_document(doc_id: str, **fields):
    if not fields:
        return get_document(doc_id)
    columns = []
    values = []
    for key, value in fields.items():
        if value is not None:
            columns.append(f"{key} = ?")
            values.append(value)
    if not columns:
        return get_document(doc_id)
    values.append(doc_id)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE documents SET {', '.join(columns)} WHERE id = ?",
        tuple(values),
    )
    conn.commit()
    conn.close()
    return get_document(doc_id)


def count_documents(status: str = None):
    conn = get_connection()
    cursor = conn.cursor()
    if status is None:
        row = cursor.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
    else:
        row = cursor.execute("SELECT COUNT(*) AS count FROM documents WHERE status = ?", (status,)).fetchone()
    conn.close()
    return row["count"] if row else 0


def append_conversation(session_id: str, role: str, message: str, msg_id: str = None):
    if msg_id is None:
        msg_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (id, session_id, role, message) VALUES (?, ?, ?, ?)",
        (msg_id, session_id, role, message),
    )
    conn.commit()
    conn.close()
    return {"id": msg_id}


def get_conversation(session_id: str, limit: int = 50):
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT role, message, ts FROM conversations WHERE session_id = ? ORDER BY ts ASC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [row_to_dict(row) for row in rows]


def clear_conversation(session_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    cleared = cursor.rowcount
    conn.commit()
    conn.close()
    return {"cleared": bool(cleared)}


def record_query(query: str, mode: str, session_id: str = None, answer_generated: bool = False, referenced_doc_ids=None):
    msg_id = str(uuid.uuid4())
    if referenced_doc_ids is None:
        referenced_doc_ids = []
    import json

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO query_events (id, query, mode, session_id, answer_generated, referenced_doc_ids)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (msg_id, query, mode, session_id, int(answer_generated), json.dumps(referenced_doc_ids)),
    )
    conn.commit()
    conn.close()
    return {"id": msg_id}


def query_stats():
    conn = get_connection()
    cursor = conn.cursor()
    total_questions = cursor.execute("SELECT COUNT(*) AS count FROM query_events").fetchone()["count"]
    answered = cursor.execute(
        "SELECT COUNT(*) AS count FROM query_events WHERE answer_generated = 1"
    ).fetchone()["count"]
    rows = cursor.execute("SELECT referenced_doc_ids FROM query_events").fetchall()
    conn.close()

    import json
    from collections import Counter

    counter = Counter()
    for row in rows:
        try:
            for doc_id in json.loads(row["referenced_doc_ids"] or "[]"):
                counter[doc_id] += 1
        except json.JSONDecodeError:
            continue
    return {
        "total_questions_answered": answered,
        "total_queries": total_questions,
        "most_queried_documents": [
            {"doc_id": doc_id, "count": count}
            for doc_id, count in counter.most_common(5)
        ],
    }
