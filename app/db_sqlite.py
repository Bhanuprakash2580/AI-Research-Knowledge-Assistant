import sqlite3
import os
from pathlib import Path

DATABASE_PATH = os.getenv("DATABASE_FILE", "data.db")
Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_connection():
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
            status TEXT DEFAULT 'uploaded'
        )
        """
    )
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
    conn.commit()
    conn.close()
