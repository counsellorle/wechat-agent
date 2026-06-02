
import sqlite3
import json
import threading
from datetime import datetime
from typing import Optional
from config import CHAT_HISTORY_DB


class ChatMemory:
    def __init__(self):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(CHAT_HISTORY_DB), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT    NOT NULL,
                    summary         TEXT    DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role            TEXT    NOT NULL,
                    content         TEXT    NOT NULL,
                    created_at      TEXT    NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id);
            """)
            self.conn.commit()

    def create_conversation(self, summary: str = "") -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO conversations (created_at, summary) VALUES (?, ?)",
                (datetime.now().isoformat(), summary),
            )
            self.conn.commit()
            return cur.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str):
        with self._lock:
            self.conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, datetime.now().isoformat()),
            )
            self.conn.commit()

    def get_conversation(self, conversation_id: int, limit: int = 10):
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            return list(reversed(rows))

    def get_recent_history(self, limit: int = 5):
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, summary, created_at FROM conversations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return rows

    def get_all_messages_for_retrain(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content FROM messages ORDER BY created_at"
            ).fetchall()
            return rows

    def update_summary(self, conversation_id: int, summary: str):
        with self._lock:
            self.conn.execute(
                "UPDATE conversations SET summary = ? WHERE id = ?",
                (summary, conversation_id),
            )
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()

