"""
database.py
------------
Lightweight SQLite persistence layer for recognition events, used to
power the analytics dashboard. Uses only the standard library so it
has zero extra dependencies and cannot fail due to missing packages.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "recognition_log.db"


class RecognitionDB:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    letter TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT 'camera'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)"
            )

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    def log_event(self, session_id: str, letter: str, confidence: float,
                  source: str = "camera") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (session_id, timestamp, letter, confidence, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, datetime.utcnow().isoformat(), letter, float(confidence), source),
            )

    def fetch_all(self) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query("SELECT * FROM events ORDER BY timestamp", conn)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def fetch_session(self, session_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp",
                conn, params=(session_id,),
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def clear_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM events")

    def session_ids(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM events "
                "GROUP BY session_id ORDER BY MAX(timestamp) DESC"
            ).fetchall()
        return [r[0] for r in rows]
