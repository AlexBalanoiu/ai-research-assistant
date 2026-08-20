"""
Persistent report history backed by SQLite. Stores each
generated report's question, synthesis, and sources, so a future
question on a similar topic can reuse prior research.
"""
import json
import time

from research_assistant.memory.db import get_connection

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS report_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    synthesis TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""


def _ensure_table(conn) -> None:
    conn.execute(_TABLE_DDL)


def save_report(question: str, synthesis: str, sources: list[dict]) -> None:
    conn = get_connection()
    try:
        _ensure_table(conn)
        conn.execute(
            "INSERT INTO report_history (question, synthesis, sources_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (question, synthesis, json.dumps(sources), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def search_past_reports(keyword: str, limit: int = 3) -> list[dict]:
    """Simple keyword match over past questions/syntheses (SQL LIKE, case-insensitive)."""
    conn = get_connection()
    try:
        _ensure_table(conn)
        pattern = f"%{keyword.lower()}%"
        rows = conn.execute(
            "SELECT question, synthesis, created_at FROM report_history "
            "WHERE lower(question) LIKE ? OR lower(synthesis) LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()