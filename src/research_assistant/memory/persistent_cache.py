"""
Persistent (cross-session) search cache backed by SQLite.
Complements the in-session cache - this one
 survives process restarts, so a query searched yesterday doesn't
hit the network again today.
"""
import json
import time

from research_assistant.memory.db import get_connection

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS search_cache (
    query_key TEXT PRIMARY KEY,
    results_json TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""


def _normalize(query: str) -> str:
    return query.strip().lower()


def _ensure_table(conn) -> None:
    conn.execute(_TABLE_DDL)


def get_cached(query: str):
    conn = get_connection()
    try:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT results_json FROM search_cache WHERE query_key = ?",
            (_normalize(query),),
        ).fetchone()
        return json.loads(row["results_json"]) if row else None
    finally:
        conn.close()


def store(query: str, results: list[dict]) -> None:
    conn = get_connection()
    try:
        _ensure_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO search_cache (query_key, results_json, created_at) "
            "VALUES (?, ?, ?)",
            (_normalize(query), json.dumps(results), time.time()),
        )
        conn.commit()
    finally:
        conn.close()