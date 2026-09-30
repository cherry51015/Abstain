"""
Content-addressed LLM response cache (SQLite, stdlib only).

Keyed by sha256(model, messages, temperature, sample_index). Two uses:
  - eval runs are reproducible and resumable: a run interrupted by a rate
    limit picks up where it stopped without re-spending tokens
  - re-evaluating an unchanged dispute replays the same samples, so the
    decision is reproducible for audit
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class ResponseCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
            self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, response: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO llm_cache (key, response) VALUES (?, ?)", (key, response))
            self._conn.commit()
