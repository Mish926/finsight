"""SQLite persistence for AgentLens.

Deliberately boring: two tables, stdlib sqlite3, WAL mode for concurrent
readers, a lock for writers. No ORM. The analysis layer reads straight
SQL because the queries it needs (per-agent rollups) are what SQL is for.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from .models import Span, Trace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,
    app             TEXT NOT NULL,
    name            TEXT,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    outcome_success INTEGER,          -- NULL = unknown, 0/1 otherwise
    outcome_meta    TEXT              -- JSON
);

CREATE TABLE IF NOT EXISTS spans (
    span_id           TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL REFERENCES traces(trace_id),
    agent             TEXT NOT NULL,
    task              TEXT,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    latency_ms        REAL NOT NULL,
    cost_usd          REAL NOT NULL,
    prompt_hash       TEXT,
    prompt_preview    TEXT,
    started_at        REAL NOT NULL,
    error             TEXT
);

CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_agent ON spans(agent);
"""


class Storage:
    def __init__(self, db_path: str = "agentlens.db"):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- writes ------------------------------------------------------------

    def insert_trace(self, trace: Trace) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO traces "
                "(trace_id, app, name, started_at, ended_at, outcome_success, outcome_meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.app,
                    trace.name,
                    trace.started_at,
                    trace.ended_at,
                    None if trace.outcome_success is None else int(trace.outcome_success),
                    json.dumps(trace.outcome_meta) if trace.outcome_meta else None,
                ),
            )
            self._conn.commit()

    def end_trace(self, trace_id: str, ended_at: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE traces SET ended_at = ? WHERE trace_id = ?", (ended_at, trace_id)
            )
            self._conn.commit()

    def record_outcome(
        self, trace_id: str, success: bool, meta: Optional[dict[str, Any]] = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE traces SET outcome_success = ?, outcome_meta = ? WHERE trace_id = ?",
                (int(success), json.dumps(meta) if meta else None, trace_id),
            )
            self._conn.commit()

    def insert_span(self, span: Span) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO spans "
                "(span_id, trace_id, agent, task, model, prompt_tokens, completion_tokens, "
                " latency_ms, cost_usd, prompt_hash, prompt_preview, started_at, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    span.span_id,
                    span.trace_id,
                    span.agent,
                    span.task,
                    span.model,
                    span.prompt_tokens,
                    span.completion_tokens,
                    span.latency_ms,
                    span.cost_usd,
                    span.prompt_hash,
                    span.prompt_preview,
                    span.started_at,
                    span.error,
                ),
            )
            self._conn.commit()

    # -- reads -------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
