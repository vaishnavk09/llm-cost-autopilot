from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.settings import abs_path, get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt TEXT NOT NULL,
    use_case TEXT NOT NULL,
    complexity_tier INTEGER NOT NULL,
    routed_model TEXT NOT NULL,
    provider TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost REAL NOT NULL,
    baseline_cost REAL NOT NULL,
    gpt4o_cost REAL NOT NULL,
    quality_score REAL,
    escalated INTEGER NOT NULL DEFAULT 0,
    original_model TEXT,
    escalated_model TEXT,
    cost_delta REAL,
    quality_gap REAL,
    verify_status TEXT NOT NULL DEFAULT 'pending',
    response_text TEXT,
    judge_text TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    request_id INTEGER,
    event_type TEXT NOT NULL,
    payload_json TEXT,
    FOREIGN KEY(request_id) REFERENCES requests(id)
);
"""


def db_path() -> Path:
    path = abs_path(get_settings().database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def insert_request(row: dict[str, Any]) -> int:
    init_db()
    cols = [
        "created_at",
        "prompt_hash",
        "prompt",
        "use_case",
        "complexity_tier",
        "routed_model",
        "provider",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "cost",
        "baseline_cost",
        "gpt4o_cost",
        "quality_score",
        "escalated",
        "original_model",
        "escalated_model",
        "cost_delta",
        "quality_gap",
        "verify_status",
        "response_text",
        "judge_text",
        "metadata_json",
    ]
    values = [row.get(c) for c in cols]
    placeholders = ",".join("?" for _ in cols)
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO requests ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )
        return int(cur.lastrowid)


def update_request(request_id: int, **fields: Any) -> None:
    if not fields:
        return
    assignments = ",".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE requests SET {assignments} WHERE id=?", [*fields.values(), request_id])


def add_event(request_id: int | None, event_type: str, payload: dict[str, Any] | None = None) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "INSERT INTO events (created_at, request_id, event_type, payload_json) VALUES (?,?,?,?)",
            [
                datetime.now(timezone.utc).isoformat(),
                request_id,
                event_type,
                json.dumps(payload or {}),
            ],
        )


def pending_verifications(limit: int = 10) -> list[sqlite3.Row]:
    init_db()
    claimed: list[sqlite3.Row] = []
    with connect() as conn:
        cur = conn.execute(
            "SELECT * FROM requests WHERE verify_status='pending' ORDER BY id ASC LIMIT ?",
            [limit],
        )
        rows = list(cur.fetchall())
        for row in rows:
            cur = conn.execute(
                "UPDATE requests SET verify_status='verifying' WHERE id=? AND verify_status='pending'",
                [row["id"]],
            )
            if cur.rowcount:
                claimed.append(row)
    return claimed


def fetch_stats() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"]
        sums = conn.execute(
            """
            SELECT
                COALESCE(SUM(cost), 0) AS routed_cost,
                COALESCE(SUM(baseline_cost), 0) AS baseline_cost,
                COALESCE(SUM(gpt4o_cost), 0) AS gpt4o_cost,
                COALESCE(AVG(quality_score), 0) AS avg_quality,
                COALESCE(AVG(latency_ms), 0) AS avg_latency,
                COALESCE(SUM(escalated), 0) AS escalations
            FROM requests
            """
        ).fetchone()
        dist = conn.execute(
            "SELECT routed_model, COUNT(*) AS n, SUM(cost) AS cost FROM requests GROUP BY routed_model"
        ).fetchall()
        daily = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   SUM(cost) AS routed_cost,
                   SUM(baseline_cost) AS baseline_cost,
                   COUNT(*) AS n,
                   SUM(escalated) AS escalations
            FROM requests
            GROUP BY day
            ORDER BY day
            """
        ).fetchall()
        quality = conn.execute(
            "SELECT quality_score FROM requests WHERE quality_score IS NOT NULL"
        ).fetchall()

    routed = float(sums["routed_cost"])
    baseline = float(sums["baseline_cost"])
    gpt4o = float(sums["gpt4o_cost"])
    saved_vs_premium = max(0.0, baseline - routed)
    pct = (saved_vs_premium / baseline * 100.0) if baseline > 0 else 0.0
    saved_vs_gpt4o = max(0.0, gpt4o - routed)
    pct_gpt4o = (saved_vs_gpt4o / gpt4o * 100.0) if gpt4o > 0 else 0.0
    return {
        "total_requests": int(total),
        "routed_cost": routed,
        "baseline_cost": baseline,
        "gpt4o_cost": gpt4o,
        "saved_vs_premium": saved_vs_premium,
        "cost_reduction_pct": pct,
        "saved_vs_gpt4o": saved_vs_gpt4o,
        "cost_reduction_vs_gpt4o_pct": pct_gpt4o,
        "avg_quality": float(sums["avg_quality"]),
        "avg_latency_ms": float(sums["avg_latency"]),
        "escalations": int(sums["escalations"]),
        "escalation_rate": (int(sums["escalations"]) / total) if total else 0.0,
        "routing_distribution": [
            {"model": r["routed_model"], "count": r["n"], "cost": r["cost"]} for r in dist
        ],
        "daily": [dict(r) for r in daily],
        "quality_scores": [r["quality_score"] for r in quality],
    }


def fetch_recent(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY id DESC LIMIT ?", [limit]
        ).fetchall()
    return [dict(r) for r in rows]
