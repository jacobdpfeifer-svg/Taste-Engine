from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from app.config import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  person_id TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS choices (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  kind TEXT,
  chosen_id TEXT,
  rejected_id TEXT,
  domain TEXT,
  weight REAL DEFAULT 1.0,
  created_at TEXT
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.db_path)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)
        cols = {row[1] for row in c.execute("PRAGMA table_info(choices)")}
        if "weight" not in cols:
            c.execute("ALTER TABLE choices ADD COLUMN weight REAL DEFAULT 1.0")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(session_id: str, person_id: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (id, person_id, created_at) VALUES (?,?,?)",
            (session_id, person_id, _now()),
        )


def add_choice(
    session_id: str,
    chosen_id: str,
    rejected_id: str,
    domain: str,
    kind: str = "pair",
    weight: float = 1.0,
) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO choices (session_id, kind, chosen_id, rejected_id, domain, weight, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (session_id, kind, chosen_id, rejected_id, domain, weight, _now()),
        )


def get_session(session_id: str) -> sqlite3.Row | None:
    with _conn() as c:
        return c.execute(
            "SELECT id, person_id, created_at FROM sessions WHERE id=?", (session_id,)
        ).fetchone()


def get_choices(session_id: str) -> list[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT chosen_id, rejected_id, domain, weight FROM choices WHERE session_id=?", (session_id,)
        ).fetchall()
