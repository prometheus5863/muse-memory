"""SQLite persistence. Plain sqlite3; vectors are float32 BLOBs."""
import os
import sqlite3
import threading

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,              -- "path#heading" or "chat:<request_id>"
    embedding BLOB NOT NULL,           -- float32, L2-normalised
    cluster INTEGER,                   -- community id, rewritten on recluster
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (source, text)              -- re-ingesting a file is idempotent
);
CREATE TABLE IF NOT EXISTS edges (
    src INTEGER NOT NULL REFERENCES nodes(id),
    dst INTEGER NOT NULL REFERENCES nodes(id),
    weight REAL NOT NULL,              -- cosine similarity
    PRIMARY KEY (src, dst), CHECK (src < dst)
);
CREATE TABLE IF NOT EXISTS calls (     -- one row per provider attempt
    id INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    outcome TEXT NOT NULL,             -- ok | rate_limited | server_error | timeout | ...
    detail TEXT,
    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    served_by TEXT,                    -- provider that finally answered the request, or NULL
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("MUSE_DB", "muse.db")
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()  # one writer at a time across FastAPI's threadpool

    def check_embedder(self, name: str) -> None:
        """Pin the DB to one embedding model: vectors from two models are not comparable."""
        row = self.db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        if row is None:
            with self.lock, self.db:
                self.db.execute("INSERT INTO meta VALUES ('embedder', ?)", (name,))
        elif row["value"] != name:
            raise RuntimeError(f"{self.path} was built with {row['value']}, not {name}. "
                               f"Set MUSE_EMBEDDER to match or use a fresh MUSE_DB.")

    def add_nodes(self, rows: list[tuple[str, str]], vecs: np.ndarray) -> list[int]:
        """Insert (text, source) rows; returns ids of rows actually inserted (dupes skipped)."""
        ids = []
        with self.lock, self.db:
            for (text, source), v in zip(rows, vecs):
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO nodes (text, source, embedding) VALUES (?, ?, ?)",
                    (text, source, v.astype(np.float32).tobytes()))
                if cur.rowcount:
                    ids.append(cur.lastrowid)
        return ids

    def load_nodes(self) -> tuple[list[dict], np.ndarray]:
        rows = self.db.execute("SELECT id, text, source, cluster, embedding FROM nodes ORDER BY id").fetchall()
        vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 0), np.float32)
        return [{k: r[k] for k in ("id", "text", "source", "cluster")} for r in rows], vecs

    def load_edges(self) -> list[tuple[int, int, float]]:
        return [tuple(r) for r in self.db.execute("SELECT src, dst, weight FROM edges")]

    def add_edges(self, edges: list[tuple[int, int, float]]) -> None:
        with self.lock, self.db:
            self.db.executemany("INSERT OR REPLACE INTO edges VALUES (?, ?, ?)",
                                [(min(a, b), max(a, b), w) for a, b, w in edges])

    def replace_edges(self, edges: list[tuple[int, int, float]]) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM edges")
        self.add_edges(edges)

    def set_clusters(self, assignment: dict[int, int]) -> None:
        with self.lock, self.db:
            self.db.executemany("UPDATE nodes SET cluster=? WHERE id=?", [(c, n) for n, c in assignment.items()])

    def log_calls(self, rows: list[dict]) -> None:
        cols = ("request_id", "provider", "attempt", "outcome", "detail", "latency_ms",
                "prompt_tokens", "completion_tokens", "served_by", "created_at")
        with self.lock, self.db:
            self.db.executemany(f"INSERT INTO calls ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                [tuple(r.get(c) for c in cols) for r in rows])

    def recent_calls(self, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,))]
