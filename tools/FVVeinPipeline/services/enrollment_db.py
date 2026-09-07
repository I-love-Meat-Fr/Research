"""
Enrollment Database for Finger Vein Recognition
================================================
SQLite-backed store for enrolled user embeddings.
No IplImage, no proprietary structs — pure NumPy/SQLite.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================================
# Schema
# ============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    name        TEXT,
    created_at  TEXT NOT NULL,
    num_samples INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embeddings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    sample_idx   INTEGER NOT NULL,
    source_path  TEXT,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_user ON embeddings(user_id);
"""


class EnrollmentDB:
    """
    Lightweight SQLite enrollment database.

    Stores raw float32 embedding blobs. Center embeddings are computed
    at runtime from stored per-sample embeddings.
    """

    def __init__(self, db_path: str = "results/enrollment.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def add_user(self, user_id: str, name: str = "") -> bool:
        """
        Create a new user row. Returns False if user_id already exists.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
            if cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO users (user_id, name, created_at, num_samples) "
                "VALUES (?, ?, ?, 0)",
                (user_id, name, datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            return True

    def upsert_user(self, user_id: str, name: str = "") -> bool:
        """
        Create or update user row. Always succeeds.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
            )
            exists = cur.fetchone() is not None
            if exists:
                cur.execute(
                    "UPDATE users SET name = ?, created_at = ? WHERE user_id = ?",
                    (name, datetime.now().isoformat(timespec="seconds"), user_id),
                )
            else:
                cur.execute(
                    "INSERT INTO users (user_id, name, created_at, num_samples) "
                    "VALUES (?, ?, ?, 0)",
                    (user_id, name, datetime.now().isoformat(timespec="seconds")),
                )
            conn.commit()
            return True

    def delete_user(self, user_id: str) -> bool:
        """Delete user and all their embeddings."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_users(self) -> list[dict]:
        """Return all users ordered by creation time (newest first)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT user_id, name, created_at, num_samples "
                "FROM users ORDER BY created_at DESC"
            )
            return [dict(row) for row in cur.fetchall()]

    def get_user(self, user_id: str) -> Optional[dict]:
        """Get a single user by ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT user_id, name, created_at, num_samples "
                "FROM users WHERE user_id = ?", (user_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def add_embedding(
        self,
        user_id: str,
        embedding: np.ndarray,
        sample_idx: int,
        source_path: str = "",
    ) -> bool:
        """
        Add one embedding blob for a user.
        User row must already exist.
        """
        emb_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
            if not cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO embeddings (user_id, embedding, sample_idx, source_path, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    emb_bytes,
                    int(sample_idx),
                    source_path,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            # Sync num_samples
            cur.execute(
                "UPDATE users SET num_samples = ("
                "  SELECT COUNT(*) FROM embeddings WHERE user_id = ?"
                ") WHERE user_id = ?",
                (user_id, user_id),
            )
            conn.commit()
            return True

    def get_user_embeddings(self, user_id: str) -> list[tuple[int, np.ndarray, str]]:
        """
        Return all embeddings for a user.
        Returns: [(sample_idx, embedding_array, source_path), ...]
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT sample_idx, embedding, source_path "
                "FROM embeddings WHERE user_id = ? ORDER BY sample_idx",
                (user_id,),
            )
            results = []
            for sample_idx, emb_bytes, source_path in cur.fetchall():
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                results.append((sample_idx, emb, source_path or ""))
            return results

    def get_user_center(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get the center embedding (mean → re-normalized) for a user.
        Returns None if user has no embeddings.
        """
        embs = self.get_user_embeddings(user_id)
        if not embs:
            return None
        arr = np.stack([e[1] for e in embs])
        center = arr.mean(axis=0)
        norm = np.linalg.norm(center)
        if norm > 0:
            center = center / norm
        return center

    def get_all_centers(self) -> dict[str, np.ndarray]:
        """
        Compute center embedding for every user in the DB.
        Returns: {user_id: center_embedding}
        """
        users = self.list_users()
        centers = {}
        for u in users:
            center = self.get_user_center(u["user_id"])
            if center is not None:
                centers[u["user_id"]] = center
        return centers

    def get_user_embedding_count(self, user_id: str) -> int:
        """Return number of embeddings stored for a user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM embeddings WHERE user_id = ?", (user_id,)
            )
            return cur.fetchone()[0]

    def count_total_embeddings(self) -> int:
        """Total number of embedding rows across all users."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM embeddings")
            return cur.fetchone()[0]

    def clear_all(self):
        """Delete all users and embeddings."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM embeddings")
            cur.execute("DELETE FROM users")
            conn.commit()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def db_info(self) -> dict:
        """Return database metadata."""
        return {
            "path": str(self.db_path),
            "exists": self.db_path.exists(),
            "size_bytes": (
                self.db_path.stat().st_size
                if self.db_path.exists()
                else 0
            ),
            "num_users": len(self.list_users()),
            "num_embeddings": self.count_total_embeddings(),
        }
