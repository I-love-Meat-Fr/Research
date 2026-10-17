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
    is_augmented INTEGER NOT NULL DEFAULT 0,
    aug_config   TEXT,
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
        # Run any pending migrations
        self._migrate_add_aug_columns()

    def _migrate_add_aug_columns(self):
        """Add is_augmented and aug_config columns if they don't exist (backward compat)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            # Check existing columns
            cur.execute("PRAGMA table_info(embeddings)")
            cols = [row[1] for row in cur.fetchall()]
            if "is_augmented" not in cols:
                cur.execute("ALTER TABLE embeddings ADD COLUMN is_augmented INTEGER NOT NULL DEFAULT 0")
            if "aug_config" not in cols:
                cur.execute("ALTER TABLE embeddings ADD COLUMN aug_config TEXT")
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
        is_augmented: int = 0,
        aug_config: str = "",
    ) -> bool:
        """
        Add one embedding blob for a user.
        User row must already exist.

        Args:
            user_id:       User identifier.
            embedding:      128-D float32 embedding array.
            sample_idx:    Index of this sample for the user.
            source_path:   Optional path/label for the source image.
            is_augmented:  1 if this is an augmented sample, 0 if real (default).
            aug_config:    JSON config string describing the augmentation applied,
                           or empty string for real samples.
        """
        emb_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
            if not cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO embeddings (user_id, embedding, sample_idx, source_path, created_at, is_augmented, aug_config) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    emb_bytes,
                    int(sample_idx),
                    source_path,
                    datetime.now().isoformat(timespec="seconds"),
                    int(is_augmented),
                    aug_config,
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

    def get_user_embeddings(self, user_id: str) -> list[tuple[int, np.ndarray, str, int, str]]:
        """
        Return all embeddings for a user.
        Returns: [(sample_idx, embedding_array, source_path, is_augmented, aug_config), ...]
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT sample_idx, embedding, source_path, is_augmented, aug_config "
                "FROM embeddings WHERE user_id = ? ORDER BY sample_idx",
                (user_id,),
            )
            results = []
            for sample_idx, emb_bytes, source_path, is_augmented, aug_config in cur.fetchall():
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                results.append((sample_idx, emb, source_path or "", int(is_augmented), aug_config or ""))
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

    # ------------------------------------------------------------------
    # Mahalanobis statistics — used by open-set identification
    # ------------------------------------------------------------------

    def get_user_stats(self, user_id: str) -> Optional[dict]:
        """
        Compute per-user statistics for Mahalanobis-distance open-set
        identification.

        Returns a dict:
            {
                "center":  np.ndarray   # L2-normalized mean embedding
                "cov":     np.ndarray   # sample covariance matrix [D, D]
                "cov_inv": np.ndarray   # regularized inverse of `cov` [D, D]
                "n":       int          # number of samples
            }

        Returns None if the user has no embeddings.

        Notes on the regularized inverse:
            With only 1-3 samples per user the sample covariance is
            rank-deficient and cannot be inverted. We add a small ridge
            term ``alpha * I`` to make it invertible while keeping the
            result close to the empirical covariance.

            IMPORTANT: alpha was raised from 1e-3 to 0.05 for the
            augmented enrollment pipeline. With 12 samples per user
            (1 real + 11 augmented) the covariance matrix has more
            eigenvalue spread. A larger ridge (alpha=0.05) prevents
            numerical instability and keeps the Mahalanobis distance
            well-behaved. Smaller values may cause instability; larger
            values over-shrink toward spherical (cosine-like).

        For the 1-sample case we fall back to cov = I (Mahalanobis ==
        Euclidean == cosine on unit vectors).
        """
        embs = self.get_user_embeddings(user_id)
        if not embs:
            return None
        arr = np.stack([e[1] for e in embs]).astype(np.float32)  # [N, D]
        n, d = arr.shape

        # Mean (center), re-normalized to unit length — same convention
        # as get_user_center so the two metrics stay comparable.
        center = arr.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)

        if n < 2:
            # Only 1 sample: covariance is undefined. Treat as identity
            # so Mahalanobis distance == Euclidean distance == (sqrt(2) - 2*cosine).
            cov = np.eye(d, dtype=np.float32)
            cov_inv = np.eye(d, dtype=np.float32)
        else:
            cov = np.cov(arr, rowvar=False).astype(np.float32)
            # Ridge regularization — alpha=0.05 for augmented enrollment (12 samples/user)
            alpha = 0.05
            cov_reg = cov + alpha * np.eye(d, dtype=np.float32)
            # Use pseudo-inverse to be safe even if ridge is insufficient
            cov_inv = np.linalg.pinv(cov_reg).astype(np.float32)

        return {
            "center": center,
            "cov": cov,
            "cov_inv": cov_inv,
            "n": n,
        }

    def get_all_user_stats(self) -> dict[str, dict]:
        """
        Compute per-user statistics (center + covariance) for every user
        in the database. Used by the Mahalanobis open-set identifier.

        Returns:
            {user_id: {"center": ..., "cov": ..., "cov_inv": ..., "n": ...}}

        Users with no embeddings are silently skipped.
        """
        users = self.list_users()
        stats: dict[str, dict] = {}
        for u in users:
            s = self.get_user_stats(u["user_id"])
            if s is not None:
                stats[u["user_id"]] = s
        return stats

    def get_all_embeddings_raw(self) -> dict[str, list[np.ndarray]]:
        """
        Return ALL raw embeddings for every user.
        Returns: {user_id: [embedding_1, embedding_2, ...]}

        Used by the re-ranking identification strategy: instead of comparing
        the query against a single center per user, we compare it against
        EVERY stored sample and aggregate. This handles the case where
        a user enrolled with diverse finger poses / lighting and the
        single "mean" embedding sits in an empty region of the space.
        """
        users = self.list_users()
        result: dict[str, list[np.ndarray]] = {}
        for u in users:
            embs = self.get_user_embeddings(u["user_id"])
            if embs:
                result[u["user_id"]] = [e[1] for e in embs]
        return result

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
