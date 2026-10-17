"""
Enrollment Database for Finger Vein Recognition
================================================
Lightweight SQLite-based store for enrolled user embeddings.

Tables:
    users:        (user_id TEXT PRIMARY KEY, name, created_at, num_samples)
    embeddings:   (id INTEGER PRIMARY KEY, user_id, embedding BLOB, sample_idx, source_path, created_at)
"""
import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict


# ============================================================================
# SCHEMA
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
    """SQLite-backed enrollment database for finger vein embeddings."""

    def __init__(self, db_path: str = "results/enrollment.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    # ----------------------------------------------------------------
    # Users
    # ----------------------------------------------------------------

    def add_user(self, user_id: str, name: str = "") -> bool:
        """Add a new user. Returns False if user_id already exists."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
            if cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO users (user_id, name, created_at, num_samples) VALUES (?, ?, ?, 0)",
                (user_id, name, datetime.now().isoformat(timespec='seconds')),
            )
            conn.commit()
            return True

    def delete_user(self, user_id: str) -> bool:
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_users(self) -> List[Dict]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT user_id, name, created_at, num_samples FROM users ORDER BY created_at DESC"
            )
            return [dict(row) for row in cur.fetchall()]

    def get_user(self, user_id: str) -> Optional[Dict]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT user_id, name, created_at, num_samples FROM users WHERE user_id = ?",
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ----------------------------------------------------------------
    # Embeddings
    # ----------------------------------------------------------------

    def add_embedding(
        self,
        user_id: str,
        embedding: np.ndarray,
        sample_idx: int,
        source_path: str = "",
    ) -> bool:
        """Add one embedding for a user (user must exist)."""
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
                    datetime.now().isoformat(timespec='seconds'),
                ),
            )
            # Update num_samples
            cur.execute(
                "UPDATE users SET num_samples = (SELECT COUNT(*) FROM embeddings WHERE user_id = ?) "
                "WHERE user_id = ?",
                (user_id, user_id),
            )
            conn.commit()
            return True

    def get_user_embeddings(self, user_id: str) -> List[Tuple[int, np.ndarray, str]]:
        """Return list of (sample_idx, embedding, source_path) for user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT sample_idx, embedding, source_path FROM embeddings WHERE user_id = ? ORDER BY sample_idx",
                (user_id,),
            )
            results = []
            for sample_idx, emb_bytes, source_path in cur.fetchall():
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                results.append((sample_idx, emb, source_path or ""))
            return results

    def get_all_users_with_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Return dict of {user_id: [embedding1, embedding2, ...]}."""
        users = self.list_users()
        result = {}
        for u in users:
            embs = [e[1] for e in self.get_user_embeddings(u['user_id'])]
            result[u['user_id']] = embs
        return result

    def count_total_embeddings(self) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM embeddings")
            return cur.fetchone()[0]

    def clear_all(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM embeddings")
            cur.execute("DELETE FROM users")
            conn.commit()

    # ----------------------------------------------------------------
    # Persistence helpers
    # ----------------------------------------------------------------

    def db_info(self) -> Dict:
        return {
            'path': str(self.db_path),
            'exists': self.db_path.exists(),
            'size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0,
            'num_users': len(self.list_users()),
            'num_embeddings': self.count_total_embeddings(),
        }


if __name__ == "__main__":
    # Quick smoke test
    import tempfile
    import gc

    tmp_dir = tempfile.mkdtemp()
    try:
        db = EnrollmentDB(Path(tmp_dir) / "test.db")
        db.add_user("u001", "Alice")
        e1 = np.random.randn(128).astype(np.float32)
        e2 = np.random.randn(128).astype(np.float32)
        db.add_embedding("u001", e1, sample_idx=0, source_path="a.png")
        db.add_embedding("u001", e2, sample_idx=1, source_path="b.png")
        db.add_user("u002", "Bob")
        db.add_embedding("u002", np.random.randn(128).astype(np.float32), 0)

        print("Users:", db.list_users())
        print("u001 embeddings:", len(db.get_user_embeddings("u001")))
        print("Info:", db.db_info())
        print("PASS")
    finally:
        gc.collect()
        import shutil
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass
