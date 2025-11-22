import sqlite3
import pickle
import os
from typing import Optional

from .utils import now_iso

DB_PATH = os.getenv("EMB_CACHE_DB", "embeddings_cache.db")


class CacheManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                filename TEXT,
                hash TEXT,
                embedding BLOB,
                updated_at TEXT
            )
            """
        )
        self._conn.commit()

    def get(self, doc_id: str) -> Optional[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT filename, hash, embedding, updated_at FROM embeddings WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        if not row:
            return None
        filename, hashv, emb_blob, updated_at = row
        embedding = pickle.loads(emb_blob)
        return {"doc_id": doc_id, "filename": filename, "hash": hashv, "embedding": embedding, "updated_at": updated_at}

    def upsert(self, doc_id: str, filename: str, hashv: str, embedding) -> None:
        cur = self._conn.cursor()
        emb_blob = pickle.dumps(embedding)
        cur.execute(
            "INSERT OR REPLACE INTO embeddings (doc_id, filename, hash, embedding, updated_at) VALUES (?,?,?,?,?)",
            (doc_id, filename, hashv, emb_blob, now_iso()),
        )
        self._conn.commit()

    def all(self):
        cur = self._conn.cursor()
        cur.execute("SELECT doc_id, filename, hash, embedding, updated_at FROM embeddings")
        out = []
        for doc_id, filename, hashv, emb_blob, updated_at in cur.fetchall():
            out.append({
                "doc_id": doc_id,
                "filename": filename,
                "hash": hashv,
                "embedding": pickle.loads(emb_blob),
                "updated_at": updated_at,
            })
        return out

    def close(self):
        self._conn.close()