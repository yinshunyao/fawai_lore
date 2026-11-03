"""Lightweight note vectors via sqlite-vec (optional dependency)."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

EmbedFn = Callable[[str], list[float]]


def sqlite_vec_available() -> bool:
    try:
        import sqlite_vec  # noqa: F401

        return True
    except ImportError:
        return False


def serialize_f32(vector: list[float]) -> bytes:
    try:
        from sqlite_vec import serialize_float32

        return serialize_float32(vector)
    except Exception:
        return struct.pack(f"{len(vector)}f", *vector)


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:40]


def fake_embed(text: str, *, dim: int = 64) -> list[float]:
    """
    Deterministic bag-of-token hashing embed for demos (no model).
    Overlapping tokens → closer vectors (good enough for smoke tests).
    """
    vals = [0.0] * dim
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", (text or "").lower())
    if not tokens:
        tokens = ["empty"]
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vals[h % dim] += 1.0
        vals[(h // dim) % dim] += 0.5
    norm = sum(v * v for v in vals) ** 0.5 or 1.0
    return [v / norm for v in vals]


class VectorIndex:
    """
    Stores note embeddings in ``{root}/vectors.sqlite`` using vec0 + cosine.

    Layout::

        meta(key, value)
        note_map(rowid INTEGER PK, note_id TEXT UNIQUE, text_hash TEXT)
        vec_notes(embedding float[dim])  -- vec0, rowid aligned with note_map
    """

    def __init__(self, root: str | Path) -> None:
        if not sqlite_vec_available():
            raise RuntimeError(
                "sqlite-vec not installed; pip install 'fawai_lore[vector]' or sqlite-vec"
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "vectors.sqlite"
        self._dim: int | None = None
        self._conn = sqlite3.connect(str(self.path))
        self._conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS note_map (
              rowid INTEGER PRIMARY KEY,
              note_id TEXT NOT NULL UNIQUE,
              text_hash TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'dim'"
        ).fetchone()
        if row:
            self._dim = int(row[0])
            self._ensure_vec_table(self._dim)

    def close(self) -> None:
        self._conn.close()

    def _ensure_vec_table(self, dim: int) -> None:
        # vec0 cannot ALTER dim; create once
        exists = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_notes'"
        ).fetchone()
        if exists:
            return
        self._conn.execute(
            f"CREATE VIRTUAL TABLE vec_notes USING vec0("
            f"embedding float[{int(dim)}] distance_metric=cosine"
            f")"
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('dim', ?)",
            (str(dim),),
        )
        self._conn.commit()
        self._dim = dim

    def upsert(
        self,
        note_id: str,
        text: str,
        embedding: list[float],
    ) -> None:
        nid = str(note_id or "").strip()
        if not nid:
            raise ValueError("note_id required")
        if not embedding:
            raise ValueError("embedding required")
        dim = len(embedding)
        if self._dim is None:
            self._ensure_vec_table(dim)
        elif self._dim != dim:
            raise ValueError(
                f"embedding dim {dim} != index dim {self._dim}; rebuild vectors.sqlite"
            )
        assert self._dim is not None
        blob = serialize_f32([float(x) for x in embedding])
        th = _hash_text(text)
        existing = self._conn.execute(
            "SELECT rowid, text_hash FROM note_map WHERE note_id = ?",
            (nid,),
        ).fetchone()
        if existing and existing[1] == th:
            return  # unchanged
        if existing:
            rid = int(existing[0])
            self._conn.execute("DELETE FROM vec_notes WHERE rowid = ?", (rid,))
            self._conn.execute(
                "UPDATE note_map SET text_hash = ? WHERE rowid = ?",
                (th, rid),
            )
            self._conn.execute(
                "INSERT INTO vec_notes(rowid, embedding) VALUES (?, ?)",
                (rid, blob),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO note_map(note_id, text_hash) VALUES (?, ?)",
                (nid, th),
            )
            rid = int(cur.lastrowid)
            self._conn.execute(
                "INSERT INTO vec_notes(rowid, embedding) VALUES (?, ?)",
                (rid, blob),
            )
        self._conn.commit()

    def delete(self, note_id: str) -> bool:
        nid = str(note_id or "").strip()
        row = self._conn.execute(
            "SELECT rowid FROM note_map WHERE note_id = ?", (nid,)
        ).fetchone()
        if not row:
            return False
        rid = int(row[0])
        self._conn.execute("DELETE FROM vec_notes WHERE rowid = ?", (rid,))
        self._conn.execute("DELETE FROM note_map WHERE rowid = ?", (rid,))
        self._conn.commit()
        return True

    def search(
        self,
        embedding: list[float],
        *,
        limit: int = 8,
    ) -> list[tuple[str, float]]:
        if self._dim is None or not embedding:
            return []
        if len(embedding) != self._dim:
            raise ValueError(f"query dim {len(embedding)} != index dim {self._dim}")
        limit = max(1, min(int(limit), 50))
        blob = serialize_f32([float(x) for x in embedding])
        knn = self._conn.execute(
            """
            SELECT rowid, distance
            FROM vec_notes
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (blob, limit),
        ).fetchall()
        out: list[tuple[str, float]] = []
        for rid, dist in knn:
            row = self._conn.execute(
                "SELECT note_id FROM note_map WHERE rowid = ?",
                (int(rid),),
            ).fetchone()
            if row:
                out.append((str(row[0]), float(dist)))
        return out

    def upsert_with_embed(
        self,
        note_id: str,
        text: str,
        embed_fn: EmbedFn,
    ) -> None:
        emb = embed_fn(text)
        self.upsert(note_id, text, emb)

    def reindex_notes(
        self,
        notes: list[Any],
        embed_fn: EmbedFn,
    ) -> int:
        """Re-embed notes that have id/title/body. Returns count upserted."""
        n = 0
        for note in notes:
            if isinstance(note, dict):
                nid = str(note.get("id") or "")
                title = str(note.get("title") or "")
                body = str(note.get("body") or "")
            else:
                nid = str(getattr(note, "id", "") or "")
                title = str(getattr(note, "title", "") or "")
                body = str(getattr(note, "body", "") or "")
            if not nid:
                continue
            text = f"{title}\n{body}".strip()
            if not text:
                continue
            self.upsert_with_embed(nid, text, embed_fn)
            n += 1
        return n


def open_vector_index(root: str | Path) -> VectorIndex:
    return VectorIndex(root)
