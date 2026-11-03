"""Retrieve KV + notes for planner injection or tools (keyword ∪ optional vectors)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fawai_lore.store import LoreStore, open_lore

EmbedFn = Callable[[str], list[float]]


@dataclass
class RetrieveHit:
    kind: str  # kv | note
    score: float
    key_or_id: str
    summary: str
    raw: dict[str, Any]
    source: str = "keyword"  # keyword | vector

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "score": self.score,
            "id": self.key_or_id,
            "summary": self.summary,
            "source": self.source,
            "raw": self.raw,
        }


def retrieve(
    store: LoreStore | str | Path,
    query: str,
    *,
    limit: int = 8,
    embed_fn: EmbedFn | None = None,
    use_vector: bool = False,
) -> list[RetrieveHit]:
    lore = store if isinstance(store, LoreStore) else open_lore(store)
    q = (query or "").strip()
    limit = max(1, min(int(limit), 30))
    hits: list[RetrieveHit] = []

    for entry in lore.kv.search(q, limit=limit):
        if not entry.enabled:
            continue
        summary = f"{entry.key} = {entry.value}"
        if entry.aliases:
            summary += f" (aliases: {', '.join(entry.aliases[:5])})"
        hits.append(
            RetrieveHit(
                kind="kv",
                score=2.0,
                key_or_id=entry.key,
                summary=summary,
                raw=entry.to_dict(),
                source="keyword",
            )
        )

    for note in lore.notes.search(q, limit=limit):
        if not note.enabled:
            continue
        body_preview = (note.body or "").replace("\n", " ")[:120]
        summary = f"{note.id}: {note.title} — {body_preview}"
        hits.append(
            RetrieveHit(
                kind="note",
                score=1.0,
                key_or_id=note.id,
                summary=summary,
                raw=note.to_dict(),
                source="keyword",
            )
        )

    if use_vector and embed_fn and q:
        try:
            from fawai_lore.vector import VectorIndex, sqlite_vec_available

            if sqlite_vec_available():
                idx = VectorIndex(lore.root)
                try:
                    emb = embed_fn(q)
                    for note_id, dist in idx.search(emb, limit=limit):
                        note = lore.notes.get(note_id)
                        if note is None or not note.enabled:
                            continue
                        # cosine distance: smaller better → score in ~[0, 2]
                        score = 1.5 / (1.0 + float(dist))
                        body_preview = (note.body or "").replace("\n", " ")[:120]
                        hits.append(
                            RetrieveHit(
                                kind="note",
                                score=score,
                                key_or_id=note.id,
                                summary=f"{note.id}: {note.title} — {body_preview}",
                                raw=note.to_dict(),
                                source="vector",
                            )
                        )
                finally:
                    idx.close()
        except Exception:
            pass

    hits.sort(key=lambda h: (-h.score, h.kind, h.key_or_id))
    seen: set[tuple[str, str]] = set()
    out: list[RetrieveHit] = []
    for h in hits:
        sig = (h.kind, h.key_or_id)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def retrieve_prompt_block(
    store: LoreStore | str | Path,
    query: str,
    *,
    limit: int = 8,
    max_chars: int = 1800,
    embed_fn: EmbedFn | None = None,
    use_vector: bool = False,
) -> str:
    hits = retrieve(
        store,
        query,
        limit=limit,
        embed_fn=embed_fn,
        use_vector=use_vector,
    )
    if not hits:
        lore = store if isinstance(store, LoreStore) else open_lore(store)
        keys = lore.kv.keys()[:12]
        if not keys:
            return ""
        lines = ["【个人知识库】（无精确命中；部分 KV 键）"]
        for k in keys:
            e = lore.kv.get(k)
            if e is None or not e.enabled:
                continue
            lines.append(f"- {e.key} = {e.value}")
        text = "\n".join(lines)
        return text[:max_chars]

    lines = ["【个人知识库】（与本目标相关；key 为英文，值可为中英原文）"]
    for h in hits:
        prefix = "KV" if h.kind == "kv" else "笔记"
        tag = "" if h.source == "keyword" else f"/{h.source}"
        lines.append(f"- [{prefix}{tag}] {h.summary}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n…(truncated)"
    return text
