"""Structured notes: narrated knowledge with tags and related KV keys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fawai_lore._io import atomic_write_json, read_json, utc_now

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_MAX_TITLE = 120
_MAX_BODY = 20000
_MAX_TAGS = 20


class NoteValidationError(ValueError):
    """Note payload rejected."""


@dataclass(frozen=True)
class Note:
    id: str
    title: str
    body: str
    tags: tuple[str, ...]
    related_keys: tuple[str, ...]
    source: str
    created_at: str
    updated_at: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "related_keys": list(self.related_keys),
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Note":
        tags = raw.get("tags") or []
        keys = raw.get("related_keys") or []
        if not isinstance(tags, list):
            tags = []
        if not isinstance(keys, list):
            keys = []
        enabled = raw.get("enabled", True)
        return cls(
            id=str(raw.get("id") or ""),
            title=str(raw.get("title") or ""),
            body=str(raw.get("body") or ""),
            tags=tuple(str(t).strip() for t in tags if str(t).strip()),
            related_keys=tuple(str(k).strip() for k in keys if str(k).strip()),
            source=str(raw.get("source") or ""),
            created_at=str(raw.get("created_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
            enabled=bool(enabled) if enabled is not None else True,
        )


def validate_note_payload(
    raw: dict[str, Any],
    *,
    existing_created_at: str | None = None,
    existing_enabled: bool | None = None,
) -> Note:
    nid = str(raw.get("id") or "").strip().lower()
    if not _ID_RE.match(nid):
        raise NoteValidationError("note id must match [a-z0-9][a-z0-9_-]{1,63}")
    title = str(raw.get("title") or "").strip()
    if not title or len(title) > _MAX_TITLE:
        raise NoteValidationError(f"title required, max {_MAX_TITLE} chars")
    body = str(raw.get("body") or "").strip()
    if not body or len(body) > _MAX_BODY:
        raise NoteValidationError(f"body required, max {_MAX_BODY} chars")
    tags_raw = raw.get("tags") or []
    if not isinstance(tags_raw, list):
        raise NoteValidationError("tags must be a list")
    tags = tuple(str(t).strip() for t in tags_raw if str(t).strip())[:_MAX_TAGS]
    keys_raw = raw.get("related_keys") or []
    if not isinstance(keys_raw, list):
        raise NoteValidationError("related_keys must be a list")
    related: list[str] = []
    for item in keys_raw:
        k = str(item or "").strip().lower()
        if k and k not in related:
            related.append(k)
        if len(related) >= 40:
            break
    source = str(raw.get("source") or "").strip()[:200]
    now = utc_now()
    created = existing_created_at or str(raw.get("created_at") or "").strip() or now
    if "enabled" in raw:
        enabled = bool(raw.get("enabled"))
    elif existing_enabled is not None:
        enabled = bool(existing_enabled)
    else:
        enabled = True
    return Note(
        id=nid,
        title=title,
        body=body,
        tags=tags,
        related_keys=tuple(related),
        source=source,
        created_at=created,
        updated_at=now,
        enabled=enabled,
    )


class NoteStore:
    """One JSON file per note under ``{root}/notes/{id}.json``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._dir = self._root / "notes"
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._dir

    def _path_for(self, note_id: str) -> Path:
        return self._dir / f"{note_id}.json"

    def get(self, note_id: str) -> Note | None:
        nid = str(note_id or "").strip().lower()
        if not nid:
            return None
        raw = read_json(self._path_for(nid))
        if not isinstance(raw, dict):
            return None
        try:
            return Note.from_dict(raw)
        except (TypeError, ValueError):
            return None

    def upsert(self, payload: dict[str, Any]) -> Note:
        nid = str(payload.get("id") or "").strip().lower()
        existing = self.get(nid) if nid else None
        note = validate_note_payload(
            payload,
            existing_created_at=existing.created_at if existing else None,
            existing_enabled=existing.enabled if existing else None,
        )
        atomic_write_json(self._path_for(note.id), note.to_dict())
        return note

    def set_enabled(self, note_id: str, enabled: bool) -> Note | None:
        """Soft-enable / disable without rewriting body. None if missing."""
        note = self.get(note_id)
        if note is None:
            return None
        updated = Note(
            id=note.id,
            title=note.title,
            body=note.body,
            tags=note.tags,
            related_keys=note.related_keys,
            source=note.source,
            created_at=note.created_at,
            updated_at=utc_now(),
            enabled=bool(enabled),
        )
        atomic_write_json(self._path_for(updated.id), updated.to_dict())
        return updated

    def delete(self, note_id: str) -> bool:
        nid = str(note_id or "").strip().lower()
        path = self._path_for(nid)
        if not path.is_file():
            return False
        try:
            path.unlink()
        except OSError:
            return False
        return True

    def list_all(self) -> list[Note]:
        out: list[Note] = []
        for path in sorted(self._dir.glob("*.json")):
            raw = read_json(path)
            if not isinstance(raw, dict):
                continue
            try:
                out.append(Note.from_dict(raw))
            except (TypeError, ValueError):
                continue
        return out

    def search(self, query: str, *, limit: int = 10) -> list[Note]:
        q = (query or "").strip().lower()
        limit = max(1, min(int(limit), 50))
        scored: list[tuple[int, Note]] = []
        for note in self.list_all():
            blob = " ".join(
                [
                    note.id,
                    note.title,
                    note.body,
                    " ".join(note.tags),
                    " ".join(note.related_keys),
                    note.source,
                ]
            ).lower()
            if not q:
                scored.append((0, note))
                continue
            score = 0
            for token in q.split():
                if token and token in blob:
                    score += 1
            if score > 0:
                scored.append((score, note))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [n for _, n in scored[:limit]]

    def by_related_key(self, key: str) -> list[Note]:
        k = str(key or "").strip().lower()
        if not k:
            return []
        return [n for n in self.list_all() if k in n.related_keys]
