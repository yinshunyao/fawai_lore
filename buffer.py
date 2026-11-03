"""Provisional buffer (non-P0 candidates) helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fawai_lore._io import utc_now
from fawai_lore.extract import Candidate
from fawai_lore.sidecar import LoreSidecar
from fawai_lore.store import open_lore


def buffer_path(root: str | Path) -> Path:
    return Path(root) / "buffer" / "provisional.jsonl"


def list_buffer(root: str | Path, *, limit: int = 50) -> list[dict[str, Any]]:
    path = buffer_path(root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for raw in lines[-max(1, min(int(limit), 200)) :]:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def clear_buffer(root: str | Path) -> int:
    path = buffer_path(root)
    if not path.is_file():
        return 0
    try:
        n = sum(1 for _ in path.open(encoding="utf-8") if _.strip())
        path.write_text("", encoding="utf-8")
        return n
    except OSError:
        return 0


def promote_buffer_row(
    root: str | Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Apply one buffered row into KV/notes via sidecar apply rules."""
    side = LoreSidecar.open(root)
    kind = str(row.get("kind") or "")
    if kind == "kv" and row.get("key"):
        cand = Candidate(
            kind="kv",
            importance=str(row.get("importance") or "p1"),
            key=str(row["key"]),
            value=row.get("value"),
            aliases=list(row.get("aliases") or []),
            evidence=str(row.get("evidence") or ""),
            confidence=str(row.get("confidence") or "mid"),
            op="set",
            reason="promote_buffer",
            locale=str(row.get("locale") or "mixed"),
        )
        return side._apply_kv(cand, task_id=str(row.get("task_id") or ""), event_kind="buffer_promote")
    if kind == "note":
        lore = open_lore(root)
        note = lore.notes.upsert(
            {
                "id": str(row.get("key") or row.get("id") or f"buf-{utc_now().replace(':', '')[:16]}"),
                "title": str(row.get("title") or "buffered"),
                "body": str(row.get("body") or row.get("evidence") or "empty"),
                "tags": ["buffered", "promoted"],
                "related_keys": list(row.get("aliases") or [])[:20],
                "source": "buffer_promote",
            }
        )
        return {"action": "note_upsert", "note": note.to_dict()}
    return {"action": "skip", "reason": "unsupported_buffer_row"}
