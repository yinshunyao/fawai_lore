"""Always-on lore sidecar: propose → grade → auto-commit P0; buffer P1/P2 by default."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fawai_lore._io import utc_now
from fawai_lore.extract import Candidate, propose_event
from fawai_lore.store import LoreStore, open_lore

LlmProposeFn = Callable[..., list[Candidate]]


@dataclass
class SidecarResult:
    applied: list[dict[str, Any]] = field(default_factory=list)
    buffered: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "buffered": self.buffered,
            "skipped": self.skipped,
            "conflicts": self.conflicts,
        }


class LoreSidecar:
    """
    Host feeds conversation/task events; P0 KV commits automatically.
    Default: P1/P2 go to ``{root}/buffer/provisional.jsonl`` for confirmation.
    Set ``auto_p1_kv=True`` to auto-commit preference KV (legacy).
    """

    def __init__(self, store: LoreStore) -> None:
        self.store = store
        self._buffer_dir = store.root / "buffer"
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_path = self._buffer_dir / "provisional.jsonl"
        self._log_path = store.root / "sidecar_log.jsonl"

    @classmethod
    def open(cls, root: str | Path) -> "LoreSidecar":
        return cls(open_lore(root))

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def process_event(
        self,
        *,
        text: str = "",
        goal: str = "",
        final_thought: str = "",
        event_kind: str = "text",
        task_id: str = "",
        auto_p0: bool = True,
        auto_p1_kv: bool = False,
        auto_p1_notes: bool = False,
        llm_propose: LlmProposeFn | None = None,
    ) -> SidecarResult:
        cands = propose_event(
            text=text,
            goal=goal,
            final_thought=final_thought,
            event_kind=event_kind,
        )
        if llm_propose is not None:
            try:
                blob = "\n".join(x for x in (text, final_thought) if (x or "").strip())
                extra = llm_propose(blob, goal=goal) or []
                from fawai_lore.llm_extract import merge_candidates

                cands = merge_candidates(cands, list(extra))
            except Exception:
                pass

        result = SidecarResult()
        for cand in cands:
            row = cand.to_dict()
            row["task_id"] = task_id
            row["event_kind"] = event_kind
            row["ts"] = utc_now()

            if cand.kind == "skip" or cand.op == "skip" or cand.importance == "p3":
                result.skipped.append(row)
                self._append_jsonl(self._log_path, {"action": "skip", **row})
                continue

            if cand.kind == "kv" and cand.key and cand.importance == "p0" and auto_p0:
                applied = self._apply_kv(cand, task_id=task_id, event_kind=event_kind)
                if applied.get("conflict"):
                    result.conflicts.append(applied)
                else:
                    result.applied.append(applied)
                continue

            if (
                cand.kind == "kv"
                and cand.key
                and cand.importance == "p1"
                and auto_p1_kv
            ):
                applied = self._apply_kv(cand, task_id=task_id, event_kind=event_kind)
                result.applied.append(applied)
                continue

            if (
                cand.kind == "note"
                and cand.importance == "p1"
                and auto_p1_notes
                and (cand.body or cand.title)
            ):
                applied = self._apply_note(cand, task_id=task_id, event_kind=event_kind)
                result.applied.append(applied)
                continue

            result.buffered.append(row)
            self._append_jsonl(self._buffer_path, row)
            self._append_jsonl(self._log_path, {"action": "buffer", **row})

        return result

    def _apply_kv(
        self,
        cand: Candidate,
        *,
        task_id: str,
        event_kind: str,
    ) -> dict[str, Any]:
        assert cand.key is not None
        prev = self.store.kv.get(cand.key)
        conflict = False
        if prev is not None and prev.value != cand.value:
            conflict = True
            hist_path = self.store.root / "kv_history.jsonl"
            self._append_jsonl(
                hist_path,
                {
                    "ts": utc_now(),
                    "key": cand.key,
                    "old": prev.value,
                    "new": cand.value,
                    "task_id": task_id,
                    "event_kind": event_kind,
                },
            )
        entry = self.store.kv.set(
            cand.key,
            cand.value,
            source=f"sidecar:{event_kind}",
            aliases=cand.aliases,
            merge_aliases=True,
        )
        out = {
            "action": "set",
            "conflict": conflict,
            "key": entry.key,
            "value": entry.value,
            "aliases": list(entry.aliases),
            "evidence": cand.evidence,
            "importance": cand.importance,
            "locale": cand.locale,
            "task_id": task_id,
            "event_kind": event_kind,
            "ts": utc_now(),
        }
        self._append_jsonl(self._log_path, out)
        return out

    def _apply_note(
        self,
        cand: Candidate,
        *,
        task_id: str,
        event_kind: str,
    ) -> dict[str, Any]:
        import re

        title = (cand.title or "memory").strip()[:120]
        nid_base = re.sub(r"[^a-z0-9_-]+", "-", title.lower()).strip("-")[:40] or "note"
        note = self.store.notes.upsert(
            {
                "id": nid_base,
                "title": title,
                "body": (cand.body or cand.evidence or title)[:8000],
                "tags": ["sidecar", cand.importance],
                "related_keys": [cand.key] if cand.key else [],
                "source": f"sidecar:{event_kind}:{task_id}",
            }
        )
        out = {
            "action": "note_upsert",
            "note": note.to_dict(),
            "importance": cand.importance,
            "task_id": task_id,
            "event_kind": event_kind,
            "ts": utc_now(),
        }
        self._append_jsonl(self._log_path, out)
        return out


def process_sidecar_event(
    root: str | Path,
    **kwargs: Any,
) -> SidecarResult:
    return LoreSidecar.open(root).process_event(**kwargs)
