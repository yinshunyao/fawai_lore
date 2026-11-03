"""Key-value facts: exact lookup by stable key (paths, prefs, handles)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fawai_lore._io import atomic_write_json, read_json, utc_now

# dotted namespaces: dataset.cub.root, host.4090d.data_dir
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
_MAX_KEY_LEN = 200
_MAX_VALUE_CHARS = 8000
_MAX_ALIASES = 20


class KvValidationError(ValueError):
    """KV payload rejected."""


@dataclass(frozen=True)
class KvEntry:
    key: str
    value: Any
    updated_at: str
    source: str = ""
    aliases: tuple[str, ...] = ()
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "updated_at": self.updated_at,
            "source": self.source,
            "aliases": list(self.aliases),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KvEntry":
        aliases = raw.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        enabled = raw.get("enabled", True)
        return cls(
            key=str(raw.get("key") or ""),
            value=raw.get("value"),
            updated_at=str(raw.get("updated_at") or ""),
            source=str(raw.get("source") or ""),
            aliases=tuple(str(a).strip() for a in aliases if str(a).strip())[
                :_MAX_ALIASES
            ],
            enabled=bool(enabled) if enabled is not None else True,
        )


def validate_key(key: str) -> str:
    k = str(key or "").strip().lower()
    if not k or len(k) > _MAX_KEY_LEN:
        raise KvValidationError(f"key required, max {_MAX_KEY_LEN} chars")
    if not _KEY_RE.match(k):
        raise KvValidationError(
            "key must be dotted lowercase tokens: [a-z][a-z0-9_]* (e.g. dataset.cub.root)"
        )
    return k


def _normalize_aliases(aliases: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    out: list[str] = []
    for item in aliases or ():
        a = str(item or "").strip()
        if a and a not in out:
            out.append(a)
        if len(out) >= _MAX_ALIASES:
            break
    return tuple(out)


def _merge_aliases(
    existing: tuple[str, ...] | list[str] | None,
    incoming: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    return _normalize_aliases([*(existing or ()), *(incoming or ())])


def _value_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (str, bytes)):
        return len(value)
    try:
        import json

        return len(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(value))


def validate_value(value: Any) -> Any:
    if value is not None and not isinstance(
        value, (str, int, float, bool, list, dict)
    ):
        raise KvValidationError(
            "value must be JSON-serializable: str|number|bool|list|dict|null"
        )
    if _value_size(value) > _MAX_VALUE_CHARS:
        raise KvValidationError(f"value too large, max {_MAX_VALUE_CHARS} chars")
    return value


class KvStore:
    """Single-file map under ``{root}/kv.json``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "kv.json"

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, dict[str, Any]]:
        raw = read_json(self._path)
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for key, entry in raw.items():
            if isinstance(entry, dict) and "value" in entry:
                out[str(key)] = entry
            else:
                out[str(key)] = {
                    "key": str(key),
                    "value": entry,
                    "updated_at": "",
                    "source": "",
                    "aliases": [],
                }
        return out

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        atomic_write_json(self._path, data)

    def get(self, key: str) -> KvEntry | None:
        k = validate_key(key)
        row = self._load().get(k)
        if row is None:
            return None
        row = {**row, "key": k}
        return KvEntry.from_dict(row)

    def set(
        self,
        key: str,
        value: Any,
        *,
        source: str = "",
        aliases: list[str] | tuple[str, ...] | None = None,
        merge_aliases: bool = True,
        enabled: bool | None = None,
    ) -> KvEntry:
        k = validate_key(key)
        v = validate_value(value)
        src = str(source or "").strip()[:200]
        data = self._load()
        prev = data.get(k)
        prev_aliases = ()
        prev_enabled = True
        if isinstance(prev, dict):
            prev_aliases = tuple(prev.get("aliases") or ())
            prev_enabled = bool(prev.get("enabled", True))
        if merge_aliases:
            merged = _merge_aliases(prev_aliases, aliases)
        else:
            merged = _normalize_aliases(aliases)
        entry = KvEntry(
            key=k,
            value=v,
            updated_at=utc_now(),
            source=src,
            aliases=merged,
            enabled=prev_enabled if enabled is None else bool(enabled),
        )
        data[k] = entry.to_dict()
        self._save(data)
        return entry

    def set_enabled(self, key: str, enabled: bool) -> KvEntry | None:
        """Soft-enable / disable without changing value. None if missing."""
        k = validate_key(key)
        data = self._load()
        row = data.get(k)
        if row is None:
            return None
        entry = KvEntry.from_dict({**row, "key": k, "enabled": bool(enabled)})
        # keep updated_at when only toggling; stamp source of truth on disk
        data[k] = {**entry.to_dict(), "updated_at": utc_now()}
        self._save(data)
        return KvEntry.from_dict(data[k])

    def delete(self, key: str) -> bool:
        k = validate_key(key)
        data = self._load()
        if k not in data:
            return False
        del data[k]
        self._save(data)
        return True

    def list_prefix(self, prefix: str = "") -> list[KvEntry]:
        p = str(prefix or "").strip().lower()
        if p:
            if not re.match(r"^[a-z0-9_.]*$", p):
                raise KvValidationError("prefix must be lowercase [a-z0-9_.]*")
        items: list[KvEntry] = []
        for key, row in sorted(self._load().items()):
            if p and not key.startswith(p):
                continue
            items.append(KvEntry.from_dict({**row, "key": key}))
        return items

    def keys(self) -> list[str]:
        return sorted(self._load().keys())

    def search(self, query: str, *, limit: int = 20) -> list[KvEntry]:
        """Keyword search over key, aliases, and string values (zh/en substring)."""
        q = (query or "").strip().lower()
        limit = max(1, min(int(limit), 50))
        if not q:
            return self.list_prefix()[:limit]
        tokens = [t for t in q.split() if t]
        scored: list[tuple[int, KvEntry]] = []
        for key, row in self._load().items():
            entry = KvEntry.from_dict({**row, "key": key})
            blob = " ".join(
                [
                    entry.key,
                    " ".join(entry.aliases),
                    str(entry.value) if entry.value is not None else "",
                ]
            ).lower()
            score = sum(1 for t in tokens if t in blob)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: (-x[0], x[1].key))
        return [e for _, e in scored[:limit]]
