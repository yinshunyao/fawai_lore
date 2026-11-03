"""Lore root facade: KV + notes under one directory."""

from __future__ import annotations

from pathlib import Path

from fawai_lore.kv import KvStore
from fawai_lore.notes import NoteStore


class LoreStore:
    """
    Layout::

        {root}/
          kv.json
          notes/{id}.json
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self.kv = KvStore(self._root)
        self.notes = NoteStore(self._root)

    @property
    def root(self) -> Path:
        return self._root


def open_lore(root: str | Path) -> LoreStore:
    """Open or create a lore store at ``root`` (host chooses the path)."""
    return LoreStore(root)
