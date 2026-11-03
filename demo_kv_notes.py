"""Local smoke for KV + notes. Params at bottom only."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fawai_lore import KvValidationError, LoreStore, NoteValidationError, open_lore


def run_demo(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    lore = open_lore(root)

    e = lore.kv.set("dataset.cub.root", "/Volumes/demo/cub", source="demo")
    assert e.key == "dataset.cub.root"
    assert lore.kv.get("dataset.cub.root") is not None
    assert lore.kv.get("dataset.cub.root").value == "/Volumes/demo/cub"
    assert len(lore.kv.list_prefix("dataset.")) == 1
    assert lore.kv.delete("dataset.cub.root") is True
    assert lore.kv.get("dataset.cub.root") is None

    try:
        lore.kv.set("Bad Key", "x")
        raise AssertionError("expected KvValidationError")
    except KvValidationError:
        pass

    note = lore.notes.upsert(
        {
            "id": "disk-h1-layout",
            "title": "H1 disk layout",
            "body": "Models under models/; training dumps under scanly/.",
            "tags": ["disk", "h1"],
            "related_keys": ["disk.h1.root"],
            "source": "demo",
        }
    )
    lore.kv.set("disk.h1.root", "/Volumes/shunyao-h1", source="demo")
    assert lore.notes.get("disk-h1-layout") is not None
    hits = lore.notes.search("H1 models")
    assert any(n.id == "disk-h1-layout" for n in hits)
    assert lore.notes.by_related_key("disk.h1.root")[0].id == note.id

    try:
        lore.notes.upsert({"id": "x", "title": "", "body": "b"})
        raise AssertionError("expected NoteValidationError")
    except NoteValidationError:
        pass

    # reopen persistence
    lore2 = LoreStore(root)
    assert lore2.kv.get("disk.h1.root").value == "/Volumes/shunyao-h1"
    assert lore2.notes.get("disk-h1-layout") is not None
    print(f"[PASS] fawai_lore kv+notes at {root}")


if __name__ == "__main__":
    ROOT = Path(tempfile.mkdtemp(prefix="fawai_lore-demo-"))  # temp demo root
    try:
        run_demo(ROOT)
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
