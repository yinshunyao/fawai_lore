"""Smoke: sqlite-vec note index + hybrid retrieve (fake embeddings)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fawai_lore import open_lore, retrieve
from fawai_lore.vector import VectorIndex, fake_embed, sqlite_vec_available


def run_demo(root: Path) -> None:
    assert sqlite_vec_available(), "pip install sqlite-vec"
    if root.exists():
        shutil.rmtree(root)
    lore = open_lore(root)
    lore.notes.upsert(
        {
            "id": "cub-layout",
            "title": "CUB dataset layout",
            "body": "Bird images live under the cub images folder on the external disk.",
            "tags": ["dataset"],
        }
    )
    lore.notes.upsert(
        {
            "id": "ssh-tips",
            "title": "SSH expect",
            "body": "Use expect scripts for password prompts on remote training hosts.",
            "tags": ["ops"],
        }
    )
    idx = VectorIndex(root)
    try:
        n = idx.reindex_notes(lore.notes.list_all(), fake_embed)
        assert n == 2, n
        q = fake_embed("where are the bird dataset images stored")
        hits = idx.search(q, limit=2)
        assert hits and any(nid == "cub-layout" for nid, _ in hits), hits
        # bag-of-token fake embed: bird/images/dataset should beat ssh
        assert hits[0][0] == "cub-layout", hits
    finally:
        idx.close()

    hybrid = retrieve(
        lore,
        "bird images external disk",
        limit=5,
        embed_fn=fake_embed,
        use_vector=True,
    )
    assert any(h.key_or_id == "cub-layout" for h in hybrid), hybrid
    print(f"[PASS] sqlite-vec hybrid retrieve at {root}")


if __name__ == "__main__":
    ROOT = Path(tempfile.mkdtemp(prefix="fawai_lore-vec-"))
    try:
        run_demo(ROOT)
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
