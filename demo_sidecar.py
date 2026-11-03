"""Smoke: bilingual P0 path extract + sidecar auto-commit."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fawai_lore import LoreSidecar, open_lore, propose_from_text


def run_demo(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    side = LoreSidecar.open(root)

    zh = "记住 CUB 数据集放在 /Volumes/shunyao-h1/datasets/cub"
    en = "Remember the pretrained weights root is /Volumes/shunyao-h1/models/预训练模型"
    tmp = "临时文件在 /tmp/scratch-run-1"

    c_zh = propose_from_text(zh)
    assert any(c.kind == "kv" and c.importance == "p0" for c in c_zh), c_zh
    assert any(c.key == "dataset.cub.root" for c in c_zh if c.kind == "kv"), c_zh

    c_en = propose_from_text(en)
    assert any(c.kind == "kv" and "pretrained" in (c.key or "") for c in c_en), c_en

    c_tmp = propose_from_text(tmp)
    assert all(c.importance == "p3" or c.op == "skip" for c in c_tmp), c_tmp

    r1 = side.process_event(text=zh, event_kind="user_text")
    assert r1.applied, r1
    r2 = side.process_event(text=en, event_kind="user_text")
    assert r2.applied, r2
    r3 = side.process_event(text=tmp, event_kind="user_text")
    assert r3.skipped and not r3.applied, r3

    lore = open_lore(root)
    cub = lore.kv.get("dataset.cub.root")
    assert cub is not None and "cub" in str(cub.value).lower()
    # zh query hits via alias/value
    hits = lore.kv.search("CUB 数据")
    assert hits, hits

    # English task finish same key family
    r4 = side.process_event(
        goal="where is cub data",
        final_thought="CUB dataset root: /Volumes/shunyao-h1/datasets/cub",
        event_kind="task_finished",
        task_id="t1",
    )
    assert r4.applied or r4.conflicts, r4
    print(f"[PASS] sidecar P0 zh/en at {root}")


if __name__ == "__main__":
    ROOT = Path(tempfile.mkdtemp(prefix="fawai_lore-side-"))
    try:
        run_demo(ROOT)
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
