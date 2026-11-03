"""Smoke: retrieve + preference extract + buffer promote + llm merge (mock)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fawai_lore import (
    LoreSidecar,
    clear_buffer,
    list_buffer,
    merge_candidates,
    open_lore,
    promote_buffer_row,
    propose_from_text,
    propose_with_llm,
    retrieve,
    retrieve_prompt_block,
)
from fawai_lore.extract import Candidate


def run_demo(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    side = LoreSidecar.open(root)
    side.process_event(
        text="记住 CUB 放在 /Volumes/shunyao-h1/datasets/cub",
        event_kind="user_text",
    )
    side.process_event(
        text="默认编辑器是 vscode",
        event_kind="user_text",
    )
    # weak path → buffer (long text, no location/remember cue → not bare P0)
    side.process_event(
        text=(
            "today we chatted about many unrelated topics and someone casually "
            "mentioned /data/orphan/project among other noise without saying it "
            "is the canonical root or that we should remember it for later use"
        ),
        event_kind="user_text",
    )

    lore = open_lore(root)
    assert lore.kv.get("dataset.cub.root") is not None
    assert lore.kv.get("pref.editor") is not None or lore.kv.get("pref.default") is not None

    hits = retrieve(lore, "CUB 数据")
    assert any(h.kind == "kv" for h in hits), hits
    block = retrieve_prompt_block(lore, "where is cub")
    assert "个人知识库" in block and "cub" in block.lower()

    buf = list_buffer(root)
    assert buf, "expected buffered weak path"
    promoted = promote_buffer_row(root, buf[-1])
    assert promoted.get("action") in {"set", "note_upsert", "skip"}

    def fake_chat(messages):
        return (
            '[{"kind":"kv","importance":"p0","key":"path.demo.root",'
            '"value":"/Volumes/demo","aliases":["demo根"],"evidence":"x",'
            '"confidence":"high","reason":"mock"}]'
        )

    llm_cands = propose_with_llm(
        "demo root /Volumes/demo",
        chat_fn=fake_chat,
        merge_rules=True,
    )
    assert any(c.key == "path.demo.root" for c in llm_cands if c.kind == "kv")
    merged = merge_candidates(
        propose_from_text("记住 /Volumes/a/b"),
        [Candidate(kind="skip", importance="p3", op="skip", reason="x")],
    )
    assert merged

    clear_buffer(root)
    assert list_buffer(root) == []
    print(f"[PASS] retrieve/pref/buffer/llm-mock at {root}")


if __name__ == "__main__":
    ROOT = Path(tempfile.mkdtemp(prefix="fawai_lore-ret-"))
    try:
        run_demo(ROOT)
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
