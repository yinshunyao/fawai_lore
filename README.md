# fawai_lore

Local **KV** + **structured notes** + **extract/sidecar/retrieve** for personal lore. Host chooses the data root.

> Renamed from `pylore`. GitHub: https://github.com/yinshunyao/fawai_lore (old `yinshunyao/pylore` redirects).

## Install

```bash
pip install -e .
# from fawai-avatar: pip install -e ./fawai_lore
```

## Layout

```text
{root}/
  kv.json
  notes/{id}.json
  buffer/provisional.jsonl
  sidecar_log.jsonl
  kv_history.jsonl
```

## Store

```python
from fawai_lore import open_lore

lore = open_lore("/path/to/lore")
lore.kv.set("dataset.cub.root", "/data/cub", aliases=["CUB数据根"])
lore.kv.search("CUB")
lore.notes.upsert({"id": "disk-h1", "title": "H1", "body": "…", "tags": ["disk"]})
lore.kv.set_enabled("dataset.cub.root", False)  # soft-disable; retrieve skips it
lore.notes.delete("disk-h1")  # hard delete
```

## Sidecar (always-on)

Rule-based bilingual cues; optional host LLM via `llm_propose`. **P0 paths + P1 pref KV auto-commit**; weaker rows → buffer.

```python
from fawai_lore import LoreSidecar

side = LoreSidecar.open("/path/to/lore")
side.process_event(text="记住 CUB 放在 /Volumes/data/cub", event_kind="user_text")
```

## Retrieve

```python
from fawai_lore import retrieve, retrieve_prompt_block

retrieve(lore, "CUB 在哪")
print(retrieve_prompt_block(lore, "where is cub"))
```

## LLM extract (optional)

```python
from fawai_lore import propose_with_llm

def chat_fn(messages: list[dict]) -> str:
    ...  # return JSON array string

propose_with_llm("text…", chat_fn=chat_fn, goal="…")
```

## Smoke

```bash
python -m fawai_lore.demo_kv_notes
python -m fawai_lore.demo_sidecar
python -m fawai_lore.demo_retrieve
```

## Vectors (optional, sqlite-vec)

Personal-PC friendly: notes only; KV stays keyword.

```bash
pip install -e ".[vector]"   # or: pip install sqlite-vec
```

```python
from fawai_lore import VectorIndex, fake_embed, retrieve

idx = VectorIndex("/path/to/lore")
idx.reindex_notes(lore.notes.list_all(), fake_embed)  # or API embed_fn
retrieve(lore, "bird images", embed_fn=fake_embed, use_vector=True)
```

File: `{root}/vectors.sqlite` (vec0 + cosine). Dim fixed on first upsert.

```bash
python -m fawai_lore.demo_vector
```

## Boundary

- **In package**: store, rule/LLM propose, sidecar apply, retrieve, buffer, sqlite-vec index.
- **Host**: event hooks, API keys / embed_fn, planner injection, tool wiring.
