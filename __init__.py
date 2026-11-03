"""fawai_lore: local KV + notes + extract/sidecar/retrieve. Host passes the data root."""

from __future__ import annotations

from fawai_lore.buffer import (
    clear_buffer,
    drop_buffer_row,
    list_buffer,
    list_sidecar_log,
    promote_buffer_row,
)
from fawai_lore.extract import Candidate, propose_event, propose_from_text
from fawai_lore.kv import KvEntry, KvStore, KvValidationError
from fawai_lore.llm_extract import merge_candidates, propose_with_llm
from fawai_lore.notes import Note, NoteStore, NoteValidationError
from fawai_lore.retrieve import RetrieveHit, retrieve, retrieve_prompt_block
from fawai_lore.sidecar import LoreSidecar, SidecarResult, process_sidecar_event
from fawai_lore.store import LoreStore, open_lore
from fawai_lore.vector import VectorIndex, fake_embed, open_vector_index, sqlite_vec_available

__all__ = [
    "Candidate",
    "KvEntry",
    "KvStore",
    "KvValidationError",
    "LoreSidecar",
    "LoreStore",
    "Note",
    "NoteStore",
    "NoteValidationError",
    "RetrieveHit",
    "SidecarResult",
    "VectorIndex",
    "clear_buffer",
    "drop_buffer_row",
    "fake_embed",
    "list_buffer",
    "list_sidecar_log",
    "merge_candidates",
    "open_lore",
    "open_vector_index",
    "process_sidecar_event",
    "promote_buffer_row",
    "propose_event",
    "propose_from_text",
    "propose_with_llm",
    "retrieve",
    "retrieve_prompt_block",
    "sqlite_vec_available",
]

__version__ = "0.3.0"
