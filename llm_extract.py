"""Optional LLM propose; host injects chat_fn so fawai_lore stays free of vendor SDKs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from fawai_lore.extract import Candidate, propose_from_text

ChatFn = Callable[[list[dict[str, str]]], str]

_SYSTEM = """You extract durable personal facts for a long-term memory store.
User text may be Chinese, English, or mixed.
Output ONLY a JSON array. Each item:
{"kind":"kv"|"note"|"skip","importance":"p0"|"p1"|"p2"|"p3",
 "key":"english.dotted.key or null","value":any,"aliases":["zh or en"],
 "title":null|"...","body":null|"...","evidence":"short quote",
 "confidence":"high"|"mid"|"low","reason":"short"}
Rules:
- kind=kv keys MUST be English dotted lowercase (e.g. dataset.cub.root).
- value/title/body/aliases KEEP the user's language; do not translate away.
- Paths, data roots, machine dirs → importance p0, kind kv.
- Preferences → p1 kv (pref.*) or note.
- Temporary /tmp / secrets → skip p3.
- If unsure, skip. Prefer precision over recall.
- Empty array if nothing durable.
"""


def _parse_candidates(content: str) -> list[Candidate]:
    text = (content or "").strip()
    if not text:
        return []
    # strip markdown fence if any
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # try array slice
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    out: list[Candidate] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "skip").strip().lower()
        importance = str(item.get("importance") or "p2").strip().lower()
        if importance not in {"p0", "p1", "p2", "p3"}:
            importance = "p2"
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        key = item.get("key")
        key_s = str(key).strip().lower() if key else None
        if key_s == "null" or key_s == "":
            key_s = None
        op = "skip" if kind == "skip" or importance == "p3" else "set"
        if kind == "note" and importance in {"p1", "p2"}:
            op = "buffer" if importance == "p2" else "set"
        out.append(
            Candidate(
                kind=kind if kind in {"kv", "note", "skip"} else "skip",
                importance=importance,
                key=key_s,
                value=item.get("value"),
                aliases=[str(a).strip() for a in aliases if str(a).strip()][:12],
                title=(str(item["title"]).strip() if item.get("title") else None),
                body=(str(item["body"]).strip() if item.get("body") else None),
                evidence=str(item.get("evidence") or "")[:500],
                confidence=str(item.get("confidence") or "mid"),
                op=op,
                reason=f"llm:{item.get('reason') or 'extract'}",
                locale="mixed",
            )
        )
    return out


def propose_with_llm(
    text: str,
    *,
    chat_fn: ChatFn,
    goal: str = "",
    merge_rules: bool = True,
) -> list[Candidate]:
    """Call host LLM; optionally merge with rule-based proposes (rules win on same path)."""
    body = (text or "").strip()
    if not body and not (goal or "").strip():
        return []
    user = body
    if goal.strip():
        user = f"goal:\n{goal.strip()}\n\ntext:\n{body}"
    try:
        content = chat_fn(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user[:8000]},
            ]
        )
    except Exception:
        content = ""
    llm_cands = _parse_candidates(content)
    if not merge_rules:
        return llm_cands
    rule_cands = propose_from_text(body or goal, goal=goal)
    return merge_candidates(rule_cands, llm_cands)


def merge_candidates(*groups: list[Candidate]) -> list[Candidate]:
    """Dedupe: prefer rule path_p0 over llm for same key/value; keep uniques."""
    out: list[Candidate] = []
    seen: set[tuple[str, str, str]] = set()
    # rules first if callers pass them first
    for group in groups:
        for cand in group:
            sig = (cand.kind, str(cand.key or ""), str(cand.value or "")[:200])
            if sig in seen:
                continue
            seen.add(sig)
            out.append(cand)
    return out


def make_openai_compatible_chat_fn(client: Any) -> ChatFn:
    """Adapter: object with .chat(messages)->dict containing content."""

    def _fn(messages: list[dict[str, str]]) -> str:
        resp = client.chat(messages)
        if not isinstance(resp, dict) or not resp.get("ok"):
            return ""
        return str(resp.get("content") or "")

    return _fn
