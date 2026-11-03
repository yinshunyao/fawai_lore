"""Propose lore candidates from text (rule-based P0 paths; bilingual cues)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fawai_lore import signals

# Absolute / home paths; stop at CJK punctuation and common wrappers
_PATH_RE = re.compile(
    r"(?:"
    r"~(?:/[^\s'\"`，。；;、！!？?）)\]]+)+"
    r"|"
    r"/(?:Users|Volumes|home|root|data|opt|mnt|media|workspace|var|private|tmp)"
    r"/[^\s'\"`，。；;、！!？?）)\]]*"
    r"|"
    r"/tmp\b"
    r"|"
    r"[A-Za-z]:\\[^\s'\"`]+"
    r")"
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class Candidate:
    kind: str  # kv | note | skip
    importance: str  # p0 | p1 | p2 | p3
    key: str | None = None
    value: Any = None
    aliases: list[str] = field(default_factory=list)
    title: str | None = None
    body: str | None = None
    evidence: str = ""
    confidence: str = "mid"  # high | mid | low
    op: str = "set"  # set | skip | buffer
    reason: str = ""
    locale: str = "mixed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "importance": self.importance,
            "key": self.key,
            "value": self.value,
            "aliases": list(self.aliases),
            "title": self.title,
            "body": self.body,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "op": self.op,
            "reason": self.reason,
            "locale": self.locale,
        }


def _trim_path(raw: str) -> str:
    p = raw.strip().rstrip("/\\")
    # strip trailing CJK/latin punctuation if regex leaked
    while p and p[-1] in "，。；;、！!？?）)]}>'\"":
        p = p[:-1]
    return p


def _slug_from_path(path: str) -> str:
    parts = [x for x in path.replace("\\", "/").split("/") if x and x != "~"]
    tail = parts[-2:] if len(parts) >= 2 else parts[-1:]
    slug = _SLUG_RE.sub("_", "_".join(tail).lower()).strip("_")
    return (slug or "unnamed")[:48]


def _hint_key(context: str, path: str) -> str | None:
    blob = f"{context} {path}".lower()
    for synonyms, key in signals.KEY_HINTS:
        if any(s.lower() in blob for s in synonyms):
            return key
    return None


def _window(text: str, start: int, end: int, *, radius: int = 48) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b]


def _aliases_for(path: str, context: str) -> list[str]:
    aliases: list[str] = []
    # short path tail as alias (language-agnostic)
    tail = path.replace("\\", "/").rstrip("/").split("/")[-1]
    if tail and len(tail) > 1:
        aliases.append(tail)
    # keep a short human phrase from context if remember-cued
    ctx = context.strip()
    if len(ctx) <= 40 and ctx:
        aliases.append(ctx)
    return aliases[:8]


def propose_from_text(
    text: str,
    *,
    source: str = "text",
    goal: str = "",
) -> list[Candidate]:
    """
    Rule extract: concrete paths + bilingual location/remember cues → P0 KV candidates.
    Does not write. Secrets-looking tokens are skipped.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    locale = signals.locale_hint(raw)
    combined = f"{goal}\n{raw}" if goal else raw
    out: list[Candidate] = []
    seen_paths: set[str] = set()

    for m in _PATH_RE.finditer(combined):
        path = _trim_path(m.group(0))
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        win = _window(combined, m.start(), m.end())
        evidence = win.strip()

        if signals.path_looks_ephemeral(path) or signals.discard_cued(win):
            out.append(
                Candidate(
                    kind="skip",
                    importance="p3",
                    value=path,
                    evidence=evidence,
                    confidence="high",
                    op="skip",
                    reason="ephemeral_or_discard_cue",
                    locale=locale,
                )
            )
            continue

        # secret-ish: password= / token= adjacent
        if re.search(r"(password|passwd|secret|token|api[_-]?key)\s*[=:]", win, re.I):
            out.append(
                Candidate(
                    kind="skip",
                    importance="p3",
                    value=path,
                    evidence=evidence,
                    confidence="high",
                    op="skip",
                    reason="secret_adjacent",
                    locale=locale,
                )
            )
            continue

        # Cue only in local window (avoid whole-message false hits like English "root")
        loc = signals.location_cued(win)
        rem = signals.remember_cued(win)
        # Short whole message with remember/location (e.g.「记住 /path」)
        short_path_msg = len(combined) <= 96 and (
            signals.location_cued(combined) or signals.remember_cued(combined)
        )

        if not (loc or rem or short_path_msg):
            out.append(
                Candidate(
                    kind="kv",
                    importance="p2",
                    key=_hint_key(win, path) or f"path.{_slug_from_path(path)}",
                    value=path,
                    aliases=_aliases_for(path, win),
                    evidence=evidence,
                    confidence="low",
                    op="buffer",
                    reason="path_without_strong_cue",
                    locale=locale,
                )
            )
            continue

        key = _hint_key(win, path) or f"path.{_slug_from_path(path)}"
        out.append(
            Candidate(
                kind="kv",
                importance="p0",
                key=key,
                value=path,
                aliases=_aliases_for(path, win),
                evidence=evidence,
                confidence="high",
                op="set",
                reason="path_p0",
                locale=locale,
            )
        )

    # Structured preference → P1 KV when possible
    pref = _propose_preference(combined, locale=locale)
    if pref is not None:
        sig = ("kv", str(pref.key), str(pref.value))
        if not any((c.kind, str(c.key), str(c.value)) == sig for c in out if c.kind == "kv"):
            out.append(pref)
    elif not any(c.kind == "kv" for c in out) and signals.preference_cued(combined):
        out.append(
            Candidate(
                kind="note",
                importance="p1",
                title="preference",
                body=raw[:2000],
                evidence=raw[:500],
                confidence="mid",
                op="buffer",
                reason="preference_cue_no_path",
                locale=locale,
            )
        )

    return out


_PREF_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?:默认|prefer|default)\s*(?:用|使用|to|is|=)?\s*[「\"']?([^「」\"'\n,，。]{1,40})",
            re.I,
        ),
        "pref.default",
    ),
    (
        re.compile(
            r"(?:编辑器|editor)\s*(?:默认|用|是|=|:)?\s*[「\"']?([a-zA-Z0-9._\-]{2,40})",
            re.I,
        ),
        "pref.editor",
    ),
    (
        re.compile(
            r"(?:以后都|always|from now on)\s*(?:用|使用|use)?\s*[「\"']?([^「」\"'\n,，。]{1,40})",
            re.I,
        ),
        "pref.always",
    ),
)


def _propose_preference(text: str, *, locale: str) -> Candidate | None:
    if not signals.preference_cued(text) and not signals.remember_cued(text):
        # still allow explicit editor/default patterns
        if not re.search(r"默认|prefer|default|editor|编辑器", text, re.I):
            return None
    for pat, key in _PREF_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        val = m.group(1).strip().strip("「」\"'")
        if not val or len(val) > 80:
            continue
        if signals.discard_cued(val):
            continue
        return Candidate(
            kind="kv",
            importance="p1",
            key=key,
            value=val,
            aliases=[val],
            evidence=m.group(0)[:200],
            confidence="mid",
            op="set",
            reason="preference_pattern",
            locale=locale,
        )
    return None


def propose_event(
    *,
    text: str = "",
    goal: str = "",
    final_thought: str = "",
    event_kind: str = "text",
) -> list[Candidate]:
    """Merge proposes from user text and task final thought."""
    chunks: list[tuple[str, str]] = []
    if text.strip():
        chunks.append((text, f"{event_kind}:text"))
    if final_thought.strip():
        chunks.append((final_thought, f"{event_kind}:final_thought"))
    if goal.strip() and event_kind == "task_finished":
        # goal alone rarely has new paths; still scan once with low weight via propose
        chunks.append((goal, f"{event_kind}:goal"))

    merged: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for body, src in chunks:
        for cand in propose_from_text(body, source=src, goal=goal if src.endswith(":goal") else ""):
            sig = (cand.kind, str(cand.key or ""), str(cand.value or ""))
            if sig in seen:
                continue
            seen.add(sig)
            cand.reason = f"{cand.reason}|{src}"
            merged.append(cand)
    return merged
