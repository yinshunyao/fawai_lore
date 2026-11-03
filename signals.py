"""Bilingual (zh/en) cue tables for lore importance grading."""

from __future__ import annotations

# Location / path predicates — P0 when near a concrete path
LOCATION_CUES_ZH = (
    "放在",
    "放到",
    "落在",
    "落盘",
    "位于",
    "目录",
    "路径",
    "根目录",
    "数据根",
    "硬盘",
    "外接盘",
    "存在",
    "存到",
    "保存到",
)
LOCATION_CUES_EN = (
    "path",
    "directory",
    "dir",
    "root",
    "located",
    "stored under",
    "stored at",
    "saved to",
    "save to",
    "lives at",
    "on disk",
    "data root",
)

# Explicit remember — boost to P0/P1
REMEMBER_CUES_ZH = ("记住", "记一下", "记下", "别忘了", "以后用")
REMEMBER_CUES_EN = (
    "remember",
    "note that",
    "don't forget",
    "keep in mind",
    "remember for later",
    "please remember",
)
# Preference — P1
PREF_CUES_ZH = ("默认", "以后都", "优先", "别再用", "改成")
PREF_CUES_EN = ("default", "prefer", "always", "instead of", "from now on")

# Discard / ephemeral — P3
DISCARD_CUES_ZH = ("临时", "先放", "随便", "一次性", "这次先")
DISCARD_CUES_EN = ("temporary", "tmp", "scratch", "for now", "one-off", "disposable")

DISCARD_PATH_PREFIXES = (
    "/tmp",
    "/var/folders",
    "/private/var/folders",
    "/dev/",
)
DISCARD_PATH_PARTS = ("/scratch/", "/tmp/", "AppData/Local/Temp")

# Hint words → key namespace (bilingual synonyms → English key stem)
KEY_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("cub", "caltech-ucsd birds", "鸟类数据集"), "dataset.cub.root"),
    (("pretrained", "预训练", "预训练模型"), "path.pretrained.root"),
    (("autodl-tmp", "autodl", "训练输出", "output"), "path.train_output.root"),
    (("models", "模型库", "权重库"), "path.models.root"),
    (("h1", "shunyao-h1", "外接盘"), "disk.h1.root"),
    (("4090", "4090d", "autodl 4090"), "host.4090d.data_dir"),
)


def locale_hint(text: str) -> str:
    """Rough locale: zh / en / mixed."""
    s = text or ""
    has_zh = any("\u4e00" <= ch <= "\u9fff" for ch in s)
    has_latin = any("a" <= ch.lower() <= "z" for ch in s if ch.isalpha())
    if has_zh and has_latin:
        return "mixed"
    if has_zh:
        return "zh"
    if has_latin:
        return "en"
    return "mixed"


def _lower(text: str) -> str:
    return (text or "").lower()


def has_any_cue(text: str, cues: tuple[str, ...]) -> bool:
    """Match cues; short ASCII tokens use word boundaries to avoid 'root' in 'canonical root' noise."""
    import re

    blob = _lower(text)
    for c in cues:
        cl = c.lower()
        if len(cl) <= 4 and cl.isascii() and cl.isalpha():
            if re.search(rf"\b{re.escape(cl)}\b", blob):
                return True
        elif cl in blob:
            return True
    return False


def location_cued(text: str) -> bool:
    return has_any_cue(text, LOCATION_CUES_ZH) or has_any_cue(text, LOCATION_CUES_EN)


def remember_cued(text: str) -> bool:
    return has_any_cue(text, REMEMBER_CUES_ZH) or has_any_cue(text, REMEMBER_CUES_EN)


def preference_cued(text: str) -> bool:
    return has_any_cue(text, PREF_CUES_ZH) or has_any_cue(text, PREF_CUES_EN)


def discard_cued(text: str) -> bool:
    return has_any_cue(text, DISCARD_CUES_ZH) or has_any_cue(text, DISCARD_CUES_EN)


def path_looks_ephemeral(path: str) -> bool:
    p = path.replace("\\", "/")
    low = p.lower()
    if any(low.startswith(pref.lower()) for pref in DISCARD_PATH_PREFIXES):
        return True
    return any(part.lower() in low for part in DISCARD_PATH_PARTS)
