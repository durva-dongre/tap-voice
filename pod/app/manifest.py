import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Optional

from . import voices

EMOTIONS = frozenset(
    {
        "happy",
        "sad",
        "angry",
        "clear",
        "fearful",
        "surprise",
        "disgust",
        "laugh",
        "chuckle",
        "sigh",
        "cough",
        "sniffle",
        "groan",
        "yawn",
        "gasp",
    }
)

FORMATS = frozenset({"ogg", "wav"})


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    language: str
    voice: str
    emotion: Optional[str]
    fmt: str
    content_hash: str


@dataclass(frozen=True)
class Rejected:
    id: str
    reason: str


def clean_text(text):
    normalized = unicodedata.normalize("NFC", text)
    out = []
    for ch in normalized:
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf", "Cs", "Co", "Cn") and ch not in ("\n", "\t"):
            continue
        out.append(ch)
    collapsed = "".join(out).replace("\t", " ").replace("\n", " ")
    return " ".join(collapsed.split())


def content_hash(text, voice, emotion, fmt, model_revision):
    parts = [
        clean_text(text),
        voice,
        emotion or "",
        fmt,
        model_revision,
    ]
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse(raw_items, settings):
    accepted = []
    rejected = []
    seen = set()
    if not isinstance(raw_items, list):
        return accepted, rejected
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            rejected.append(Rejected(id=f"index:{index}", reason="not_an_object"))
            continue
        item_id = raw.get("id")
        if not isinstance(item_id, (str, int)) or str(item_id) == "":
            rejected.append(Rejected(id=f"index:{index}", reason="bad_id"))
            continue
        item_id = str(item_id)
        if len(item_id) > 128:
            rejected.append(Rejected(id=item_id[:32], reason="id_too_long"))
            continue
        if item_id in seen:
            rejected.append(Rejected(id=item_id, reason="duplicate_id"))
            continue
        seen.add(item_id)
        if len(accepted) + len(rejected) > settings.max_items * 2:
            rejected.append(Rejected(id=item_id, reason="batch_too_large"))
            continue
        text = raw.get("text")
        if not isinstance(text, str):
            rejected.append(Rejected(id=item_id, reason="bad_text"))
            continue
        cleaned = clean_text(text)
        if not cleaned:
            rejected.append(Rejected(id=item_id, reason="empty_text"))
            continue
        if len(cleaned) > settings.max_text_chars:
            rejected.append(Rejected(id=item_id, reason="text_too_long"))
            continue
        lang, voice = voices.resolve_voice(raw.get("language"), raw.get("voice"))
        if lang is None:
            rejected.append(Rejected(id=item_id, reason="unsupported_language"))
            continue
        if voice is None:
            rejected.append(Rejected(id=item_id, reason="voice_not_allowed"))
            continue
        emotion = raw.get("emotion")
        if emotion is not None and emotion != "":
            if not isinstance(emotion, str):
                rejected.append(Rejected(id=item_id, reason="bad_emotion"))
                continue
            emotion = emotion.strip().lower().strip("<>")
            if emotion not in EMOTIONS:
                rejected.append(Rejected(id=item_id, reason="emotion_not_allowed"))
                continue
        else:
            emotion = None
        fmt = raw.get("format") or "ogg"
        if not isinstance(fmt, str) or fmt.lower() not in FORMATS:
            rejected.append(Rejected(id=item_id, reason="format_not_allowed"))
            continue
        fmt = fmt.lower()
        if len(accepted) >= settings.max_items:
            rejected.append(Rejected(id=item_id, reason="batch_too_large"))
            continue
        digest = content_hash(cleaned, voice, emotion, fmt, settings.model_revision)
        accepted.append(
            Item(
                id=item_id,
                text=cleaned,
                language=lang,
                voice=voice,
                emotion=emotion,
                fmt=fmt,
                content_hash=digest,
            )
        )
    return accepted, rejected