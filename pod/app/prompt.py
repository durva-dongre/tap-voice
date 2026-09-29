import re
from dataclasses import dataclass
from typing import List

SOH = 128259
EOT = 128009
EOH = 128260
EOS = 128258
START_OF_SPEECH = 128257
START_OF_AI = 128261
# Orpheus-style models normally emit these before the first audio token. The codec drops them.
HEADER_TOKENS = (START_OF_AI, START_OF_SPEECH)
AUDIO_OFFSET = 128266
AUDIO_END = AUDIO_OFFSET + 7 * 4096
DEFAULT_BOS = 128000

CONTEXT_SLACK = 16

TAG_END = re.compile(r"<[a-z]+>\s*$")

_ID_CACHE = {}


@dataclass(frozen=True)
class Prepared:
    item: object
    prompt_ids: List[int]
    max_tokens: int
    seed: int
    room: int = 0


def text_with_emotion(text, emotion):
    if not emotion or TAG_END.search(text):
        return text
    return f"{text} <{emotion}>"


def bos_id(tokenizer):
    value = getattr(tokenizer, "bos_token_id", None)
    return value if value is not None else DEFAULT_BOS


def body_for(voice, text, emotion):
    return f"{voice}: {text_with_emotion(text, emotion)}"


def build_ids(tokenizer, voice, text, emotion, bos):
    encoded = tokenizer.encode(body_for(voice, text, emotion), add_special_tokens=False)
    return [SOH, bos] + list(encoded) + [EOT, EOH]


def estimate_max_tokens(text, settings):
    estimate = int(len(text) * settings.tokens_per_char) + 140
    return max(settings.min_max_tokens, min(settings.max_max_tokens, estimate))


def attempt_cap(text, room, settings, attempt):
    """Token cap per attempt.

    First pass: the length estimate (a rambling clip is cut off cheaply).
    Second pass: 1.5x the estimate, so a clip that was only slightly short still gets through.
    Later passes: everything the context allows. A clip that loops forever therefore burns
    the full window at most once, not on every retry.
    """
    estimate = estimate_max_tokens(text, settings)
    if attempt <= 0:
        cap = estimate
    elif attempt == 1:
        cap = int(estimate * settings.retry_growth)
    else:
        cap = room
    return max(1, min(room, cap))


def seed_from_hash(content_hash, base_seed, attempt):
    return (int(content_hash[:12], 16) + base_seed + attempt * 7919) % (2**31 - 1)


def _prompt_ids(items, tokenizer):
    bos = bos_id(tokenizer)
    missing = [i for i in items if i.content_hash not in _ID_CACHE]
    if missing:
        bodies = [body_for(i.voice, i.text, i.emotion) for i in missing]
        encoded = tokenizer(bodies, add_special_tokens=False)["input_ids"]
        for item, ids in zip(missing, encoded):
            _ID_CACHE[item.content_hash] = [SOH, bos] + list(ids) + [EOT, EOH]
    return [_ID_CACHE[i.content_hash] for i in items]


def prepare(items, tokenizer, settings, attempt=0):
    """Build prompts. max_tokens never exceeds the space left in the context window."""
    ids_list = _prompt_ids(items, tokenizer)
    out = []
    for item, ids in zip(items, ids_list):
        room = settings.max_model_len - len(ids) - CONTEXT_SLACK
        out.append(
            Prepared(
                item=item,
                prompt_ids=ids,
                max_tokens=attempt_cap(item.text, room, settings, attempt),
                seed=seed_from_hash(item.content_hash, settings.base_seed, attempt),
                room=room,
            )
        )
    return out


def fits_context(prepared, settings):
    """True if the context window can hold the audio this text is expected to need."""
    needed = int(len(prepared.item.text) * settings.min_tokens_per_char)
    return prepared.room >= needed


def sort_by_length(prepared, longest_first=True):
    # Longest first keeps the GPU full to the end instead of leaving long clips for the tail.
    return sorted(prepared, key=lambda p: p.max_tokens, reverse=longest_first)


def clear_cache():
    _ID_CACHE.clear()