import re
from dataclasses import dataclass
from typing import List

SOH = 128259
EOT = 128009
EOH = 128260
EOS = 128258
AUDIO_OFFSET = 128266
AUDIO_END = AUDIO_OFFSET + 7 * 4096
DEFAULT_BOS = 128000

TAG_END = re.compile(r"<[a-z]+>\s*$")

_ID_CACHE = {}


@dataclass(frozen=True)
class Prepared:
    item: object
    prompt_ids: List[int]
    max_tokens: int
    seed: int


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
    ids_list = _prompt_ids(items, tokenizer)
    return [
        Prepared(
            item=item,
            prompt_ids=ids,
            max_tokens=estimate_max_tokens(item.text, settings),
            seed=seed_from_hash(item.content_hash, settings.base_seed, attempt),
        )
        for item, ids in zip(items, ids_list)
    ]


def sort_by_length(prepared):
    return sorted(prepared, key=lambda p: p.max_tokens)


def clear_cache():
    _ID_CACHE.clear()