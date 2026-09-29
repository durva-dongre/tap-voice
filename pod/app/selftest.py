import asyncio
import os
import sys

import requests

from . import audio, config
from .codec import Codec
from .engine import Engine
from .manifest import Item, content_hash
from .prompt import fits_context, prepare
from .storage import Store
from .voices import LANGUAGE_VOICES

SAMPLES = {
    "english": "Hello, this is a short self test of the speech pipeline.",
    "hindi": "नमस्ते, यह एक छोटा परीक्षण है।",
    "marathi": "नमस्कार, ही एक छोटी चाचणी आहे.",
    "kannada": "ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಸಣ್ಣ ಪರೀಕ್ಷೆ.",
    "punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਇੱਕ ਛੋਟਾ ਟੈਸਟ ਹੈ।",
}


def make_item(settings, item_id, language, text):
    voice = LANGUAGE_VOICES[language]
    return Item(
        id=item_id,
        text=text,
        language=language,
        voice=voice,
        emotion=None,
        fmt="ogg",
        content_hash=content_hash(text, voice, None, "ogg", settings.model_revision),
    )


def build_items(settings):
    return [make_item(settings, f"selftest-{lang}", lang, text) for lang, text in SAMPLES.items()]


def root_cause(exc):
    """Innermost exception in the chain, flattened to one line."""
    while exc.__cause__ is not None:
        exc = exc.__cause__
    text = f"{type(exc).__name__}: {exc}"
    return " ".join(text.replace(",", ";").split())[:240]


def context_failures(settings, tokenizer):
    """Tokenizer only, no GPU: a maximum-length text in every language must leave enough
    context for its audio. Otherwise those items would fail for good in production."""
    failures = []
    for language, text in SAMPLES.items():
        long_text = " ".join([text] * 40)[: settings.max_text_chars]
        item = make_item(settings, f"context-{language}", language, long_text)
        prepared = prepare([item], tokenizer, settings)[0]
        if not fits_context(prepared, settings):
            failures.append(f"context_{language}:room_{prepared.room}")
    return failures


def check_upload(store, item, data, content_type, ext):
    name = f"selftest/{item.language}/{item.content_hash[:32]}.{ext}"
    url = store.upload(name, data, content_type)
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return f"{item.id}:http_{response.status_code}"
    if "audio/ogg" not in response.headers.get("Content-Type", ""):
        return f"{item.id}:bad_content_type"
    return None


async def run(settings):
    failures = []
    store = Store(settings)
    try:
        store.probe_list()
    except Exception as exc:
        return [f"gcs_list_failed:{type(exc).__name__}"]
    engine = Engine(settings)
    try:
        await engine.start()
        sys.stdout.write(f"selftest_info kv_cache_dtype={engine.kv_dtype} extras={sorted(engine.extras)}\n")
        codec = Codec(
            settings.snac_dir, half=settings.snac_half, tolerance=settings.bad_code_tolerance
        )
        codec.warmup()
        failures += context_failures(settings, engine.tokenizer)
        items = build_items(settings)
        prepared = prepare(items, engine.tokenizer, settings)
        by_id = {p.item.id: p.item for p in prepared}
        finished = []
        async for result in engine.generate(prepared, lambda: False):
            finished.append(result)
        good = []
        for f in finished:
            if f.error:
                failures.append(f"{f.key}:{f.error}")
            elif f.finish_reason == "length":
                failures.append(f"{f.key}:truncated")
            else:
                good.append((f.key, f.tokens))
        decoded, bad = codec.decode_batch(good)
        failures += [f"{k}:{r}" for k, r in bad]
        for key, wave in decoded:
            item = by_id[key]
            processed, reason = audio.process(wave, settings, len(item.text))
            if processed is None:
                failures.append(f"{key}:{reason}")
                continue
            data, content_type, ext = audio.encode(processed, "ogg", settings)
            failure = check_upload(store, item, data, content_type, ext)
            if failure:
                failures.append(failure)
    except Exception as exc:
        failures.append(f"engine:{root_cause(exc)}")
    finally:
        await engine.shutdown()
    return failures


def main():
    settings = config.get()
    failures = asyncio.run(run(settings))
    if failures:
        sys.stderr.write("selftest_failed " + ",".join(failures) + "\n")
        return 1
    sys.stdout.write("selftest_ok\n")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)