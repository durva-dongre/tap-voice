import asyncio
import sys

import requests

from . import audio, config
from .codec import Codec
from .engine import Engine
from .manifest import Item, content_hash
from .prompt import prepare
from .storage import Store
from .voices import LANGUAGE_VOICES

SAMPLES = {
    "english": "Hello, this is a short self test of the speech pipeline.",
    "hindi": "नमस्ते, यह एक छोटा परीक्षण है।",
    "marathi": "नमस्कार, ही एक छोटी चाचणी आहे.",
    "kannada": "ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಸಣ್ಣ ಪರೀಕ್ಷೆ.",
    "punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਇੱਕ ਛੋਟਾ ਟੈਸਟ ਹੈ।",
}


def build_items(settings):
    items = []
    for language, text in SAMPLES.items():
        voice = LANGUAGE_VOICES[language]
        items.append(
            Item(
                id=f"selftest-{language}",
                text=text,
                language=language,
                voice=voice,
                emotion=None,
                fmt="ogg",
                content_hash=content_hash(text, voice, None, "ogg", settings.model_revision),
            )
        )
    return items


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
    engine = Engine(settings)
    await engine.start()
    codec = Codec(settings.snac_dir)
    store = Store(settings)
    items = build_items(settings)
    prepared = prepare(items, engine.tokenizer, settings)
    by_id = {p.item.id: p.item for p in prepared}
    finished = []
    async for result in engine.generate(prepared, lambda: False):
        finished.append(result)
    entries = [(f.key, f.tokens) for f in finished if not f.error]
    decoded, bad = codec.decode_batch(entries)
    failures = [f"{k}:{r}" for k, r in bad]
    failures += [f"{f.key}:{f.error}" for f in finished if f.error]
    for key, wave in decoded:
        item = by_id[key]
        processed, reason = audio.process(wave, settings)
        if processed is None:
            failures.append(f"{key}:{reason}")
            continue
        data, content_type, ext = audio.encode(processed, "ogg", settings)
        failure = check_upload(store, item, data, content_type, ext)
        if failure:
            failures.append(failure)
    await engine.shutdown()
    return failures


def main():
    settings = config.get()
    failures = asyncio.run(run(settings))
    if failures:
        sys.stderr.write("selftest_failed " + ",".join(failures) + "\n")
        sys.exit(1)
    sys.stdout.write("selftest_ok\n")


if __name__ == "__main__":
    main()