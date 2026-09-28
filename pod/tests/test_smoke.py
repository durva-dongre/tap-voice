from types import SimpleNamespace

import numpy as np

from app import audio, manifest, prompt, voices
from app.codec import BAND, clean_tokens, tokens_to_layers, valid_codes


def make_settings():
    return SimpleNamespace(max_items=10, max_text_chars=600, model_revision="rev")


def test_resolve_voice_default():
    assert voices.resolve_voice("hi", None) == ("hindi", "Hindi (Female)")


def test_resolve_voice_unknown_language():
    assert voices.resolve_voice("xx", None) == (None, None)


def test_clean_text_strips_control_chars():
    assert manifest.clean_text("  a\u200b  b\n") == "a b"


def test_parse_rejects_duplicate_ids():
    raw = [
        {"id": "a", "text": "Hello", "language": "en"},
        {"id": "a", "text": "Hi", "language": "en"},
    ]
    accepted, rejected = manifest.parse(raw, make_settings())
    assert [item.id for item in accepted] == ["a"]
    assert rejected[0].reason == "duplicate_id"


def test_text_with_emotion():
    assert prompt.text_with_emotion("Hi", "happy") == "Hi <happy>"
    assert prompt.text_with_emotion("Hi <sad>", "happy") == "Hi <sad>"
    assert prompt.text_with_emotion("Hi", None) == "Hi"


def test_seed_is_deterministic():
    digest = "a" * 64
    first = prompt.seed_from_hash(digest, 1234, 0)
    assert first == prompt.seed_from_hash(digest, 1234, 0)
    assert first != prompt.seed_from_hash(digest, 1234, 1)


def test_tokens_to_layers():
    codes = [10, 11, 12, 13, 14, 15, 16]
    tokens = [prompt.AUDIO_OFFSET + slot * BAND + code for slot, code in enumerate(codes)]
    layer0, layer1, layer2 = tokens_to_layers(clean_tokens(tokens))
    assert layer0.tolist() == [10]
    assert layer1.tolist() == [11, 14]
    assert layer2.tolist() == [12, 13, 15, 16]
    assert valid_codes((layer0, layer1, layer2))


def test_clean_tokens_drops_partial_frame():
    assert clean_tokens([prompt.AUDIO_OFFSET] * 9).size == 7


def test_peak_normalize():
    out = audio.peak_normalize(np.array([0.5, -0.25], dtype=np.float32), 0.9)
    assert abs(float(np.max(np.abs(out))) - 0.9) < 1e-5


def test_bitrate_bps():
    assert audio.bitrate_bps("24k") == 24000
