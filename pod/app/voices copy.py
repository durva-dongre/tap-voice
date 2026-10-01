LANGUAGE_VOICES = {
    "english": "English (Female)",
    "hindi": "Hindi (Female)",
    "marathi": "Marathi (Female)",
    "kannada": "Kannada (Female)",
    "punjabi": "Punjabi (Female)",
}

LANGUAGE_ALIASES = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
    "kn": "kannada",
    "pa": "punjabi",
    "english": "english",
    "hindi": "hindi",
    "marathi": "marathi",
    "kannada": "kannada",
    "punjabi": "punjabi",
}

ALLOWED_VOICES = frozenset(LANGUAGE_VOICES.values())


def normalize_language(value):
    if not isinstance(value, str):
        return None
    return LANGUAGE_ALIASES.get(value.strip().lower())


def default_voice(language):
    return LANGUAGE_VOICES.get(language)


def resolve_voice(language, voice):
    lang = normalize_language(language)
    if lang is None:
        return None, None
    if voice is None or voice == "":
        return lang, LANGUAGE_VOICES[lang]
    if not isinstance(voice, str) or voice not in ALLOWED_VOICES:
        return lang, None
    return lang, voice