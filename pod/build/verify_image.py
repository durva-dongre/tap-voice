import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

MIN_VOCAB = 156938
APP_DIR = "/app"
MODELS_ROOT = os.environ.get("MODELS_DIR", "/opt/models")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models/svara-fp8")
SNAC_DIR = os.environ.get("SNAC_DIR", "/opt/models/snac_24khz")
SNAC_WEIGHTS = ("pytorch_model.bin", "model.pt", "snac.pt")
MODULES = [
    "app.prompt",
    "app.voices",
    "app.manifest",
    "app.codec",
    "app.audio",
    "app.logits",
    "app.storage",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model(errors):
    if not os.path.isdir(MODEL_DIR):
        errors.append(f"model dir missing: {MODEL_DIR}")
        return
    names = os.listdir(MODEL_DIR)
    if not any(name.endswith(".safetensors") for name in names):
        errors.append("model has no .safetensors file")
    if "tokenizer.json" not in names and "tokenizer.model" not in names:
        errors.append("model has no tokenizer.json or tokenizer.model")
    if "tokenizer_config.json" not in names:
        errors.append("model has no tokenizer_config.json")
    config_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.isfile(config_path):
        errors.append("model config.json missing")
    else:
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception as exc:
            errors.append(f"model config.json unparsable: {type(exc).__name__}")
            config = None
        if isinstance(config, dict):
            size = config.get("vocab_size")
            if not isinstance(size, int) or size < MIN_VOCAB:
                errors.append(f"config vocab_size {size} below {MIN_VOCAB}")
            if "quantization_config" not in config:
                sys.stdout.write("note: no quantization_config, engine will quantize on load\n")
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    except Exception as exc:
        errors.append(f"tokenizer failed to load: {type(exc).__name__}: {exc}")


def check_snac(errors):
    if not os.path.isfile(os.path.join(SNAC_DIR, "config.json")):
        errors.append("snac config.json missing")
    if not any(os.path.isfile(os.path.join(SNAC_DIR, name)) for name in SNAC_WEIGHTS):
        errors.append("snac weights missing")


def locate(bases, rel):
    for base in bases:
        path = os.path.join(base, rel)
        if os.path.isfile(path):
            return path
    return None


def check_manifest(errors):
    path = os.path.join(MODELS_ROOT, "weights_manifest.json")
    if not os.path.isfile(path):
        errors.append("weights_manifest.json missing")
        return
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        errors.append(f"weights_manifest.json unparsable: {type(exc).__name__}")
        return
    targets = {"snac": SNAC_DIR, "svara_fp8": MODEL_DIR}
    for name, directory in targets.items():
        entry = manifest.get(name) if isinstance(manifest, dict) else None
        if not isinstance(entry, dict) or not entry.get("files"):
            errors.append(f"manifest entry missing or empty: {name}")
            continue
        bases = [directory, os.path.dirname(directory)]
        for rel, meta in entry["files"].items():
            found = locate(bases, rel)
            if found is None:
                errors.append(f"manifest file missing: {name}/{rel}")
                continue
            if os.path.getsize(found) != meta.get("bytes"):
                errors.append(f"size mismatch: {name}/{rel}")
                continue
            if sha256_file(found) != meta.get("sha256"):
                errors.append(f"sha256 mismatch: {name}/{rel}")


def check_tree(errors):
    for base, dirs, names in os.walk(MODELS_ROOT):
        for name in dirs + names:
            if os.path.islink(os.path.join(base, name)):
                errors.append(f"symlink found: {os.path.join(base, name)}")
    for root in (MODELS_ROOT, APP_DIR):
        for base, dirs, names in os.walk(root):
            for name in dirs:
                path = os.path.join(base, name)
                if not os.path.islink(path) and not os.access(path, os.R_OK | os.X_OK):
                    errors.append(f"directory not accessible: {path}")
            for name in names:
                path = os.path.join(base, name)
                if not os.path.islink(path) and not os.access(path, os.R_OK):
                    errors.append(f"file not readable: {path}")
    try:
        with tempfile.NamedTemporaryFile(dir="/tmp"):
            pass
    except Exception as exc:
        errors.append(f"/tmp not writable: {type(exc).__name__}")


def check_entrypoint(errors):
    path = os.path.join(APP_DIR, "entrypoint.sh")
    if not os.path.isfile(path):
        errors.append("entrypoint.sh missing")
    elif not os.access(path, os.X_OK):
        errors.append("entrypoint.sh not executable")


def check_media(errors):
    if shutil.which("ffmpeg") is None:
        errors.append("ffmpeg not on PATH")
    else:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
            if b"libopus" not in result.stdout:
                errors.append("ffmpeg lacks libopus encoder")
        except Exception as exc:
            errors.append(f"ffmpeg probe failed: {type(exc).__name__}")
    for name in ("numpy", "soxr", "soundfile"):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"import {name} failed: {type(exc).__name__}: {exc}")
            continue
        if name == "soundfile":
            try:
                if not module.__libsndfile_version__:
                    errors.append("libsndfile version empty")
            except Exception as exc:
                errors.append(f"libsndfile check failed: {type(exc).__name__}")


def check_modules(errors):
    loaded = {}
    for name in MODULES:
        try:
            loaded[name] = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"import {name} failed: {type(exc).__name__}: {exc}")
    manifest_mod = loaded.get("app.manifest")
    prompt = loaded.get("app.prompt")
    try:
        if manifest_mod is not None:
            cleaned = manifest_mod.clean_text("  Hello \u200b  world \n")
            if not isinstance(cleaned, str) or "Hello" not in cleaned:
                errors.append("manifest.clean_text returned unexpected output")
        if prompt is not None:
            tagged = prompt.text_with_emotion("Hello", "happy")
            if not tagged.startswith("Hello") or not tagged.endswith("<happy>"):
                errors.append("prompt.text_with_emotion returned unexpected output")
            if prompt.text_with_emotion("Hello", None) != "Hello":
                errors.append("prompt.text_with_emotion mishandled empty emotion")
    except Exception as exc:
        errors.append(f"helper sanity call failed: {type(exc).__name__}: {exc}")


def main():
    errors = []
    for check in (
        check_model,
        check_snac,
        check_manifest,
        check_tree,
        check_entrypoint,
        check_media,
        check_modules,
    ):
        try:
            check(errors)
        except Exception as exc:
            errors.append(f"{check.__name__} crashed: {type(exc).__name__}: {exc}")
    if errors:
        sys.stderr.write("verify_image failed:\n")
        for message in errors:
            sys.stderr.write(f"- {message}\n")
        return 1
    sys.stdout.write("verify_image ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())