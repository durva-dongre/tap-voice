import fnmatch
import hashlib
import json
import os
import shutil
import sys

MIN_VOCAB = 156938
ALLOW = ["*.json", "*.safetensors", "tokenizer*", "special_tokens_map.json", "*.model", "*.txt"]
TOKENIZER_NAMES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]
CALIBRATION_TEXTS = [
    "English (Female): Hello, this is a short calibration sentence for the speech model.",
    "English (Female): The quick brown fox jumps over the lazy dog <happy>",
    "Hindi (Female): नमस्ते, यह एक छोटा परीक्षण है।",
    "Marathi (Female): नमस्कार, ही एक छोटी चाचणी आहे.",
    "Kannada (Female): ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಸಣ್ಣ ಪರೀಕ್ಷೆ.",
    "Punjabi (Female): ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਇੱਕ ਛੋਟਾ ਟੈਸਟ ਹੈ।",
    "English (Female): Please hold on for a moment while I check that for you <sigh>",
    "English (Female): Numbers like 2024 and dates like March 5th should be read clearly.",
]


class BuildError(Exception):
    pass


def env(name, default=""):
    return os.environ.get(name, "").strip() or default


def redact(text):
    token = env("HF_TOKEN")
    text = str(text)
    return text.replace(token, "***") if token else text


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_files(root):
    found = []
    for base, _, names in os.walk(root):
        for name in names:
            found.append(os.path.relpath(os.path.join(base, name), root))
    return sorted(found)


def allowed(name):
    return any(fnmatch.fnmatch(name, pattern) for pattern in ALLOW)


def tokenizer_complete(directory):
    has_vocab = os.path.isfile(os.path.join(directory, "tokenizer.json")) or os.path.isfile(
        os.path.join(directory, "tokenizer.model")
    )
    return has_vocab and os.path.isfile(os.path.join(directory, "tokenizer_config.json"))


def ensure_tokenizer(dest, raw):
    if tokenizer_complete(dest):
        return
    if not os.path.isdir(raw):
        raise BuildError("tokenizer files missing and raw svara-bf16 is unavailable")
    for name in TOKENIZER_NAMES:
        source = os.path.join(raw, name)
        target = os.path.join(dest, name)
        if os.path.isfile(source) and not os.path.exists(target):
            shutil.copy2(source, target)
    if not tokenizer_complete(dest):
        raise BuildError("tokenizer files still missing after copying from raw model")


def require_model_files(dest):
    if not os.path.isfile(os.path.join(dest, "config.json")):
        raise BuildError("config.json missing in quantized model directory")
    if not any(name.endswith(".safetensors") for name in os.listdir(dest)):
        raise BuildError("no .safetensors file in quantized model directory")


def fetch_prequantized(repo, dest, raw):
    from huggingface_hub import HfApi, snapshot_download

    token = env("HF_TOKEN") or None
    info = HfApi(token=token).model_info(
        repo, revision=env("PREQUANTIZED_REVISION", "main"), files_metadata=True
    )
    sha = info.sha
    if not sha:
        raise BuildError("could not resolve prequantized revision to a commit sha")
    shutil.rmtree(dest, ignore_errors=True)
    snapshot_download(
        repo_id=repo, revision=sha, local_dir=dest, allow_patterns=ALLOW, token=token
    )
    shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)
    for sibling in info.siblings:
        if not allowed(sibling.rfilename):
            continue
        path = os.path.join(dest, sibling.rfilename)
        if not os.path.isfile(path):
            raise BuildError(f"downloaded file missing: {sibling.rfilename}")
        if sibling.size is not None and os.path.getsize(path) != sibling.size:
            raise BuildError(f"size mismatch for {sibling.rfilename}")
    require_model_files(dest)
    ensure_tokenizer(dest, raw)
    return sha


def calibration_args(tokenizer):
    from datasets import Dataset

    data = Dataset.from_dict({"text": CALIBRATION_TEXTS})
    data = data.map(
        lambda row: tokenizer(
            row["text"], add_special_tokens=False, truncation=True, max_length=256
        ),
        remove_columns=["text"],
    )
    return {
        "dataset": data,
        "max_seq_length": 256,
        "num_calibration_samples": len(CALIBRATION_TEXTS),
    }


def quantize_local(raw, dest):
    if not os.path.isfile(os.path.join(raw, "config.json")):
        raise BuildError(f"raw BF16 model not found at {raw}")
    import torch

    if not torch.cuda.is_available():
        raise BuildError(
            "no CUDA GPU available: build on a GPU host or set PREQUANTIZED_REPO"
        )
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor import oneshot
    except ImportError:
        from llmcompressor.transformers import oneshot

    tokenizer = AutoTokenizer.from_pretrained(raw, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        raw, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head", "re:.*embed_tokens.*"],
    )
    extra = calibration_args(tokenizer) if env("CALIBRATE", "0") == "1" else {}
    oneshot(model=model, recipe=recipe, **extra)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    model.save_pretrained(dest, save_compressed=True)
    tokenizer.save_pretrained(dest)
    require_model_files(dest)
    ensure_tokenizer(dest, raw)


def check_vocab(dest):
    with open(os.path.join(dest, "config.json"), encoding="utf-8") as handle:
        size = json.load(handle).get("vocab_size")
    if not isinstance(size, int) or size < MIN_VOCAB:
        raise BuildError(f"config vocab_size {size} is below {MIN_VOCAB}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(dest, local_files_only=True)
    if len(tokenizer) < MIN_VOCAB:
        raise BuildError(f"tokenizer size {len(tokenizer)} is below {MIN_VOCAB}")
    return size


def warn_if_unquantized(dest):
    with open(os.path.join(dest, "config.json"), encoding="utf-8") as handle:
        declared = json.load(handle).get("quantization_config")
    if not declared:
        sys.stderr.write(
            "quantize_fp8 warning: checkpoint has no quantization_config; "
            "weights are full precision on disk and vLLM will quantize at load\n"
        )


def record_manifest(models_dir, dest, source, revision):
    files = {}
    for rel in walk_files(dest):
        path = os.path.join(dest, rel)
        if os.path.islink(path):
            raise BuildError(f"symlink not allowed: {rel}")
        files[rel] = {"sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    path = os.path.join(models_dir, "weights_manifest.json")
    manifest = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    manifest["svara_fp8"] = {"repo": source, "revision": revision, "files": files}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return files


def main():
    models_dir = env("MODELS_DIR", "/opt/models")
    raw = os.path.join(env("RAW_DIR", "/opt/raw"), "svara-bf16")
    dest = os.path.join(models_dir, "svara-fp8")
    repo = env("PREQUANTIZED_REPO")
    try:
        if repo:
            source = repo
            revision = fetch_prequantized(repo, dest, raw)
        else:
            source = "local"
            quantize_local(raw, dest)
            revision = env("SVARA_REVISION", "local")
        vocab = check_vocab(dest)
        warn_if_unquantized(dest)
        files = record_manifest(models_dir, dest, source, revision)
    except BuildError as exc:
        sys.stderr.write(f"quantize_fp8 failed: {redact(exc)}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"quantize_fp8 failed: {type(exc).__name__}: {redact(exc)}\n")
        return 1
    sys.stdout.write(
        f"quantize_fp8 ok: source={source} revision={revision[:12]} "
        f"files={len(files)} vocab_size={vocab}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())