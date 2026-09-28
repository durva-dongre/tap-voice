import hashlib
import json
import os
import shutil
import sys

SMALL = ["*.json", "tokenizer*", "special_tokens_map.json", "*.model", "*.txt"]


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


def describe(directory):
    files = {}
    for base, _, names in os.walk(directory):
        for name in names:
            path = os.path.join(base, name)
            if os.path.islink(path):
                raise RuntimeError(f"symlink not allowed: {path}")
            rel = os.path.relpath(path, directory)
            files[rel] = {"sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    return dict(sorted(files.items()))


def fetch(repo, revision, dest, patterns):
    from huggingface_hub import HfApi, snapshot_download

    token = env("HF_TOKEN") or None
    sha = HfApi(token=token).model_info(repo, revision=revision).sha
    if not sha:
        raise RuntimeError(f"could not resolve {repo}@{revision}")
    shutil.rmtree(dest, ignore_errors=True)
    snapshot_download(
        repo_id=repo, revision=sha, local_dir=dest, allow_patterns=patterns, token=token
    )
    shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)
    if not os.path.isfile(os.path.join(dest, "config.json")):
        raise RuntimeError(f"config.json missing after downloading {repo}")
    return {"repo": repo, "revision": sha, "files": describe(dest)}


def main():
    models_dir = env("MODELS_DIR", "/opt/models")
    raw_dir = env("RAW_DIR", "/opt/raw")
    svara_repo = env("SVARA_REPO", "kenpath/svara-tts-v1")
    snac_repo = env("SNAC_REPO", "hubertsiuzdak/snac_24khz")
    svara_patterns = list(SMALL)
    if not env("PREQUANTIZED_REPO"):
        svara_patterns.append("*.safetensors")
    try:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        svara = fetch(
            svara_repo,
            env("SVARA_REVISION", "main"),
            os.path.join(raw_dir, "svara-bf16"),
            svara_patterns,
        )
        snac = fetch(
            snac_repo,
            env("SNAC_REVISION", "main"),
            os.path.join(models_dir, "snac_24khz"),
            ["config.json", "pytorch_model.bin"],
        )
        with open(os.path.join(models_dir, "weights_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"svara": svara, "snac": snac}, handle, indent=2, sort_keys=True)
    except Exception as exc:
        sys.stderr.write(f"fetch_weights failed: {type(exc).__name__}: {redact(exc)}\n")
        return 1
    sys.stdout.write(
        f"fetch_weights ok: svara={svara['revision'][:12]} snac={snac['revision'][:12]}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())