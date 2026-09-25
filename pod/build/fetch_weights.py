import os
import sys
from huggingface_hub import snapshot_download

def clean_token(raw):
    if raw is None:
        return None
    t = raw.strip()
    return t or None

token = clean_token(os.environ.get("HF_TOKEN"))
kokoro_rev = os.environ.get("KOKORO_REV") or None
indic_rev = os.environ.get("INDIC_REV") or None

print(f"HF_TOKEN present: {token is not None}", flush=True)
if token:
    print(f"HF_TOKEN length: {len(token)} prefix: {token[:6]}... suffix: ...{token[-4:]}", flush=True)
else:
    print("WARNING: HF_TOKEN is empty or missing after cleaning", flush=True)

print("Fetching Kokoro weights...", flush=True)
snapshot_download(
    repo_id="hexgrad/Kokoro-82M",
    revision=kokoro_rev,
    token=token,
)

print("Fetching Indic Parler-TTS weights...", flush=True)
try:
    snapshot_download(
        repo_id="ai4bharat/indic-parler-tts",
        revision=indic_rev,
        token=token,
    )
except Exception as e:
    print(f"FAILED to fetch ai4bharat/indic-parler-tts: {e}", file=sys.stderr, flush=True)
    print(f"token was present: {token is not None}, length: {len(token) if token else 0}", file=sys.stderr, flush=True)
    raise

print("All weights fetched successfully.", flush=True)