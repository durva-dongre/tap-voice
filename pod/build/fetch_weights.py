import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    repo_id="hexgrad/Kokoro-82M",
    revision=os.environ.get("KOKORO_REV") or None,
    token=token,
)

snapshot_download(
    repo_id="ai4bharat/indic-parler-tts",
    revision=os.environ.get("INDIC_REV") or None,
    token=token,
)

snapshot_download(
    repo_id="google/flan-t5-large",
    revision=os.environ.get("FLAN_T5_REV") or None,
    token=token,
)