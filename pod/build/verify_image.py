import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models/svara-fp8")
SNAC_DIR = os.environ.get("SNAC_DIR", "/opt/models/snac_24khz")
FFMPEG = os.environ.get("FFMPEG_BIN") or "ffmpeg"

problems = []
notes = []


def check(name, fn):
    try:
        detail = fn()
        notes.append(f"ok    {name}" + (f": {detail}" if detail else ""))
    except Exception as exc:
        problems.append(f"{name}: {type(exc).__name__}: {exc}")


def has_weights(path):
    return any(n.endswith((".safetensors", ".bin", ".pt")) for n in os.listdir(path))


def encoders_of(binary):
    return subprocess.run(
        [binary, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.decode()


def model_files():
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        raise RuntimeError(f"missing config.json in {MODEL_DIR}")
    if not any(n.endswith(".safetensors") for n in os.listdir(MODEL_DIR)):
        raise RuntimeError("no .safetensors weights in model dir")
    if not (
        os.path.isfile(os.path.join(MODEL_DIR, "tokenizer.json"))
        or os.path.isfile(os.path.join(MODEL_DIR, "tokenizer.model"))
    ):
        raise RuntimeError("no tokenizer files in model dir")
    with open(os.path.join(MODEL_DIR, "config.json")) as handle:
        config = json.load(handle)
    quant = config.get("quantization_config")
    if quant:
        return f"checkpoint declares quantization {quant.get('quant_method')}"
    return "no quantization_config: the engine will apply fp8 itself at load time"


def snac_files():
    if not os.path.isfile(os.path.join(SNAC_DIR, "config.json")):
        raise RuntimeError(f"missing config.json in {SNAC_DIR}")
    if not has_weights(SNAC_DIR):
        raise RuntimeError("no weight file in SNAC dir")


def tokenizer_ok():
    from transformers import AutoTokenizer

    from app.prompt import AUDIO_END

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if len(tok) < AUDIO_END:
        raise RuntimeError(f"tokenizer has {len(tok)} tokens, audio range needs {AUDIO_END}")
    ids = tok("Hindi (Female): नमस्ते", add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError("tokenizer returned no ids")
    return f"{len(tok)} tokens"


def imports_ok():
    for name in ("transformers", "snac", "soxr", "soundfile", "requests", "google.cloud.storage", "numpy"):
        importlib.import_module(name)
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("vllm is not installed")
    return "vllm " + importlib.metadata.version("vllm")


def app_modules():
    for name in ("app.audio", "app.codec", "app.config", "app.engine", "app.pipeline", "app.prompt"):
        importlib.import_module(name)


def gcc_present():
    path = shutil.which("gcc")
    if not path:
        raise RuntimeError("gcc not found (Triton needs it)")
    return path


def ffmpeg_path():
    resolved = shutil.which(FFMPEG)
    if not resolved:
        raise RuntimeError(f"{FFMPEG} not found")
    if "libopus" not in encoders_of(resolved):
        raise RuntimeError(f"{resolved} has no libopus encoder")
    return resolved


def bare_ffmpeg():
    resolved = shutil.which("ffmpeg")
    if not resolved:
        raise RuntimeError("ffmpeg not on PATH")
    if "libopus" not in encoders_of(resolved):
        raise RuntimeError(f"ffmpeg on PATH ({resolved}) has no libopus encoder")
    return resolved


def opus_ok():
    import numpy as np

    from app.audio import OpusEncoder

    t = np.arange(16000) / 16000.0
    pcm = (np.sin(2 * np.pi * 300 * t) * 12000).astype(np.int16)
    data = OpusEncoder(16000, "24k", FFMPEG).encode(pcm)
    if len(data) < 500:
        raise RuntimeError("opus output is suspiciously small")
    return f"{len(data)} bytes for 1 s"


def runs_as_non_root():
    if os.getuid() == 0:
        raise RuntimeError("running as root")


def entrypoint_present():
    path = os.path.join(ROOT, "entrypoint.sh")
    if not os.access(path, os.X_OK):
        raise RuntimeError("entrypoint.sh is missing or not executable")
    with open(path, "rb") as handle:
        if b"\r\n" in handle.read():
            raise RuntimeError("entrypoint.sh has Windows line endings")


for label, fn in (
    ("model files", model_files),
    ("snac files", snac_files),
    ("tokenizer", tokenizer_ok),
    ("imports", imports_ok),
    ("app modules", app_modules),
    ("gcc", gcc_present),
    ("ffmpeg binary", ffmpeg_path),
    ("ffmpeg on PATH", bare_ffmpeg),
    ("opus encode", opus_ok),
    ("entrypoint", entrypoint_present),
    ("non-root user", runs_as_non_root),
):
    check(label, fn)

for line in notes:
    print(line)
if problems:
    for line in problems:
        print("FAIL  " + line, file=sys.stderr)
    sys.exit(1)
print("verify_image_ok")