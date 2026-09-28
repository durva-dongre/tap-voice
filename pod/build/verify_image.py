import importlib
import importlib.util
import os
import sys

for name in ("GCS_BUCKET", "GCS_SERVICE_ACCOUNT_JSON_B64", "CDN_BASE_URL", "MODEL_REVISION"):
    if not os.environ.get(name, "").strip():
        os.environ[name] = "verify"
os.environ.setdefault("SELFTEST", "1")

REQUIRED_MODULES = [
    "torch",
    "transformers",
    "snac",
    "numpy",
    "soundfile",
    "soxr",
    "requests",
    "google.cloud.storage",
]

APP_MODULES = [
    "app.audio",
    "app.codec",
    "app.config",
    "app.control",
    "app.engine",
    "app.logits",
    "app.manifest",
    "app.pipeline",
    "app.prompt",
    "app.storage",
    "app.telemetry",
    "app.voices",
    "app.main",
]


def fail(message):
    sys.stderr.write("verify_failed " + message + "\n")
    sys.exit(1)


def main():
    if importlib.util.find_spec("vllm") is None:
        fail("vllm_missing")
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            fail(f"import_{name}:{type(exc).__name__}:{exc}")
    for name in APP_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            fail(f"import_{name}:{type(exc).__name__}:{exc}")
    model_dir = os.environ.get("MODEL_DIR", "/opt/models/svara-fp8")
    snac_dir = os.environ.get("SNAC_DIR", "/opt/models/snac_24khz")
    for path in (model_dir, snac_dir):
        if not os.path.isdir(path):
            fail(f"missing_dir:{path}")
        if not os.path.isfile(os.path.join(path, "config.json")):
            fail(f"missing_config:{path}")
    if not any(name.endswith(".safetensors") for name in os.listdir(model_dir)):
        fail("missing_weights")
    sys.stdout.write("verify_ok\n")


main()