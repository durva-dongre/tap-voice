import os

E = os.environ
RUN_ID = E.get("RUN_ID", "")
RUN_TOKEN = E.get("RUN_TOKEN", "")
WORKER_URL = E.get("WORKER_URL", "").rstrip("/")
MAX_JOBS = int(E.get("MAX_JOBS", "400"))
LIMIT_S = int(E.get("POD_LIMIT_SECONDS", "2400"))
BEAT_S = int(E.get("BEAT_SECONDS", "60"))
DELIVER_WORKERS = int(E.get("DELIVER_WORKERS", "4"))
MOCK = E.get("MOCK_ENGINES") == "1"
KOKORO_VOICE = E.get("KOKORO_VOICE", "af_heart")
INDIC_REPO = E.get("INDIC_REPO", "ai4bharat/indic-parler-tts")
INDIC_SPEAKER = E.get(
    "INDIC_SPEAKER",
    "A female speaker delivers clear, natural speech with a moderate pace and very clear audio, with no background noise.",
)
INDIC_DTYPE = E.get("INDIC_DTYPE", "float16")