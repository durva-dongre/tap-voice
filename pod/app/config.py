import os
from dataclasses import dataclass


def _raw(name):
    return os.environ.get(name, "").strip()


def _req(name):
    value = _raw(name)
    if not value:
        raise RuntimeError(f"missing required env {name}")
    return value


def _str(name, default):
    return _raw(name) or default


def _int(name, default):
    value = _raw(name)
    return int(value) if value else default


def _float(name, default):
    value = _raw(name)
    return float(value) if value else default


def _bool(name, default):
    value = _raw(name)
    if not value:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _need(name, local, default):
    return _str(name, default) if local else _req(name)


@dataclass(frozen=True)
class Settings:
    selftest: bool
    run_id: str
    run_token: str
    server_url: str
    max_num_seqs: int
    gpu_memory_utilization: float
    gpu_total_mb: int
    max_model_len: int
    kv_cache_dtype: str
    quantization: str
    model_dir: str
    snac_dir: str
    logits_mask: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    use_seeds: bool
    base_seed: int
    opus_bitrate: str
    opus_sample_rate: int
    source_sample_rate: int
    trim_threshold: float
    trim_pad_seconds: float
    peak_target: float
    rms_gate: float
    min_duration: float
    max_duration: float
    loudness_normalize: bool
    loudness_target_db: float
    gcs_bucket: str
    gcs_prefix: str
    gcs_credentials_b64: str
    cdn_base_url: str
    model_revision: str
    pod_limit_seconds: int
    seconds_per_item_cap: float
    startup_timeout_seconds: int
    idle_timeout_seconds: int
    max_consecutive_failures: int
    max_spend_usd: float
    gpu_hourly_rate: float
    upload_threads: int
    decode_microbatch: int
    decode_flush_seconds: float
    beat_seconds: int
    progress_every_items: int
    progress_every_seconds: float
    stop_grace_seconds: float
    max_items: int
    max_text_chars: int
    max_retries: int
    queue_depth: int
    slot_timeout_seconds: float
    codec_vram_reserve_mb: int
    min_max_tokens: int
    max_max_tokens: int
    tokens_per_char: float
    min_tokens_per_char: float
    startup_margin_seconds: int
    opus_size_low: float
    opus_size_high: float
    storage_backend: str
    local_dir: str
    local_port: int


def load():
    selftest = _bool("SELFTEST", False)
    if selftest:
        run_id = _str("RUN_ID", "selftest")
        run_token = _str("RUN_TOKEN", "selftest")
        server_url = _str("SERVER_URL", "http://localhost").rstrip("/")
    else:
        run_id = _req("RUN_ID")
        run_token = _req("RUN_TOKEN")
        server_url = _req("SERVER_URL").rstrip("/")
    backend = _str("STORAGE_BACKEND", "gcs").lower()
    if backend not in ("gcs", "local"):
        raise RuntimeError(f"unsupported STORAGE_BACKEND {backend}")
    local = backend == "local"
    return Settings(
        selftest=selftest,
        run_id=run_id,
        run_token=run_token,
        server_url=server_url,
        max_num_seqs=_int("MAX_NUM_SEQS", 48),
        gpu_memory_utilization=_float("GPU_MEMORY_UTILIZATION", 0.90),
        gpu_total_mb=_int("GPU_TOTAL_MB", 24564),
        max_model_len=_int("MAX_MODEL_LEN", 2700),
        kv_cache_dtype=_str("KV_CACHE_DTYPE", "fp8"),
        quantization=_str("QUANTIZATION", ""),
        model_dir=_str("MODEL_DIR", "/opt/models/svara-fp8"),
        snac_dir=_str("SNAC_DIR", "/opt/models/snac_24khz"),
        logits_mask=_str("LOGITS_MASK", "range"),
        temperature=_float("TEMPERATURE", 0.72),
        top_p=_float("TOP_P", 0.92),
        top_k=_int("TOP_K", 50),
        repetition_penalty=_float("REPETITION_PENALTY", 1.0),
        use_seeds=_bool("USE_SEEDS", False),
        base_seed=_int("BASE_SEED", 1234),
        opus_bitrate=_str("OPUS_BITRATE", "24k"),
        opus_sample_rate=_int("OPUS_SAMPLE_RATE", 16000),
        source_sample_rate=24000,
        trim_threshold=_float("TRIM_THRESHOLD", 0.006),
        trim_pad_seconds=_float("TRIM_PAD_SECONDS", 0.05),
        peak_target=_float("PEAK_TARGET", 0.92),
        rms_gate=_float("RMS_GATE", 1e-3),
        min_duration=_float("MIN_DURATION", 0.4),
        max_duration=_float("MAX_DURATION", 45.0),
        loudness_normalize=_bool("LOUDNESS_NORMALIZE", True),
        loudness_target_db=_float("LOUDNESS_TARGET_DB", -20.0),
        gcs_bucket=_need("GCS_BUCKET", local, "local"),
        gcs_prefix=_str("GCS_PREFIX", "tts"),
        gcs_credentials_b64=_need("GCS_SERVICE_ACCOUNT_JSON_B64", local, ""),
        cdn_base_url=_need("CDN_BASE_URL", local, "").rstrip("/"),
        model_revision=_need("MODEL_REVISION", local, "local"),
        pod_limit_seconds=_int("POD_LIMIT_SECONDS", 7200),
        seconds_per_item_cap=_float("SECONDS_PER_ITEM_CAP", 3.0),
        startup_timeout_seconds=_int("STARTUP_TIMEOUT_SECONDS", 600),
        idle_timeout_seconds=_int("IDLE_TIMEOUT_SECONDS", 300),
        max_consecutive_failures=_int("MAX_CONSECUTIVE_FAILURES", 25),
        max_spend_usd=_float("MAX_SPEND_USD", 1.0),
        gpu_hourly_rate=_float("GPU_HOURLY_RATE", 0.34),
        upload_threads=_int("UPLOAD_THREADS", 8),
        decode_microbatch=_int("DECODE_MICROBATCH", 8),
        decode_flush_seconds=_float("DECODE_FLUSH_SECONDS", 0.25),
        beat_seconds=_int("BEAT_SECONDS", 30),
        progress_every_items=_int("PROGRESS_EVERY_ITEMS", 5),
        progress_every_seconds=_float("PROGRESS_EVERY_SECONDS", 5.0),
        stop_grace_seconds=_float("STOP_GRACE_SECONDS", 20.0),
        max_items=_int("MAX_ITEMS", 5000),
        max_text_chars=_int("MAX_TEXT_CHARS", 300),
        max_retries=_int("MAX_RETRIES", 2),
        queue_depth=_int("QUEUE_DEPTH", 64),
        slot_timeout_seconds=_float("SLOT_TIMEOUT_SECONDS", 120.0),
        codec_vram_reserve_mb=_int("CODEC_VRAM_RESERVE_MB", 500),
        min_max_tokens=_int("MIN_MAX_TOKENS", 400),
        max_max_tokens=_int("MAX_MAX_TOKENS", 2400),
        tokens_per_char=_float("TOKENS_PER_CHAR", 9.0),
        min_tokens_per_char=_float("MIN_TOKENS_PER_CHAR", 7.0),
        startup_margin_seconds=_int("STARTUP_MARGIN_SECONDS", 600),
        opus_size_low=_float("OPUS_SIZE_LOW", 0.6),
        opus_size_high=_float("OPUS_SIZE_HIGH", 1.6),
        storage_backend=backend,
        local_dir=_str("LOCAL_STORAGE_DIR", "/tmp/clips"),
        local_port=_int("LOCAL_STORAGE_PORT", 8081),
    )


_SETTINGS = None


def get():
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load()
    return _SETTINGS


def reset():
    global _SETTINGS
    _SETTINGS = None


def hard_limit_seconds(settings, item_count):
    dynamic = int(item_count * settings.seconds_per_item_cap) + settings.startup_margin_seconds
    return max(300, min(settings.pod_limit_seconds, dynamic))