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
    # --- engine ---
    max_num_seqs: int
    gpu_memory_utilization: float
    gpu_total_mb: int
    max_model_len: int
    kv_cache_dtype: str
    quantization: str
    model_dir: str
    snac_dir: str
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    num_scheduler_steps: int
    max_num_batched_tokens: int
    final_only_outputs: bool
    skip_detokenize: bool
    warmup_voices: int
    # --- sampling ---
    logits_mask: str
    allow_header_tokens: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    use_seeds: bool
    base_seed: int
    retry_growth: float
    # --- codec / audio ---
    snac_half: bool
    bad_code_tolerance: float
    opus_bitrate: str
    opus_sample_rate: int
    source_sample_rate: int
    trim_threshold: float
    trim_pad_seconds: float
    fade_seconds: float
    peak_target: float
    rms_gate: float
    min_duration: float
    max_duration: float
    loudness_normalize: bool
    loudness_target_db: float
    quality_pace_gate: bool
    min_sec_per_char: float
    max_sec_per_char: float
    pace_slack_seconds: float
    # --- storage ---
    gcs_bucket: str
    gcs_prefix: str
    gcs_credentials_b64: str
    cdn_base_url: str
    model_revision: str
    # --- limits / cost guards ---
    pod_limit_seconds: int
    seconds_per_item_cap: float
    startup_timeout_seconds: int
    startup_alarm_margin_seconds: int
    idle_timeout_seconds: int
    max_consecutive_failures: int
    breaker_window: int
    breaker_fail_ratio: float
    max_spend_usd: float
    gpu_hourly_rate: float
    # --- pipeline ---
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
        # "auto" = model dtype. FP8 KV needs CUDA arch >= 8.9 and is guarded in engine.py.
        kv_cache_dtype=_str("KV_CACHE_DTYPE", "auto"),
        quantization=_str("QUANTIZATION", ""),
        model_dir=_str("MODEL_DIR", "/opt/models/svara-fp8"),
        snac_dir=_str("SNAC_DIR", "/opt/models/snac_24khz"),
        # Prompts are unique, so prefix caching only adds a kernel path (the one that crashed on Ampere).
        enable_prefix_caching=_bool("ENABLE_PREFIX_CACHING", False),
        enable_chunked_prefill=_bool("ENABLE_CHUNKED_PREFILL", True),
        num_scheduler_steps=_int("NUM_SCHEDULER_STEPS", 1),
        max_num_batched_tokens=_int("MAX_NUM_BATCHED_TOKENS", 0),
        final_only_outputs=_bool("FINAL_ONLY_OUTPUTS", True),
        skip_detokenize=_bool("SKIP_DETOKENIZE", True),
        warmup_voices=_int("WARMUP_VOICES", 1),
        logits_mask=_str("LOGITS_MASK", "range"),
        allow_header_tokens=_bool("ALLOW_HEADER_TOKENS", True),
        temperature=_float("TEMPERATURE", 0.72),
        top_p=_float("TOP_P", 0.92),
        top_k=_int("TOP_K", 50),
        repetition_penalty=_float("REPETITION_PENALTY", 1.0),
        use_seeds=_bool("USE_SEEDS", False),
        base_seed=_int("BASE_SEED", 1234),
        retry_growth=_float("RETRY_GROWTH", 1.5),
        snac_half=_bool("SNAC_HALF", True),
        bad_code_tolerance=_float("BAD_CODE_TOLERANCE", 0.01),
        opus_bitrate=_str("OPUS_BITRATE", "24k"),
        opus_sample_rate=_int("OPUS_SAMPLE_RATE", 16000),
        source_sample_rate=24000,
        trim_threshold=_float("TRIM_THRESHOLD", 0.006),
        trim_pad_seconds=_float("TRIM_PAD_SECONDS", 0.05),
        fade_seconds=_float("FADE_SECONDS", 0.008),
        peak_target=_float("PEAK_TARGET", 0.92),
        rms_gate=_float("RMS_GATE", 1e-3),
        min_duration=_float("MIN_DURATION", 0.4),
        max_duration=_float("MAX_DURATION", 45.0),
        loudness_normalize=_bool("LOUDNESS_NORMALIZE", True),
        loudness_target_db=_float("LOUDNESS_TARGET_DB", -20.0),
        quality_pace_gate=_bool("QUALITY_PACE_GATE", True),
        min_sec_per_char=_float("MIN_SEC_PER_CHAR", 0.03),
        max_sec_per_char=_float("MAX_SEC_PER_CHAR", 0.35),
        pace_slack_seconds=_float("PACE_SLACK_SECONDS", 2.0),
        gcs_bucket=_need("GCS_BUCKET", local, "local"),
        gcs_prefix=_str("GCS_PREFIX", "tts"),
        gcs_credentials_b64=_need("GCS_SERVICE_ACCOUNT_JSON_B64", local, ""),
        cdn_base_url=_need("CDN_BASE_URL", local, "").rstrip("/"),
        model_revision=_need("MODEL_REVISION", local, "local"),
        pod_limit_seconds=_int("POD_LIMIT_SECONDS", 7200),
        seconds_per_item_cap=_float("SECONDS_PER_ITEM_CAP", 3.0),
        startup_timeout_seconds=_int("STARTUP_TIMEOUT_SECONDS", 420),
        startup_alarm_margin_seconds=_int("STARTUP_ALARM_MARGIN_SECONDS", 90),
        idle_timeout_seconds=_int("IDLE_TIMEOUT_SECONDS", 300),
        max_consecutive_failures=_int("MAX_CONSECUTIVE_FAILURES", 25),
        breaker_window=_int("BREAKER_WINDOW", 100),
        breaker_fail_ratio=_float("BREAKER_FAIL_RATIO", 0.6),
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
        # Margin added on top of items * SECONDS_PER_ITEM_CAP for the RUN alarm (startup is separate).
        startup_margin_seconds=_int("STARTUP_MARGIN_SECONDS", 120),
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


def run_limit_seconds(settings, pending):
    """Alarm for the generation phase only. Model loading has its own, tighter alarm."""
    dynamic = int(pending * settings.seconds_per_item_cap) + settings.startup_margin_seconds
    return max(180, min(settings.pod_limit_seconds, dynamic))


hard_limit_seconds = run_limit_seconds