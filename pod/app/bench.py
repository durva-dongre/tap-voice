import argparse
import asyncio
import dataclasses
import itertools
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PREFIX = "BENCH_RESULT "
LENGTHS = ("short", "medium", "long")
TRUE_VALUES = ("1", "true", "yes", "on")

SENTENCES = {
    "english": (
        "Hello, your appointment is confirmed for tomorrow morning.",
        "Thank you for calling. Please listen carefully, because our menu options have recently changed, and we want to make sure that you reach the right department quickly.",
    ),
    "hindi": (
        "नमस्ते, आपकी अपॉइंटमेंट कल सुबह के लिए पक्की हो गई है।",
        "कॉल करने के लिए धन्यवाद। कृपया ध्यान से सुनें, क्योंकि हमारे मेनू के विकल्प हाल ही में बदल गए हैं, और हम चाहते हैं कि आप सही विभाग तक जल्दी पहुँचें।",
    ),
    "marathi": (
        "नमस्कार, तुमची भेट उद्या सकाळसाठी निश्चित झाली आहे.",
        "फोन केल्याबद्दल धन्यवाद. कृपया काळजीपूर्वक ऐका, कारण आमचे मेनू पर्याय नुकतेच बदलले आहेत, आणि तुम्ही योग्य विभागाशी लवकर संपर्क साधावा अशी आमची इच्छा आहे.",
    ),
    "kannada": (
        "ನಮಸ್ಕಾರ, ನಿಮ್ಮ ಭೇಟಿ ನಾಳೆ ಬೆಳಿಗ್ಗೆಗೆ ಖಚಿತವಾಗಿದೆ.",
        "ಕರೆ ಮಾಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ದಯವಿಟ್ಟು ಗಮನವಿಟ್ಟು ಕೇಳಿ, ಏಕೆಂದರೆ ನಮ್ಮ ಮೆನು ಆಯ್ಕೆಗಳು ಇತ್ತೀಚೆಗೆ ಬದಲಾಗಿವೆ, ಮತ್ತು ನೀವು ಸರಿಯಾದ ವಿಭಾಗವನ್ನು ಬೇಗ ತಲುಪಬೇಕೆಂದು ನಾವು ಬಯಸುತ್ತೇವೆ.",
    ),
    "punjabi": (
        "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਹਾਡੀ ਮੁਲਾਕਾਤ ਕੱਲ੍ਹ ਸਵੇਰ ਲਈ ਪੱਕੀ ਹੋ ਗਈ ਹੈ।",
        "ਕਾਲ ਕਰਨ ਲਈ ਧੰਨਵਾਦ। ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਨਾਲ ਸੁਣੋ, ਕਿਉਂਕਿ ਸਾਡੇ ਮੈਨੂ ਦੇ ਵਿਕਲਪ ਹਾਲ ਹੀ ਵਿੱਚ ਬਦਲ ਗਏ ਹਨ, ਅਤੇ ਅਸੀਂ ਚਾਹੁੰਦੇ ਹਾਂ ਕਿ ਤੁਸੀਂ ਸਹੀ ਵਿਭਾਗ ਤੱਕ ਜਲਦੀ ਪਹੁੰਚੋ।",
    ),
}


def texts_for(language):
    short, medium = SENTENCES[language]
    return {"short": short, "medium": medium, "long": f"{medium} {short}"}


def load_pool(path):
    from app.voices import LANGUAGE_VOICES

    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)
    pool = []
    for entry in entries:
        language = entry["language"]
        text = entry["text"]
        if language not in LANGUAGE_VOICES:
            raise ValueError(f"unknown language {language}")
        if not text.strip():
            raise ValueError("empty text in texts file")
        pool.append((language, text))
    if not pool:
        raise ValueError("texts file is empty")
    return pool


def apply_env():
    os.environ["SELFTEST"] = "1"
    for name in ("GCS_BUCKET", "GCS_SERVICE_ACCOUNT_JSON_B64", "CDN_BASE_URL", "MODEL_REVISION"):
        if not os.environ.get(name, "").strip():
            os.environ[name] = "bench"


def read_gpu():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        ).stdout.decode().strip().splitlines()[0]
        name, used, total, util = [part.strip() for part in out.split(",")]
        return name, float(used), float(total), float(util)
    except Exception:
        return None


class Sampler(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.halt = threading.Event()
        self.peak_mb = 0.0
        self.utils = []

    def run(self):
        while not self.halt.wait(self.interval):
            reading = read_gpu()
            if reading is None:
                continue
            self.peak_mb = max(self.peak_mb, reading[1])
            self.utils.append(reading[3])

    def average_util(self):
        return round(sum(self.utils) / len(self.utils), 1) if self.utils else None


def build_items(settings, count, pool=None):
    from app.manifest import Item, content_hash
    from app.voices import LANGUAGE_VOICES

    languages = list(SENTENCES)
    table = {language: texts_for(language) for language in languages}
    items = []
    for index in range(count):
        if pool:
            language, base = pool[index % len(pool)]
        else:
            language = languages[index % len(languages)]
            length = LENGTHS[(index // len(languages)) % len(LENGTHS)]
            base = table[language][length]
        text = f"{base} {index}"
        voice = LANGUAGE_VOICES[language]
        items.append(
            Item(
                id=f"bench-{index}",
                text=text,
                language=language,
                voice=voice,
                emotion=None,
                fmt="ogg",
                content_hash=content_hash(text, voice, None, "ogg", settings.model_revision),
            )
        )
    return items


async def run_single(settings, count, pool=None):
    from app import audio
    from app.codec import Codec
    from app.engine import Engine
    from app.prompt import prepare, sort_by_length
    from app.voices import ALLOWED_VOICES

    sampler = Sampler()
    sampler.start()
    began = time.time()
    engine = Engine(settings)
    await engine.start()
    codec = Codec(
        settings.snac_dir, half=settings.snac_half, tolerance=settings.bad_code_tolerance
    )
    codec.warmup()
    await engine.warmup(sorted(ALLOWED_VOICES))
    startup = time.time() - began

    items = build_items(settings, count, pool)
    lock = threading.Lock()
    stage = {"decode": 0.0, "encode": 0.0}
    failures = {}
    totals = {"tokens": 0, "truncated_first_pass": 0, "retried": 0}
    truncated_by_language = {}
    lang_tokens = {}
    lang_chars = {}
    pool_exec = ThreadPoolExecutor(max_workers=settings.upload_threads)
    futures = []
    loop = asyncio.get_running_loop()

    def note(reason):
        with lock:
            failures[reason] = failures.get(reason, 0) + 1

    def timed_decode(entries):
        start = time.time()
        result = codec.decode_batch(entries)
        with lock:
            stage["decode"] += time.time() - start
        return result

    def encode_job(item, wave):
        start = time.time()
        try:
            processed, reason = audio.process(wave, settings, len(item.text))
            if processed is None:
                outcome = ("fail", reason, 0.0)
            else:
                data, _, _ = audio.encode(processed, item.fmt, settings)
                if data:
                    outcome = ("ok", "", processed.size / float(settings.source_sample_rate))
                else:
                    outcome = ("fail", "empty_encode", 0.0)
        except Exception as exc:
            outcome = ("fail", type(exc).__name__, 0.0)
        with lock:
            stage["encode"] += time.time() - start
        return outcome

    async def drain(buffer, by_id):
        entries = [(f.key, f.tokens) for f in buffer]
        decoded, bad = await loop.run_in_executor(None, timed_decode, entries)
        for _, reason in bad:
            note(reason)
        for key, wave in decoded:
            futures.append(pool_exec.submit(encode_job, by_id[key], wave))

    async def one_pass(batch):
        prepared = sort_by_length(prepare(batch, engine.tokenizer, settings, attempt=0))
        by_id = {p.item.id: p.item for p in prepared}
        buffer = []
        last = time.time()
        async for finished in engine.generate(prepared, lambda: False):
            totals["tokens"] += finished.spent_tokens
            totals["truncated_first_pass"] += finished.truncations
            totals["retried"] += finished.generations - 1
            if finished.error:
                note(finished.error)
                continue
            source = by_id[finished.key]
            if finished.finish_reason == "length":
                note("truncated")
                truncated_by_language[source.language] = truncated_by_language.get(source.language, 0) + 1
                continue
            lang_tokens[source.language] = lang_tokens.get(source.language, 0) + len(finished.tokens)
            lang_chars[source.language] = lang_chars.get(source.language, 0) + len(source.text)
            buffer.append(finished)
            if (
                len(buffer) >= settings.decode_microbatch
                or time.time() - last >= settings.decode_flush_seconds
            ):
                ready, buffer = buffer, []
                last = time.time()
                await drain(ready, by_id)
        if buffer:
            await drain(buffer, by_id)

    run_start = time.time()
    await one_pass(items)
    generate_wall = time.time() - run_start

    ok = 0
    audio_seconds = 0.0
    for future in futures:
        status, reason, seconds = future.result()
        if status == "ok":
            ok += 1
            audio_seconds += seconds
        else:
            note(reason)
    wall = time.time() - run_start
    pool_exec.shutdown(wait=True)
    sampler.halt.set()
    await engine.shutdown()

    hours = wall / 3600.0
    gpu = read_gpu()
    tokens_per_char = {
        language: round(lang_tokens[language] / float(lang_chars[language]), 2)
        for language in lang_tokens
        if lang_chars[language]
    }
    return {
        "gpu_name": gpu[0] if gpu else "unknown",
        "model_dir": settings.model_dir,
        "max_num_seqs": settings.max_num_seqs,
        "decode_microbatch": settings.decode_microbatch,
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "logits_mask": settings.logits_mask,
        "num_scheduler_steps": settings.num_scheduler_steps,
        "max_num_batched_tokens": settings.max_num_batched_tokens,
        "enable_chunked_prefill": settings.enable_chunked_prefill,
        "upload_threads": settings.upload_threads,
        "kv_cache_dtype": engine.kv_dtype,
        "gpu_hourly_rate": settings.gpu_hourly_rate,
        "clips_total": count,
        "clips_ok": ok,
        "fail_rate": round((count - ok) / float(count), 4),
        "failure_reasons": failures,
        "truncated_clips": totals["truncated_first_pass"],
        "truncated_by_language": truncated_by_language,
        "tokens_per_char_by_language": tokens_per_char,
        "retried_clips": totals["retried"],
        "startup_seconds": round(startup, 1),
        "wall_seconds": round(wall, 2),
        "generate_wall_seconds": round(generate_wall, 2),
        "decode_seconds": round(stage["decode"], 2),
        "encode_seconds": round(stage["encode"], 2),
        "clips_per_minute": round(ok / (wall / 60.0), 1) if wall > 0 else 0.0,
        "tokens_per_second": round(totals["tokens"] / wall, 1) if wall > 0 else 0.0,
        "audio_seconds": round(audio_seconds, 1),
        "avg_clip_seconds": round(audio_seconds / ok, 2) if ok else None,
        "gpu_seconds_per_audio_minute": round(wall / (audio_seconds / 60.0), 2) if audio_seconds else None,
        "peak_vram_mb": round(sampler.peak_mb),
        "avg_gpu_util_pct": sampler.average_util(),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "cost_per_1000_usd": round(hours / ok * 1000.0 * settings.gpu_hourly_rate, 4) if ok else None,
        "cost_per_1000_audio_minutes_usd": round(
            hours / (audio_seconds / 60.0) * 1000.0 * settings.gpu_hourly_rate, 4
        )
        if audio_seconds
        else None,
    }


def single(args):
    apply_env()
    from app import config

    settings = config.load()
    gpu = read_gpu()
    overrides = {
        "max_num_seqs": args.max_num_seqs,
        "decode_microbatch": args.decode_microbatch,
        "gpu_memory_utilization": args.gpu_util,
        "gpu_hourly_rate": args.rate,
    }
    if args.logits_mask:
        overrides["logits_mask"] = args.logits_mask
    if args.scheduler_steps > 0:
        overrides["num_scheduler_steps"] = args.scheduler_steps
    if args.kv_dtype:
        overrides["kv_cache_dtype"] = args.kv_dtype
    if args.batched_tokens >= 0:
        overrides["max_num_batched_tokens"] = args.batched_tokens
    if args.chunked:
        overrides["enable_chunked_prefill"] = args.chunked.lower() in TRUE_VALUES
    if args.upload_threads > 0:
        overrides["upload_threads"] = args.upload_threads
    if gpu:
        overrides["gpu_total_mb"] = int(gpu[2])
    if args.model_dir:
        overrides["model_dir"] = args.model_dir
    settings = dataclasses.replace(settings, **overrides)
    pool = load_pool(args.texts) if args.texts else None
    result = asyncio.run(run_single(settings, args.count, pool))
    sys.stdout.write(PREFIX + json.dumps(result) + "\n")
    sys.stdout.flush()
    os._exit(0)


def count_preemptions(log_path):
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    cumulative = [int(v) for v in re.findall(r"total_num_cumulative_preemption=(\d+)", text)]
    if cumulative:
        return max(cumulative)
    return len(re.findall(r"preempted", text))


def run_child(args, cfg):
    command = [
        sys.executable,
        "-m",
        "app.bench",
        "--single",
        "--count",
        str(args.count),
        "--max-num-seqs",
        str(cfg["seqs"]),
        "--decode-microbatch",
        str(cfg["micro"]),
        "--gpu-util",
        str(cfg["util"]),
        "--rate",
        str(args.rate),
        "--logits-mask",
        cfg["mask"],
        "--scheduler-steps",
        str(cfg["steps"]),
        "--kv-dtype",
        cfg["kv"],
        "--batched-tokens",
        str(cfg["tokens"]),
        "--chunked",
        cfg["chunked"],
        "--upload-threads",
        str(cfg["threads"]),
    ]
    if args.model_dir:
        command += ["--model-dir", args.model_dir]
    if args.texts:
        command += ["--texts", args.texts]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = (
        f"s{cfg['seqs']}_m{cfg['micro']}_u{cfg['util']}_{cfg['mask'] or 'default'}"
        f"_k{cfg['steps']}_kv{cfg['kv'] or 'default'}_t{cfg['tokens']}"
        f"_c{cfg['chunked'] or 'default'}_w{cfg['threads']}"
    )
    log_path = os.path.join(args.out_dir, f"run_{tag}.log")
    base = {
        "max_num_seqs": cfg["seqs"],
        "decode_microbatch": cfg["micro"],
        "gpu_memory_utilization": cfg["util"],
        "logits_mask": cfg["mask"] or "default",
        "num_scheduler_steps": cfg["steps"] or 1,
        "kv_cache_dtype": cfg["kv"] or "default",
        "max_num_batched_tokens": cfg["tokens"] if cfg["tokens"] >= 0 else "default",
        "enable_chunked_prefill": cfg["chunked"] or "default",
        "upload_threads": cfg["threads"] or "default",
    }
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=log,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return dict(base, error="timeout", log=log_path)
    for line in reversed(proc.stdout.decode("utf-8", "replace").splitlines()):
        if line.startswith(PREFIX):
            row = json.loads(line[len(PREFIX):])
            row["preempted"] = count_preemptions(log_path)
            row["log"] = log_path
            return row
    return dict(base, error=f"exit_{proc.returncode}", log=log_path)


def print_rows(rows):
    header = (
        f"{'gpu':<20}{'seqs':>5}{'mb':>4}{'util':>6}{'mask':>7}{'k':>3}{'kv':>6}{'bt':>6}{'thr':>4}"
        f"{'ok':>11}{'clips/min':>10}{'tok/s':>8}{'gpu%':>6}{'vramMB':>8}{'pre':>5}{'$/1000':>9}\n"
    )
    sys.stdout.write(header)
    for row in rows:
        name = str(row.get("gpu_name", "-"))[:19]
        lead = (
            f"{name:<20}{row['max_num_seqs']:>5}{row['decode_microbatch']:>4}"
            f"{row['gpu_memory_utilization']:>6}{str(row.get('logits_mask', '-')):>7}"
            f"{row.get('num_scheduler_steps', 1):>3}{str(row.get('kv_cache_dtype', '-')):>6}"
            f"{str(row.get('max_num_batched_tokens', '-')):>6}{str(row.get('upload_threads', '-')):>4}"
        )
        if row.get("error"):
            sys.stdout.write(lead + f"  FAILED: {row['error']}\n")
            continue
        ok = f"{row['clips_ok']}/{row['clips_total']}"
        cost = row["cost_per_1000_usd"]
        cost_text = f"{cost:.4f}" if cost is not None else "-"
        util = row.get("avg_gpu_util_pct")
        preempted = row.get("preempted")
        sys.stdout.write(
            lead
            + f"{ok:>11}{row['clips_per_minute']:>10}{row['tokens_per_second']:>8}"
            + f"{(util if util is not None else '-'):>6}{row['peak_vram_mb']:>8}"
            + f"{(preempted if preempted is not None else '-'):>5}{cost_text:>9}\n"
        )


def best_of(rows, max_fail):
    valid = [
        r
        for r in rows
        if not r.get("error") and r["cost_per_1000_usd"] is not None and r["fail_rate"] <= max_fail
    ]
    return min(valid, key=lambda r: r["cost_per_1000_usd"]) if valid else None


def report_best(rows, max_fail):
    best = best_of(rows, max_fail)
    if best is None:
        sys.stdout.write("no configuration met the failure-rate limit\n")
        return 1
    sys.stdout.write(
        f"\nbest: {best['gpu_name']} at ${best['cost_per_1000_usd']:.4f} per 1000 clips, "
        f"{best['clips_per_minute']} clips/min, peak VRAM {best['peak_vram_mb']} MB, "
        f"avg clip {best.get('avg_clip_seconds')} s, first-pass truncated {best.get('truncated_clips', 0)}\n"
        f"MAX_NUM_SEQS={best['max_num_seqs']} DECODE_MICROBATCH={best['decode_microbatch']} "
        f"GPU_MEMORY_UTILIZATION={best['gpu_memory_utilization']} "
        f"LOGITS_MASK={best.get('logits_mask', 'range')} "
        f"NUM_SCHEDULER_STEPS={best.get('num_scheduler_steps', 1)} "
        f"KV_CACHE_DTYPE={best.get('kv_cache_dtype', 'auto')} "
        f"MAX_NUM_BATCHED_TOKENS={best.get('max_num_batched_tokens', 0)} "
        f"ENABLE_CHUNKED_PREFILL={int(bool(best.get('enable_chunked_prefill', True)))} "
        f"UPLOAD_THREADS={best.get('upload_threads', 8)}\n"
    )
    per_clip = best["wall_seconds"] / max(best["clips_ok"], 1)
    sys.stdout.write(f"suggested SECONDS_PER_ITEM_CAP={round(per_clip * 2, 2)} (2x measured {per_clip:.2f}s/clip)\n")
    if best.get("truncated_by_language"):
        sys.stdout.write(f"first-pass truncation by language: {best['truncated_by_language']}\n")
    if best.get("tokens_per_char_by_language"):
        sys.stdout.write(f"measured tokens per char by language: {best['tokens_per_char_by_language']}\n")
    if best.get("preempted"):
        sys.stdout.write("note: the best run had KV preemptions, lower MAX_NUM_SEQS or free KV memory before trusting this number\n")
    if (best.get("avg_gpu_util_pct") or 100) < 70:
        sys.stdout.write("note: GPU under 70% busy, the CPU side is the limit; try NUM_SCHEDULER_STEPS=4, fewer UPLOAD_THREADS or a newer vLLM\n")
    return 0


def split(value, cast, default):
    if not value:
        return [default]
    return [cast(v.strip()) for v in value.split(",")]


def sweep(args):
    os.makedirs(args.out_dir, exist_ok=True)
    seqs_list = split(args.seqs, int, 0)
    micro_list = split(args.microbatch, int, 0)
    util_list = split(args.gpu_util_list, float, 0.90)
    mask_list = split(args.masks, str, "")
    step_list = split(args.steps, int, 0)
    kv_list = split(args.kv_dtypes, str, "")
    token_list = split(args.batched_tokens_list, int, -1)
    chunk_list = split(args.chunked_list, str, "")
    thread_list = split(args.upload_threads_list, int, 0)
    rows = []
    for util, micro, mask, steps, kv, tokens, chunked, threads, seqs in itertools.product(
        util_list, micro_list, mask_list, step_list, kv_list, token_list, chunk_list, thread_list, seqs_list
    ):
        cfg = {
            "seqs": seqs,
            "micro": micro,
            "util": util,
            "mask": mask,
            "steps": steps,
            "kv": kv,
            "tokens": tokens,
            "chunked": chunked,
            "threads": threads,
        }
        sys.stdout.write(
            f"running seqs={seqs} microbatch={micro} util={util} mask={mask or 'default'} "
            f"steps={steps or 1} kv={kv or 'default'} batched_tokens={tokens if tokens >= 0 else 'default'} "
            f"chunked={chunked or 'default'} upload_threads={threads or 'default'}\n"
        )
        sys.stdout.flush()
        rows.append(run_child(args, cfg))
    rows.sort(key=lambda r: (r.get("error") is not None, r.get("cost_per_1000_usd") or 1e9))
    sys.stdout.write("\n")
    print_rows(rows)
    with open(os.path.join(args.out_dir, "report.json"), "w") as handle:
        json.dump({"results": rows}, handle, indent=2)
    return report_best(rows, args.max_fail)


def compare(args):
    rows = []
    for path in args.compare:
        with open(path) as handle:
            rows.extend(json.load(handle)["results"])
    rows.sort(key=lambda r: (r.get("error") is not None, r.get("cost_per_1000_usd") or 1e9))
    print_rows(rows)
    return report_best(rows, args.max_fail)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--compare", nargs="+")
    parser.add_argument("--count", type=int, default=480)
    parser.add_argument("--texts", default="", help="json file: list of {language, text} used instead of the built-in sentences")
    parser.add_argument("--seqs", default="64,80,112")
    parser.add_argument("--microbatch", default="8,16")
    parser.add_argument("--gpu-util-list", default="0.90")
    parser.add_argument("--masks", default="", help="comma list of range,none (empty = configured default)")
    parser.add_argument("--steps", default="", help="comma list of num_scheduler_steps (empty = 1)")
    parser.add_argument("--kv-dtypes", default="", help="comma list such as auto,fp8 (empty = configured default)")
    parser.add_argument("--batched-tokens-list", default="", help="comma list of max_num_batched_tokens (empty = configured default)")
    parser.add_argument("--chunked-list", default="", help="comma list of 0,1 for chunked prefill (empty = configured default)")
    parser.add_argument("--upload-threads-list", default="", help="comma list of upload thread counts (empty = configured default)")
    parser.add_argument("--max-num-seqs", type=int, default=80)
    parser.add_argument("--decode-microbatch", type=int, default=16)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--logits-mask", default="")
    parser.add_argument("--scheduler-steps", type=int, default=0)
    parser.add_argument("--kv-dtype", default="")
    parser.add_argument("--batched-tokens", type=int, default=-1)
    parser.add_argument("--chunked", default="")
    parser.add_argument("--upload-threads", type=int, default=0)
    parser.add_argument("--rate", type=float, default=float(os.environ.get("GPU_HOURLY_RATE", "0.28") or 0.28))
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-fail", type=float, default=0.02)
    parser.add_argument("--out-dir", default="/tmp/bench")
    return parser.parse_args()


def main():
    args = parse()
    if args.single:
        single(args)
        return 0
    if args.compare:
        return compare(args)
    return sweep(args)


if __name__ == "__main__":
    sys.exit(main())