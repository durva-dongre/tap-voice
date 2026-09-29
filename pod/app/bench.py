import argparse
import asyncio
import dataclasses
import json
import os
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PREFIX = "BENCH_RESULT "
LENGTHS = ("short", "medium", "long")

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


def build_items(settings, count):
    from app.manifest import Item, content_hash
    from app.voices import LANGUAGE_VOICES

    languages = list(SENTENCES)
    table = {language: texts_for(language) for language in languages}
    items = []
    for index in range(count):
        language = languages[index % len(languages)]
        length = LENGTHS[(index // len(languages)) % len(LENGTHS)]
        text = f"{table[language][length]} {index}"
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


async def run_single(settings, count):
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
    codec = Codec(settings.snac_dir)
    await engine.warmup(sorted(ALLOWED_VOICES))
    startup = time.time() - began

    items = build_items(settings, count)
    by_id = {item.id: item for item in items}
    prepared = sort_by_length(prepare(items, engine.tokenizer, settings))

    lock = threading.Lock()
    stage = {"decode": 0.0, "encode": 0.0}
    failures = {}
    totals = {"tokens": 0, "truncated": 0}
    pool = ThreadPoolExecutor(max_workers=settings.upload_threads)
    futures = []
    buffer = []
    last = time.time()
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
            processed, reason = audio.process(wave, settings)
            if processed is None:
                outcome = ("fail", reason, 0.0)
            else:
                data, _, _ = audio.encode(processed, item.fmt, settings)
                outcome = ("ok", "", processed.size / float(settings.source_sample_rate))
                if not data:
                    outcome = ("fail", "empty_encode", 0.0)
        except Exception as exc:
            outcome = ("fail", type(exc).__name__, 0.0)
        with lock:
            stage["encode"] += time.time() - start
        return outcome

    async def drain():
        nonlocal buffer, last
        if not buffer:
            return
        entries = [(f.key, f.tokens) for f in buffer]
        buffer = []
        last = time.time()
        decoded, bad = await loop.run_in_executor(None, timed_decode, entries)
        for _, reason in bad:
            note(reason)
        for key, wave in decoded:
            futures.append(pool.submit(encode_job, by_id[key], wave))

    run_start = time.time()
    async for finished in engine.generate(prepared, lambda: False):
        if finished.error:
            note(finished.error)
            continue
        totals["tokens"] += len(finished.tokens)
        if finished.finish_reason == "length":
            totals["truncated"] += 1
        buffer.append(finished)
        if (
            len(buffer) >= settings.decode_microbatch
            or time.time() - last >= settings.decode_flush_seconds
        ):
            await drain()
    await drain()
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
    pool.shutdown(wait=True)
    sampler.halt.set()
    await engine.shutdown()

    hours = wall / 3600.0
    gpu = read_gpu()
    return {
        "gpu_name": gpu[0] if gpu else "unknown",
        "model_dir": settings.model_dir,
        "max_num_seqs": settings.max_num_seqs,
        "decode_microbatch": settings.decode_microbatch,
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "gpu_hourly_rate": settings.gpu_hourly_rate,
        "clips_total": count,
        "clips_ok": ok,
        "fail_rate": round((count - ok) / float(count), 4),
        "failure_reasons": failures,
        "truncated_clips": totals["truncated"],
        "startup_seconds": round(startup, 1),
        "wall_seconds": round(wall, 2),
        "generate_wall_seconds": round(generate_wall, 2),
        "decode_seconds": round(stage["decode"], 2),
        "encode_seconds": round(stage["encode"], 2),
        "clips_per_minute": round(ok / (wall / 60.0), 1) if wall > 0 else 0.0,
        "tokens_per_second": round(totals["tokens"] / wall, 1) if wall > 0 else 0.0,
        "audio_seconds": round(audio_seconds, 1),
        "gpu_seconds_per_audio_minute": round(wall / (audio_seconds / 60.0), 2) if audio_seconds else None,
        "peak_vram_mb": round(sampler.peak_mb),
        "avg_gpu_util_pct": sampler.average_util(),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "cost_per_1000_usd": round(hours / ok * 1000.0 * settings.gpu_hourly_rate, 4) if ok else None,
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
    if gpu:
        overrides["gpu_total_mb"] = int(gpu[2])
    if args.model_dir:
        overrides["model_dir"] = args.model_dir
    settings = dataclasses.replace(settings, **overrides)
    result = asyncio.run(run_single(settings, args.count))
    sys.stdout.write(PREFIX + json.dumps(result) + "\n")
    sys.stdout.flush()
    os._exit(0)


def run_child(args, seqs, micro, util):
    command = [
        sys.executable,
        "-m",
        "app.bench",
        "--single",
        "--count",
        str(args.count),
        "--max-num-seqs",
        str(seqs),
        "--decode-microbatch",
        str(micro),
        "--gpu-util",
        str(util),
        "--rate",
        str(args.rate),
    ]
    if args.model_dir:
        command += ["--model-dir", args.model_dir]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(args.out_dir, f"run_s{seqs}_m{micro}_u{util}.log")
    base = {"max_num_seqs": seqs, "decode_microbatch": micro, "gpu_memory_utilization": util}
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
            return json.loads(line[len(PREFIX):])
    return dict(base, error=f"exit_{proc.returncode}", log=log_path)


def print_rows(rows):
    header = f"{'gpu':<20}{'seqs':>5}{'mb':>4}{'util':>6}{'ok':>11}{'clips/min':>10}{'tok/s':>8}{'vramMB':>8}{'rssMB':>7}{'$/1000':>9}"
    sys.stdout.write(header + "\n")
    for row in rows:
        name = str(row.get("gpu_name", "-"))[:19]
        lead = f"{name:<20}{row['max_num_seqs']:>5}{row['decode_microbatch']:>4}{row['gpu_memory_utilization']:>6}"
        if row.get("error"):
            sys.stdout.write(lead + f"  FAILED: {row['error']}\n")
            continue
        ok = f"{row['clips_ok']}/{row['clips_total']}"
        cost = row["cost_per_1000_usd"]
        cost_text = f"{cost:.4f}" if cost is not None else "-"
        sys.stdout.write(
            lead
            + f"{ok:>11}{row['clips_per_minute']:>10}{row['tokens_per_second']:>8}"
            + f"{row['peak_vram_mb']:>8}{row['peak_rss_mb']:>7}{cost_text:>9}\n"
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
        f"{best['clips_per_minute']} clips/min, peak VRAM {best['peak_vram_mb']} MB\n"
        f"MAX_NUM_SEQS={best['max_num_seqs']} DECODE_MICROBATCH={best['decode_microbatch']} "
        f"GPU_MEMORY_UTILIZATION={best['gpu_memory_utilization']}\n"
    )
    per_clip = best["wall_seconds"] / max(best["clips_ok"], 1)
    sys.stdout.write(f"suggested SECONDS_PER_ITEM_CAP={round(per_clip * 2, 2)} (2x measured {per_clip:.2f}s/clip)\n")
    return 0


def sweep(args):
    os.makedirs(args.out_dir, exist_ok=True)
    seqs_list = [int(v) for v in args.seqs.split(",")]
    micro_list = [int(v) for v in args.microbatch.split(",")]
    util_list = [float(v) for v in args.gpu_util_list.split(",")]
    rows = []
    for util in util_list:
        for micro in micro_list:
            for seqs in seqs_list:
                sys.stdout.write(f"running seqs={seqs} microbatch={micro} util={util}\n")
                sys.stdout.flush()
                rows.append(run_child(args, seqs, micro, util))
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
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seqs", default="16,32,48,64")
    parser.add_argument("--microbatch", default="8")
    parser.add_argument("--gpu-util-list", default="0.90")
    parser.add_argument("--max-num-seqs", type=int, default=48)
    parser.add_argument("--decode-microbatch", type=int, default=8)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--rate", type=float, default=float(os.environ.get("GPU_HOURLY_RATE", "0.34") or 0.34))
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