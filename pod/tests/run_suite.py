import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT, "tests")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STAGE_ORDER = ("selftest", "bench", "e2e")
LOG_LIMIT = 3000
UPLOAD_LOG_LINES = 1500
MAX_FAIL_RATE = 0.02
# The first batch of results the server sees must be a small part of the run.
MAX_FIRST_BATCH_SHARE = 0.25

BENCH_KEYS = (
    "gpu_name",
    "max_num_seqs",
    "decode_microbatch",
    "gpu_memory_utilization",
    "clips_ok",
    "clips_total",
    "fail_rate",
    "failure_reasons",
    "truncated_clips",
    "startup_seconds",
    "wall_seconds",
    "clips_per_minute",
    "tokens_per_second",
    "gpu_seconds_per_audio_minute",
    "peak_vram_mb",
    "avg_gpu_util_pct",
    "cost_per_1000_usd",
    "error",
)

STAT_KEYS = (
    "planned",
    "skipped_existing",
    "rejected",
    "generated",
    "decoded",
    "uploaded",
    "truncated",
    "failed",
    "failed_attempts",
    "retried",
    "tokens",
    "audio_seconds",
    "elapsed_seconds",
    "startup_seconds",
    "run_seconds",
    "tokens_per_second",
    "gpu_seconds_per_audio_minute",
    "cost_proxy_usd",
    "cost_per_1000_clips_usd",
    "marginal_cost_per_1000_clips_usd",
    "marginal_seconds_per_clip",
    "failure_reasons",
)


def env_str(name, default):
    value = os.environ.get(name, "").strip()
    return value or default


def env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def env_float(name, default):
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def env_flag(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        ).stdout.decode().strip().splitlines()[0]
        name, total = [part.strip() for part in out.split(",")]
        return {"name": name, "memory_mb": float(total)}
    except Exception:
        return {"name": "unknown", "memory_mb": None}


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_stage(name, command, env, timeout):
    started = time.time()
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    finished = threading.Event()
    flags = {"timed_out": False}

    def watchdog():
        if not finished.wait(timeout):
            flags["timed_out"] = True
            kill_group(proc)

    threading.Thread(target=watchdog, daemon=True).start()
    lines = collections.deque(maxlen=LOG_LIMIT)
    for line in proc.stdout:
        text = line.rstrip("\n")
        lines.append(text)
        sys.stdout.write(f"[{name}] {text}\n")
        sys.stdout.flush()
    code = proc.wait()
    finished.set()
    return {
        "code": code,
        "timed_out": flags["timed_out"],
        "seconds": round(time.time() - started, 1),
        "lines": list(lines),
    }


def stage_record(name, ok, result, details, note=""):
    return {
        "stage": name,
        "ok": bool(ok),
        "seconds": result["seconds"] if result else 0.0,
        "timed_out": result["timed_out"] if result else False,
        "exit_code": result["code"] if result else None,
        "note": note,
        "details": details,
    }


def stage_selftest(timeout):
    env = dict(os.environ)
    env["SELFTEST"] = "1"
    result = run_stage("selftest", [sys.executable, "-m", "app.selftest"], env, timeout)
    text = "\n".join(result["lines"])
    ok = result["code"] == 0 and "selftest_ok" in text
    note = ""
    if not ok:
        failed = [line for line in result["lines"] if "selftest_failed" in line]
        note = failed[-1] if failed else ("timeout" if result["timed_out"] else "selftest did not finish")
    return stage_record("selftest", ok, result, {}, note), result["lines"]


def compact(row):
    return {key: row[key] for key in BENCH_KEYS if key in row}


def stage_bench(timeout, rate):
    out_dir = "/tmp/bench"
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.json")
    if os.path.exists(report_path):
        os.remove(report_path)
    command = [
        sys.executable,
        "-m",
        "app.bench",
        "--count",
        str(env_int("BENCH_COUNT", 120)),
        "--seqs",
        env_str("BENCH_SEQS", "32,64"),
        "--microbatch",
        env_str("BENCH_MICROBATCH", "8"),
        "--gpu-util-list",
        env_str("BENCH_GPU_UTIL", "0.90"),
        "--rate",
        str(rate),
        "--out-dir",
        out_dir,
    ]
    result = run_stage("bench", command, dict(os.environ), timeout)
    rows = []
    try:
        with open(report_path) as handle:
            rows = json.load(handle)["results"]
    except Exception:
        rows = []
    valid = [
        row
        for row in rows
        if not row.get("error")
        and row.get("cost_per_1000_usd") is not None
        and row.get("fail_rate", 1.0) <= MAX_FAIL_RATE
    ]
    best = min(valid, key=lambda row: row["cost_per_1000_usd"]) if valid else None
    details = {"configs": [compact(row) for row in rows], "best": compact(best) if best else None}
    if best:
        per_clip = best["wall_seconds"] / max(best["clips_ok"], 1)
        details["recommended_env"] = {
            "MAX_NUM_SEQS": str(best["max_num_seqs"]),
            "DECODE_MICROBATCH": str(best["decode_microbatch"]),
            "GPU_MEMORY_UTILIZATION": str(best["gpu_memory_utilization"]),
            "SECONDS_PER_ITEM_CAP": str(round(per_clip * 2, 2)),
        }
    ok = best is not None
    note = ""
    if not ok:
        note = "timeout" if result["timed_out"] else "no configuration met the failure-rate limit"
    return stage_record("bench", ok, result, details, note), result["lines"]


def wait_for(url, seconds):
    end = time.time() + seconds
    while time.time() < end:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def check_urls(samples):
    checks = []
    for language, urls in samples.items():
        for url in urls[:1]:
            try:
                response = requests.get(url, timeout=30)
                content_type = response.headers.get("Content-Type", "")
                good = (
                    response.status_code == 200
                    and "audio/ogg" in content_type
                    and len(response.content) > 500
                )
                checks.append(
                    {
                        "language": language,
                        "url": url,
                        "status": response.status_code,
                        "content_type": content_type,
                        "bytes": len(response.content),
                        "ok": good,
                    }
                )
            except requests.RequestException as exc:
                checks.append({"language": language, "url": url, "ok": False, "error": type(exc).__name__})
    return checks


def progress_is_incremental(incremental, total):
    """True when results reached the server while the run was going, not all at the end."""
    if total < 20:
        return True
    if incremental.get("progress_calls", 0) < 3:
        return False
    return incremental.get("first_done_batch", total) <= max(5, total * MAX_FIRST_BATCH_SHARE)


def project_daily_cost(stats, rate, clips_per_batch, batches_per_day):
    """Fixed startup plus marginal per-clip time, scaled to a real batch.

    Not included: pod scheduling and image pull time before the container starts. Add the
    'Uptime' RunPod shows for a pod minus the elapsed_seconds in this report to estimate it.
    """
    startup = stats.get("startup_seconds")
    per_clip = stats.get("marginal_seconds_per_clip")
    if startup is None or per_clip is None:
        return None
    batch_seconds = startup + clips_per_batch * per_clip
    batch_cost = batch_seconds / 3600.0 * rate
    return {
        "clips_per_batch": clips_per_batch,
        "batches_per_day": batches_per_day,
        "startup_seconds": startup,
        "marginal_seconds_per_clip": per_clip,
        "batch_seconds": round(batch_seconds, 1),
        "cost_per_batch_usd": round(batch_cost, 4),
        "cost_per_day_usd": round(batch_cost * batches_per_day, 4),
        "cost_per_month_usd": round(batch_cost * batches_per_day * 30, 2),
        "cost_per_1000_clips_at_this_batch_size_usd": round(batch_cost / clips_per_batch * 1000.0, 4),
        "excludes": "pod scheduling and image pull time before the container starts",
    }


def stage_e2e(timeout, run_id, rate):
    count = env_int("TEST_E2E_COUNT", 200)
    port = env_int("MOCK_PORT", 8000)
    report_path = "/tmp/e2e_mock_report.json"
    if os.path.exists(report_path):
        os.remove(report_path)
    server_env = dict(os.environ)
    server_env.update({"MOCK_PORT": str(port), "MOCK_REPORT": report_path, "MOCK_COUNT": str(count)})
    server_log = open("/tmp/mock_server.log", "w")
    server = subprocess.Popen(
        [sys.executable, os.path.join(TESTS_DIR, "mock_server.py")],
        cwd=ROOT,
        env=server_env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        if not wait_for(f"http://127.0.0.1:{port}/health", 30):
            return stage_record("e2e", False, None, {}, "mock server did not start"), []
        env = dict(os.environ)
        env.pop("SELFTEST", None)
        env.update(
            {
                "SERVER_URL": f"http://127.0.0.1:{port}",
                "RUN_ID": run_id,
                "RUN_TOKEN": "test",
                "MODEL_REVISION": f"{env_str('MODEL_REVISION', 'test')}-{run_id}",
                "GCS_PREFIX": env_str("TEST_GCS_PREFIX", "test-e2e"),
                "GPU_HOURLY_RATE": str(rate),
                # The parent stops the pod; the child must not.
                "RUNPOD_API_KEY": "",
            }
        )
        if not env.get("SECONDS_PER_ITEM_CAP", "").strip():
            env["SECONDS_PER_ITEM_CAP"] = "6"
        if not env.get("MAX_SPEND_USD", "").strip():
            env["MAX_SPEND_USD"] = "3"
        result = run_stage("e2e", [sys.executable, "-m", "app.main"], env, timeout)
    finally:
        kill_group(server)
        server_log.close()
    try:
        with open(report_path) as handle:
            report = json.load(handle)
    except Exception:
        note = "timeout" if result["timed_out"] else "backend never received the completion call"
        return stage_record("e2e", False, result, {}, note), result["lines"]
    total = max(report["items_in_manifest"], 1)
    fail_rate = report["failed"] / float(total)
    complete = report.get("complete") or {}
    stats = complete.get("stats") or {}
    checks = check_urls(report["sample_urls"]) if env_flag("TEST_URL_CHECK", True) else []
    urls_ok = all(check["ok"] for check in checks)
    incremental = report.get("incremental", {})
    incremental_ok = progress_is_incremental(incremental, total)
    projection = project_daily_cost(
        stats,
        rate,
        env_int("PROJECT_CLIPS_PER_BATCH", 2000),
        env_int("PROJECT_BATCHES_PER_DAY", 2),
    )
    details = {
        "clips_requested": report["items_in_manifest"],
        "done": report["done"],
        "failed": report["failed"],
        "fail_rate": round(fail_rate, 4),
        "failure_reasons": report["failure_reasons"],
        "done_by_language": report["done_by_language"],
        "finish_reason": complete.get("reason"),
        "gpu_seconds": complete.get("gpu_seconds"),
        "stats": {key: stats[key] for key in STAT_KEYS if key in stats},
        "progress_arrival": incremental,
        "progress_incremental": incremental_ok,
        "cost_projection": projection,
        "gcs_prefix": env["GCS_PREFIX"],
        "sample_urls": report["sample_urls"],
        "url_checks": checks,
    }
    ok = report["done"] > 0 and fail_rate <= MAX_FAIL_RATE and urls_ok and incremental_ok
    note = ""
    if not ok:
        if not urls_ok:
            note = "uploaded files are not publicly readable as audio/ogg"
        elif not incremental_ok:
            note = "results reached the server in one lump at the end instead of as clips finished"
        elif fail_rate > MAX_FAIL_RATE:
            note = "failure rate above limit"
        else:
            note = "no clips completed"
    return stage_record("e2e", ok, result, details, note), result["lines"]


def publish(run_id, report, log_text):
    from app import config
    from app.storage import Store

    previous = os.environ.get("SELFTEST")
    os.environ["SELFTEST"] = "1"
    try:
        settings = config.load()
    finally:
        if previous is None:
            os.environ.pop("SELFTEST", None)
        else:
            os.environ["SELFTEST"] = previous
    store = Store(settings)
    prefix = env_str("TEST_REPORT_PREFIX", "benchmarks")
    files = (
        ("report.json", json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"), "application/json"),
        ("log.txt", log_text.encode("utf-8"), "text/plain; charset=utf-8"),
    )
    urls = {}
    for filename, data, content_type in files:
        urls[filename] = store.upload(f"{prefix}/{run_id}/{filename}", data, content_type)
    return urls


def describe(record):
    status = "PASS" if record["ok"] else "FAIL"
    text = f"{record['stage']}: {status} ({record['seconds']}s)"
    details = record["details"]
    if record["stage"] == "bench" and details.get("best"):
        best = details["best"]
        text += (
            f" best seqs={best['max_num_seqs']} microbatch={best['decode_microbatch']}"
            f" cost_per_1000=${best['cost_per_1000_usd']} clips_per_min={best['clips_per_minute']}"
        )
    if record["stage"] == "e2e" and details:
        stats = details.get("stats", {})
        text += (
            f" done={details['done']}/{details['clips_requested']}"
            f" marginal_cost_per_1000=${stats.get('marginal_cost_per_1000_clips_usd')}"
            f" startup_s={stats.get('startup_seconds')}"
            f" tokens_per_second={stats.get('tokens_per_second')}"
        )
        projection = details.get("cost_projection")
        if projection:
            text += (
                f" | projected {projection['clips_per_batch']} clips x {projection['batches_per_day']}/day"
                f" = ${projection['cost_per_day_usd']}/day"
            )
    if record["note"]:
        text += f" note={record['note']}"
    return text


def main():
    started = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    requested = [name.strip() for name in env_str("TEST_STAGES", ",".join(STAGE_ORDER)).split(",") if name.strip()]
    unknown = [name for name in requested if name not in STAGE_ORDER]
    if unknown:
        sys.stderr.write("unknown test stages: " + ",".join(unknown) + "\n")
        return 2
    stages = [name for name in STAGE_ORDER if name in requested]
    deadline = started + env_int("TEST_TIMEOUT_MINUTES", 90) * 60
    rate = env_float("GPU_HOURLY_RATE", 0.34)
    results = []
    logs = []
    for name in stages:
        remaining = deadline - time.time()
        if remaining < 30:
            results.append(stage_record(name, False, None, {}, "skipped: time budget used up"))
            continue
        if name == "e2e" and any(r["stage"] == "selftest" and not r["ok"] for r in results):
            results.append(stage_record(name, False, None, {}, "skipped: selftest failed"))
            continue
        sys.stdout.write(f"=== stage {name} started ===\n")
        sys.stdout.flush()
        try:
            if name == "selftest":
                record, lines = stage_selftest(remaining)
            elif name == "bench":
                record, lines = stage_bench(remaining, rate)
            else:
                record, lines = stage_e2e(remaining, run_id, rate)
        except Exception as exc:
            record, lines = stage_record(name, False, None, {}, f"harness error {type(exc).__name__}: {exc}"), []
        results.append(record)
        logs.append((name, lines))
    passed = bool(results) and all(record["ok"] for record in results)
    report = {
        "run_id": run_id,
        "mode": "test",
        "verdict": "pass" if passed else "fail",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_seconds": round(time.time() - started, 1),
        "gpu": gpu_info(),
        "gpu_hourly_rate": rate,
        "model_revision": env_str("MODEL_REVISION", "test"),
        "stages": results,
    }
    log_parts = []
    for name, lines in logs:
        log_parts.append(f"===== {name} =====")
        log_parts.extend(lines[-UPLOAD_LOG_LINES:])
    log_text = "\n".join(log_parts) + "\n"
    sys.stdout.write("=== TEST SUMMARY ===\n")
    for record in results:
        sys.stdout.write(describe(record) + "\n")
    sys.stdout.write(f"verdict: {report['verdict']}\n")
    sys.stdout.flush()
    try:
        urls = publish(run_id, report, log_text)
    except Exception as exc:
        sys.stdout.write(f"report_upload_failed {type(exc).__name__}: {exc}\n")
        sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 3
    sys.stdout.write("report: " + urls["report.json"] + "\n")
    sys.stdout.write("log: " + urls["log.txt"] + "\n")
    sys.stdout.flush()
    return 0 if passed else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)