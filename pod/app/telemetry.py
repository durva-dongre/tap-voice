import json
import sys
import threading
import time


class Telemetry:
    def __init__(self, run_id):
        self.run_id = run_id
        self.start = time.time()
        self.lock = threading.Lock()
        self.counters = {
            "planned": 0,
            "skipped_existing": 0,
            "rejected": 0,
            "generated": 0,
            "decoded": 0,
            "uploaded": 0,
            "failed": 0,
            "retried": 0,
            "tokens": 0,
            "audio_seconds": 0.0,
        }
        self.reasons = {}

    def emit(self, event, **fields):
        record = {"ts": round(time.time(), 3), "run": self.run_id, "event": event}
        record.update(fields)
        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def add(self, name, value=1):
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value

    def fail(self, reason):
        with self.lock:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            self.counters["failed"] += 1

    def value(self, name):
        with self.lock:
            return self.counters.get(name, 0)

    def elapsed(self):
        return time.time() - self.start

    def spend(self, hourly_rate):
        return self.elapsed() / 3600.0 * hourly_rate

    def snapshot(self):
        with self.lock:
            counters = dict(self.counters)
            reasons = dict(self.reasons)
        elapsed = max(time.time() - self.start, 1e-6)
        counters["elapsed_seconds"] = round(elapsed, 2)
        counters["tokens_per_second"] = round(counters["tokens"] / elapsed, 1)
        minutes = counters["audio_seconds"] / 60.0
        counters["gpu_seconds_per_audio_minute"] = (
            round(elapsed / minutes, 2) if minutes > 0 else None
        )
        counters["failure_reasons"] = reasons
        return counters

    def summary(self, hourly_rate):
        snap = self.snapshot()
        cost = snap["elapsed_seconds"] / 3600.0 * hourly_rate
        snap["cost_proxy_usd"] = round(cost, 4)
        uploaded = snap["uploaded"]
        snap["cost_per_1000_clips_usd"] = (
            round(cost / uploaded * 1000.0, 4) if uploaded else None
        )
        self.emit("summary", **snap)
        return snap