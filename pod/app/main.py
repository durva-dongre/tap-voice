import asyncio
import logging
import os
import signal
import sys
import time

from . import config, manifest as manifest_mod
from .codec import Codec
from .control import ClientError, Control, Heartbeat, Stop, stop_event, terminate_self
from .engine import Engine, FatalEngineError
from .pipeline import Pipeline
from .storage import Store
from .telemetry import Telemetry
from .voices import ALLOWED_VOICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("main")

# Both are BaseException so no library's `except Exception` can swallow them.
class Timeout(BaseException):
    pass


class StartupTimeout(BaseException):
    pass


_PHASE = {"name": "startup"}


def _alarm(*_):
    if _PHASE["name"] == "startup":
        raise StartupTimeout()
    raise Timeout()


def report_rejected(control, rejected):
    if not rejected:
        return
    records = [
        {"id": r.id, "status": "failed", "url": None, "error": r.reason, "content_hash": None}
        for r in rejected
    ]
    try:
        control.progress(records)
    except Stop:
        stop_event.set()
    except ClientError as exc:
        log.warning("reject_report_failed %s", exc)
    except Exception as exc:
        log.warning("reject_report_failed %s", type(exc).__name__)


def report_existing(control, store, items, telemetry):
    """Everything is already stored: tell the server and finish without touching the GPU."""
    records = [
        {
            "id": item.id,
            "status": "done",
            "url": store.url_for_key(store.key_for(item)),
            "error": None,
            "content_hash": item.content_hash,
        }
        for item in items
    ]
    for start in range(0, len(records), 200):
        control.progress(records[start : start + 200])
    telemetry.add("skipped_existing", len(records))


async def load_engine(settings, engine):
    # engine.start() blocks the event loop while the model loads, so asyncio cannot time it
    # out. The SIGALRM set in body() is the real guard.
    await engine.start()
    return Codec(
        settings.snac_dir, half=settings.snac_half, tolerance=settings.bad_code_tolerance
    )


async def body(settings, control, heartbeat, telemetry, deadline):
    heartbeat.set("fetching_manifest", 0)
    payload = control.manifest()
    items, rejected = manifest_mod.parse(payload.get("items", []), settings)
    telemetry.add("rejected", len(rejected))
    report_rejected(control, rejected)
    if not items:
        return "empty"

    heartbeat.set("loading_storage", len(items))
    store = Store(settings)
    store.load_existing(sorted({item.language for item in items}))
    pending = [item for item in items if not store.exists(store.key_for(item))]
    if not pending:
        report_existing(control, store, items, telemetry)
        return "completed"

    heartbeat.set("loading_engine", len(pending))
    _PHASE["name"] = "startup"
    signal.alarm(settings.startup_timeout_seconds + settings.startup_alarm_margin_seconds)
    engine = Engine(settings)
    try:
        codec = await load_engine(settings, engine)
        await asyncio.wait_for(
            engine.warmup(sorted(ALLOWED_VOICES)), timeout=settings.startup_timeout_seconds
        )
        codec.warmup()
    except asyncio.TimeoutError as exc:
        await engine.shutdown()
        raise StartupTimeout() from exc
    telemetry.mark_ready()

    _PHASE["name"] = "run"
    remaining_budget = max(60, int(deadline - time.time()))
    signal.alarm(min(config.run_limit_seconds(settings, len(pending)), remaining_budget))
    pipeline = Pipeline(settings, engine, codec, store, control, telemetry, heartbeat)
    try:
        failed, tripped = await pipeline.run(items)
    finally:
        await engine.shutdown()
    if tripped:
        return f"breaker_{tripped}"
    if stop_event.is_set():
        return "stopped"
    return "completed" if failed == 0 else "completed_with_failures"


def map_reason(exc):
    if isinstance(exc, Timeout):
        return "timeout"
    if isinstance(exc, StartupTimeout):
        return "startup_timeout"
    if isinstance(exc, Stop):
        return "stopped"
    return "exception"


def main():
    """Runs one batch. Always reports completion and asks RunPod to delete the pod.
    Returns a process exit code."""
    started = time.time()
    reason = "completed"
    settings = None
    control = None
    heartbeat = None
    telemetry = None
    try:
        settings = config.get()
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(settings.pod_limit_seconds)
        control = Control(settings)
        telemetry = Telemetry(settings.run_id)
        heartbeat = Heartbeat(control, settings.beat_seconds)
        heartbeat.start()
        deadline = started + settings.pod_limit_seconds
        reason = asyncio.run(body(settings, control, heartbeat, telemetry, deadline))
    except (Timeout, StartupTimeout, Stop, FatalEngineError) as exc:
        reason = map_reason(exc)
    except Exception:
        log.exception("fatal")
        reason = "exception"
    finally:
        signal.alarm(0)
        if heartbeat is not None:
            heartbeat.done.set()
        gpu_seconds = time.time() - started
        stats = {}
        if telemetry is not None and settings is not None:
            stats = telemetry.summary(settings.gpu_hourly_rate)
        if control is not None:
            try:
                control.complete(reason, gpu_seconds, stats)
            except Exception:
                log.exception("complete_failed")
        terminate_self()
    return 0 if reason.startswith("completed") or reason == "empty" else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # vLLM background threads can keep the interpreter alive after the work is done.
    os._exit(code)