import asyncio
import logging
import signal
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


class Timeout(BaseException):
    pass


def _alarm(*_):
    raise Timeout()


class StartupTimeout(Exception):
    pass


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


async def load_engine(settings, engine, item_count):
    await asyncio.wait_for(engine.start(), timeout=settings.startup_timeout_seconds)
    return Codec(settings.snac_dir)


async def body(settings, control, heartbeat, telemetry, deadline):
    heartbeat.set("fetching_manifest", 0)
    payload = control.manifest()
    items, rejected = manifest_mod.parse(payload.get("items", []), settings)
    telemetry.add("rejected", len(rejected))
    report_rejected(control, rejected)
    if not items:
        return "empty"
    limit = config.hard_limit_seconds(settings, len(items))
    remaining_budget = max(60, int(deadline - time.time()))
    signal.alarm(min(limit, remaining_budget))
    heartbeat.set("loading_storage", len(items))
    store = Store(settings)
    store.load_existing()
    heartbeat.set("loading_engine", len(items))
    engine = Engine(settings)
    try:
        codec = await load_engine(settings, engine, len(items))
        await asyncio.wait_for(
            engine.warmup(sorted(ALLOWED_VOICES)), timeout=settings.startup_timeout_seconds
        )
    except asyncio.TimeoutError as exc:
        await engine.shutdown()
        raise StartupTimeout() from exc
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
    if isinstance(exc, FatalEngineError):
        return "exception"
    return "exception"


def main():
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


if __name__ == "__main__":
    main()