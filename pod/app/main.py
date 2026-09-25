# pod/app/main.py
import gc, logging, signal, time
from . import config as C
from .api import Api, Heartbeat, Stop, stop_event, terminate_self
from .route import plan
from .deliver import Deliverer
from .engines import Kokoro, Indic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("main")

class Timeout(BaseException): ...
def _alarm(*_): raise Timeout()

def run_group(name, jobs, hb, dl, api, models, remaining):
    eng = Kokoro() if name == "kokoro" else Indic()
    sr = eng.sr
    for j in jobs:
        if stop_event.is_set():
            raise Stop("worker requested stop")
        try:
            dl.submit(j["id"], eng(j), sr)
        except Exception as e:
            log.exception("generate %s", j["id"])
            api.fail(j["id"], f"gen: {e}"[:200], True)
        remaining[0] -= 1
        hb.set(f"running_{name}", remaining[0], models)
    del eng
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

def main():
    t0, reason = time.time(), "completed"
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(C.LIMIT_S)
    api = Api()
    hb = Heartbeat(api)
    hb.start()
    try:
        jobs = api.claim(C.MAX_JOBS)
        if not jobs:
            reason = "empty"
            return
        groups, other, models = plan(jobs)
        for j in other:
            api.fail(j["id"], "unsupported_language", False)
        remaining = [len(jobs) - len(other)]
        hb.set("generating", remaining[0], models)
        with Deliverer(api) as dl:
            for name, g in groups:
                run_group(name, g, hb, dl, api, models, remaining)
    except Timeout:
        reason = "timeout"
    except Stop as e:
        reason = f"stopped:{e}"[:60]
    except Exception:
        log.exception("fatal")
        reason = "exception"
    finally:
        signal.alarm(0)
        hb.done.set()
        try:
            api.complete(reason, time.time() - t0)
        except Exception:
            log.exception("complete failed")
        terminate_self()

if __name__ == "__main__":
    main()