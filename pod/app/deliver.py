import logging, threading
from concurrent.futures import ThreadPoolExecutor
from . import config as C
from .api import Stop, stop_event
from .audio import to_ogg

log = logging.getLogger("deliver")

class Deliverer:
    def __init__(s, api):
        s.api, s.ex = api, ThreadPoolExecutor(C.DELIVER_WORKERS)
        s.sem = threading.BoundedSemaphore(C.DELIVER_WORKERS * 4)
    def submit(s, jid, audio, sr):
        s.sem.acquire()
        s.ex.submit(s._do, jid, audio, sr).add_done_callback(lambda _: s.sem.release())
    def _do(s, jid, audio, sr):
        try: s.api.deliver(jid, to_ogg(audio, sr))
        except Stop: stop_event.set()
        except Exception as e:
            log.warning("deliver %s failed: %s", jid, e)
            try: s.api.fail(jid, str(e)[:200], True)
            except Exception: pass
    def __enter__(s): return s
    def __exit__(s, et, *_): s.ex.shutdown(wait=True, cancel_futures=et is not None)
