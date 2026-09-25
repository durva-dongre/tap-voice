import logging, os, threading, time
import requests
from . import config as C

log = logging.getLogger("api")
stop_event = threading.Event()

class Stop(Exception): ...
class ClientError(Exception): ...

class Api:
    def __init__(s):
        s.s = requests.Session()
        s.s.headers.update({"X-Run-Id": C.RUN_ID, "X-Run-Token": C.RUN_TOKEN})

    def _req(s, method, path, retries=3, timeout=30, backoff_cap=2, **kw):
        err = None
        for i in range(retries):
            try:
                r = s.s.request(method, C.WORKER_URL + path, timeout=timeout, **kw)
            except requests.RequestException as e:
                err = e
            else:
                if r.status_code in (401, 403, 409): raise Stop(f"{r.status_code} {r.text[:80]}")
                if r.status_code < 400: return r.json()
                if r.status_code < 500 and r.status_code != 429: raise ClientError(f"{r.status_code} {r.text[:120]}")
                err = RuntimeError(str(r.status_code))
            if i < retries - 1: time.sleep(min(2 ** i, backoff_cap))
        raise err

    def claim(s, limit): return s._req("POST", "/pod/claim", json={"limit": limit})["jobs"]
    def deliver(s, jid, ogg): return s._req("POST", f"/pod/deliver/{jid}", data=ogg, headers={"Content-Type": "audio/ogg"}, timeout=30, retries=2, backoff_cap=1)
    def fail(s, jid, err, retryable=True): return s._req("POST", f"/pod/fail/{jid}", json={"error": err, "retryable": retryable}, retries=2, backoff_cap=1)
    def complete(s, reason, gpu_s): return s._req("POST", "/pod/complete", retries=2, backoff_cap=1, timeout=15, json={"reason": reason, "gpu_seconds": gpu_s})
    def beat(s, phase, remaining, models):
        if s._req("POST", "/pod/beat", retries=1, timeout=10, json={"phase": phase, "remaining": remaining, "models": models}).get("stop"):
            stop_event.set()

class Heartbeat(threading.Thread):
    def __init__(s, api):
        super().__init__(daemon=True); s.api, s.done, s.state = api, threading.Event(), ("starting", 0, "none")
    def set(s, phase, remaining, models): s.state = (phase, remaining, models)
    def run(s):
        while not s.done.wait(C.BEAT_S):
            try: s.api.beat(*s.state)
            except Stop: stop_event.set()
            except Exception as e: log.warning("beat failed (non-fatal): %s", e)

def terminate_self():
    pid, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not (pid and key): return
    try:
        requests.delete(f"{os.environ.get('RUNPOD_API_BASE', 'https://api.runpod.io/v2')}/pods/{pid}",
                        headers={"Authorization": f"Bearer {key}"}, timeout=10)
    except Exception as e:
        log.error("self-terminate failed (%s) - Worker watchdog is the backstop", e)