import logging
import os
import threading
import time

import requests

log = logging.getLogger("control")

stop_event = threading.Event()


class Stop(Exception):
    pass


class ClientError(Exception):
    pass


class Control:
    def __init__(self, settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {"X-Run-Id": settings.run_id, "X-Run-Token": settings.run_token}
        )

    def _request(self, method, path, retries=4, timeout=30, backoff_cap=8, **kwargs):
        error = None
        for attempt in range(retries):
            try:
                response = self.session.request(
                    method, self.settings.server_url + path, timeout=timeout, **kwargs
                )
            except requests.RequestException as exc:
                error = exc
            else:
                code = response.status_code
                if code in (401, 403, 409):
                    raise Stop(f"http_{code}")
                if code < 400:
                    if not response.content:
                        return {}
                    return response.json()
                if code < 500 and code != 429:
                    raise ClientError(f"http_{code}")
                error = RuntimeError(f"http_{code}")
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, backoff_cap))
        raise error

    def manifest(self):
        return self._request("GET", "/internal/pod/manifest", timeout=60)

    def progress(self, records):
        return self._request(
            "POST",
            "/internal/pod/progress",
            json={"records": records},
            retries=5,
            timeout=45,
        )

    def complete(self, reason, gpu_seconds, stats):
        return self._request(
            "POST",
            "/internal/pod/complete",
            json={"reason": reason, "gpu_seconds": gpu_seconds, "stats": stats},
            retries=3,
            timeout=20,
            backoff_cap=4,
        )

    def beat(self, phase, remaining):
        response = self._request(
            "POST",
            "/internal/pod/beat",
            json={"phase": phase, "remaining": remaining},
            retries=1,
            timeout=10,
        )
        if response.get("stop"):
            stop_event.set()
        return response


class Heartbeat(threading.Thread):
    def __init__(self, control, interval):
        super().__init__(daemon=True)
        self.control = control
        self.interval = interval
        self.done = threading.Event()
        self.phase = "starting"
        self.remaining = 0

    def set(self, phase, remaining):
        self.phase = phase
        self.remaining = remaining

    def run(self):
        while not self.done.wait(self.interval):
            try:
                self.control.beat(self.phase, self.remaining)
            except Stop:
                stop_event.set()
            except Exception as exc:
                log.warning("heartbeat_failed %s", type(exc).__name__)


def terminate_self():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not api_key:
        return
    base = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1")
    try:
        requests.delete(
            f"{base}/pods/{pod_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
    except Exception as exc:
        log.error("terminate_self_failed %s", type(exc).__name__)