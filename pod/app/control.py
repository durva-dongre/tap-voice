import logging
import os
import sys
import threading
import time

import requests

log = logging.getLogger("control")

stop_event = threading.Event()

PROGRESS_CHUNK = 200
DEFAULT_PREFIX = "/internal/pod/"


class Stop(Exception):
    pass


class ClientError(Exception):
    pass


def unwrap(payload):
    if isinstance(payload, dict) and isinstance(payload.get("message"), dict):
        return payload["message"]
    return payload


def build_url(settings, endpoint):
    prefix = getattr(settings, "api_path_prefix", "") or DEFAULT_PREFIX
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return settings.server_url + prefix + endpoint


class Control:
    def __init__(self, settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {"X-Run-Id": settings.run_id, "X-Run-Token": settings.run_token}
        )

    def url(self, endpoint):
        return build_url(self.settings, endpoint)

    def _request(self, method, endpoint, retries=4, timeout=30, backoff_cap=8, **kwargs):
        error = None
        for attempt in range(retries):
            try:
                response = self.session.request(
                    method, self.url(endpoint), timeout=timeout, **kwargs
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
                    try:
                        return unwrap(response.json())
                    except ValueError:
                        error = RuntimeError("bad_json")
                elif code < 500 and code != 429:
                    raise ClientError(f"http_{code}")
                else:
                    error = RuntimeError(f"http_{code}")
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, backoff_cap))
        raise error

    def manifest(self):
        return self._request("POST", "manifest", json={}, timeout=60)

    def progress(self, records):
        result = {}
        for start in range(0, len(records), PROGRESS_CHUNK):
            result = self._request(
                "POST",
                "progress",
                json={"records": records[start : start + PROGRESS_CHUNK]},
                retries=5,
                timeout=45,
            )
        return result

    def complete(self, reason, gpu_seconds, stats):
        return self._request(
            "POST",
            "complete",
            json={"reason": reason, "gpu_seconds": gpu_seconds, "stats": stats},
            retries=3,
            timeout=20,
            backoff_cap=4,
        )

    def beat(self, phase, remaining):
        response = self._request(
            "POST",
            "beat",
            json={"phase": phase, "remaining": remaining},
            retries=1,
            timeout=10,
        )
        if isinstance(response, dict) and response.get("stop"):
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


def _say(message):
    sys.stderr.write(f"terminate_self {message}\n")
    sys.stderr.flush()


def terminate_self(attempts=3):
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not pod_id or not api_key:
        _say("skipped: RUNPOD_POD_ID or RUNPOD_API_KEY is not set")
        return False
    base = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(attempts):
        try:
            response = requests.delete(f"{base}/pods/{pod_id}", headers=headers, timeout=15)
            if response.status_code < 300 or response.status_code == 404:
                _say(f"ok delete http_{response.status_code}")
                return True
            _say(f"delete failed http_{response.status_code} {response.text[:200]}")
        except Exception as exc:
            _say(f"delete error {type(exc).__name__}")
        if attempt < attempts - 1:
            time.sleep(2 ** attempt)
    try:
        response = requests.post(f"{base}/pods/{pod_id}/stop", headers=headers, timeout=15)
        _say(f"fallback stop http_{response.status_code} {response.text[:200]}")
        return response.status_code < 300
    except Exception as exc:
        _say(f"fallback stop error {type(exc).__name__}")
        return False