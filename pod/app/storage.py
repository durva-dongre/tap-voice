import base64
import functools
import json
import os
import random
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

CACHE_CONTROL = "public, max-age=31536000, immutable"

CONTENT_TYPES = {
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
}

_SERVER = None
_SERVER_LOCK = threading.Lock()


class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        return

    def guess_type(self, path):
        extension = os.path.splitext(path)[1].lower()
        return CONTENT_TYPES.get(extension, "application/octet-stream")


def serve_local(directory, port):
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is not None:
            return _SERVER
        os.makedirs(directory, exist_ok=True)
        handler = functools.partial(_Handler, directory=directory)
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        except OSError:
            return None
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _SERVER = server
        return server


def object_key(prefix, language, content_hash, ext):
    return f"{prefix}/{language}/{content_hash[:32]}.{ext}"


class Store:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.existing = set()
        self.lock = threading.Lock()
        self.local = settings.storage_backend == "local"
        self.client = None
        self.bucket = None
        self.transient = ()
        self.precondition_failed = None
        if self.local:
            self.root = os.path.abspath(settings.local_dir)
            os.makedirs(self.root, exist_ok=True)
            serve_local(self.root, settings.local_port)
            return
        from google.api_core import exceptions as gexc
        from google.cloud import storage
        from google.oauth2 import service_account

        self.transient = (
            gexc.TooManyRequests,
            gexc.InternalServerError,
            gexc.BadGateway,
            gexc.ServiceUnavailable,
            gexc.GatewayTimeout,
            ConnectionError,
            TimeoutError,
        )
        self.precondition_failed = gexc.PreconditionFailed
        if client is None:
            info = json.loads(base64.b64decode(settings.gcs_credentials_b64).decode("utf-8"))
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
            )
            client = storage.Client(credentials=credentials, project=info.get("project_id"))
        self.client = client
        self.bucket = client.bucket(settings.gcs_bucket)

    def key_for(self, item):
        return object_key(self.settings.gcs_prefix, item.language, item.content_hash, item.fmt)

    def url_for_key(self, key):
        if self.local:
            return f"http://127.0.0.1:{self.settings.local_port}/{key}"
        return f"{self.settings.cdn_base_url}/{key}"

    def load_existing(self):
        names = set()
        prefix = f"{self.settings.gcs_prefix}/"
        if self.local:
            for base, _, files in os.walk(self.root):
                for name in files:
                    rel = os.path.relpath(os.path.join(base, name), self.root)
                    rel = rel.replace(os.sep, "/")
                    if rel.startswith(prefix):
                        names.add(rel)
        else:
            for blob in self.client.list_blobs(
                self.settings.gcs_bucket,
                prefix=prefix,
                fields="items(name),nextPageToken",
            ):
                names.add(blob.name)
        with self.lock:
            self.existing = names
        return len(names)

    def exists(self, key):
        with self.lock:
            return key in self.existing

    def _remember(self, key):
        with self.lock:
            self.existing.add(key)

    def _upload_local(self, key, data):
        path = os.path.abspath(os.path.join(self.root, key))
        if not path.startswith(self.root + os.sep):
            raise ValueError("bad_key")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            tmp = f"{path}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        self._remember(key)
        return self.url_for_key(key)

    def upload(self, key, data, content_type):
        if self.local:
            return self._upload_local(key, data)
        blob = self.bucket.blob(key)
        blob.cache_control = CACHE_CONTROL
        attempt = 0
        while True:
            try:
                blob.upload_from_string(
                    data,
                    content_type=content_type,
                    if_generation_match=0,
                    retry=None,
                )
                self._remember(key)
                return self.url_for_key(key)
            except self.precondition_failed:
                self._remember(key)
                return self.url_for_key(key)
            except self.transient:
                attempt += 1
                if attempt > 5:
                    raise
                delay = min(20.0, (2 ** attempt) * 0.5) * (0.5 + random.random())
                time.sleep(delay)