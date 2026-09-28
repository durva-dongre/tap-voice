import base64
import json
import random
import threading
import time

from google.api_core import exceptions as gexc
from google.cloud import storage
from google.oauth2 import service_account

TRANSIENT = (
    gexc.TooManyRequests,
    gexc.InternalServerError,
    gexc.BadGateway,
    gexc.ServiceUnavailable,
    gexc.GatewayTimeout,
    ConnectionError,
    TimeoutError,
)

CACHE_CONTROL = "public, max-age=31536000, immutable"


def object_key(prefix, language, content_hash, ext):
    return f"{prefix}/{language}/{content_hash[:32]}.{ext}"


class Store:
    def __init__(self, settings, client=None):
        self.settings = settings
        if client is None:
            info = json.loads(base64.b64decode(settings.gcs_credentials_b64).decode("utf-8"))
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
            )
            client = storage.Client(credentials=credentials, project=info.get("project_id"))
        self.client = client
        self.bucket = client.bucket(settings.gcs_bucket)
        self.existing = set()
        self.lock = threading.Lock()

    def key_for(self, item):
        return object_key(self.settings.gcs_prefix, item.language, item.content_hash, item.fmt)

    def url_for_key(self, key):
        return f"{self.settings.cdn_base_url}/{key}"

    def load_existing(self):
        names = set()
        for blob in self.client.list_blobs(
            self.settings.gcs_bucket,
            prefix=f"{self.settings.gcs_prefix}/",
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

    def upload(self, key, data, content_type):
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
            except gexc.PreconditionFailed:
                self._remember(key)
                return self.url_for_key(key)
            except TRANSIENT:
                attempt += 1
                if attempt > 5:
                    raise
                delay = min(20.0, (2 ** attempt) * 0.5) * (0.5 + random.random())
                time.sleep(delay)