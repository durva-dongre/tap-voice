import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import audio
from .control import ClientError, Stop, stop_event
from .prompt import prepare, sort_by_length


class Breaker(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Pipeline:
    def __init__(self, settings, engine, codec, store, control, telemetry, heartbeat):
        self.s = settings
        self.engine = engine
        self.codec = codec
        self.store = store
        self.control = control
        self.tel = telemetry
        self.hb = heartbeat
        self.pool = ThreadPoolExecutor(max_workers=settings.upload_threads)
        self.slots = threading.BoundedSemaphore(settings.queue_depth)
        self.records = []
        self.records_lock = threading.Lock()
        self.last_flush = time.time()
        self.finished_ids = set()
        self.finished_lock = threading.Lock()
        self.total = 0
        self.last_progress = time.time()
        self.consecutive_failures = 0
        self.tripped = None
        self.trip_lock = threading.Lock()

    def remaining(self):
        with self.finished_lock:
            return max(0, self.total - len(self.finished_ids))

    def _mark_finished(self, item_id):
        with self.finished_lock:
            self.finished_ids.add(item_id)

    def _record(self, item, status, url=None, error=None):
        record = {
            "id": item.id,
            "status": status,
            "url": url,
            "error": error,
            "content_hash": item.content_hash,
        }
        with self.records_lock:
            self.records.append(record)

    def _flush(self, force=False):
        with self.records_lock:
            if not self.records:
                return
            due = (
                len(self.records) >= self.s.progress_every_items
                or time.time() - self.last_flush >= self.s.progress_every_seconds
            )
            if not (due or force):
                return
            batch = self.records
            self.records = []
            self.last_flush = time.time()
        try:
            self.control.progress(batch)
        except Stop:
            stop_event.set()
            with self.records_lock:
                self.records = batch + self.records
        except ClientError:
            with self.records_lock:
                self.records = batch + self.records
        except Exception:
            with self.records_lock:
                self.records = batch + self.records

    def _success(self):
        self.consecutive_failures = 0
        self.last_progress = time.time()

    def _failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.s.max_consecutive_failures:
            self._trip("consecutive_failures")

    def _trip(self, reason):
        with self.trip_lock:
            if self.tripped is None:
                self.tripped = reason
        stop_event.set()

    def _guards(self):
        if time.time() - self.last_progress > self.s.idle_timeout_seconds:
            self._trip("idle_timeout")
        if self.tel.spend(self.s.gpu_hourly_rate) > self.s.max_spend_usd:
            self._trip("spend_cap")

    def _encode_upload(self, item, wave):
        try:
            processed, reason = audio.process(wave, self.s)
            if processed is None:
                return ("retry", item, reason)
            data, content_type, _ = audio.encode(processed, item.fmt, self.s)
            key = self.store.key_for(item)
            url = self.store.upload(key, data, content_type)
            self.tel.add("uploaded")
            self.tel.add("audio_seconds", processed.size / self.s.source_sample_rate)
            return ("ok", item, url)
        except Stop:
            stop_event.set()
            return ("retry", item, "stopped")
        except Exception as exc:
            return ("retry", item, type(exc).__name__)
        finally:
            self.slots.release()

    def _acquire_slot(self):
        return self.slots.acquire(timeout=self.s.slot_timeout_seconds)

    def _plan(self, items):
        pending = []
        for item in items:
            key = self.store.key_for(item)
            if self.store.exists(key):
                self.tel.add("skipped_existing")
                self._record(item, "done", url=self.store.url_for_key(key))
                self._mark_finished(item.id)
            else:
                pending.append(item)
        self.tel.add("planned", len(pending))
        return pending

    def _settle(self, futures, failures):
        for future in futures:
            try:
                status, item, detail = future.result()
            except Exception as exc:
                continue
            if status == "ok":
                self._record(item, "done", url=detail)
                self._mark_finished(item.id)
                self._success()
            else:
                failures.append((item, detail))
                self._failure()
            self._flush()

    async def _submit(self, loop, futures, item, wave):
        acquired = await loop.run_in_executor(None, self._acquire_slot)
        if not acquired:
            return False
        try:
            futures.append(self.pool.submit(self._encode_upload, item, wave))
        except Exception:
            self.slots.release()
            raise
        return True

    async def _generate_decode(self, prepared, futures, failures):
        loop = asyncio.get_running_loop()
        by_id = {p.item.id: p.item for p in prepared}
        buffer = []
        last_drain = time.time()

        async def drain():
            nonlocal buffer, last_drain
            if not buffer:
                return
            entries = [(f.key, f.tokens) for f in buffer]
            buffer = []
            last_drain = time.time()
            decoded, bad = await loop.run_in_executor(None, self.codec.decode_batch, entries)
            for key, reason in bad:
                failures.append((by_id[key], reason))
                self._failure()
            for key, wave in decoded:
                self.tel.add("decoded")
                submitted = await self._submit(loop, futures, by_id[key], wave)
                if not submitted:
                    failures.append((by_id[key], "slot_timeout"))
                    self._failure()

        async for finished in self.engine.generate(prepared, stop_event.is_set):
            self.tel.add("generated")
            if finished.error:
                failures.append((by_id[finished.key], finished.error))
                self._failure()
            else:
                self.tel.add("tokens", len(finished.tokens))
                buffer.append(finished)
            if (
                len(buffer) >= self.s.decode_microbatch
                or time.time() - last_drain >= self.s.decode_flush_seconds
            ):
                await drain()
            self.hb.set("generating", self.remaining())
            self._guards()
            self._flush()
        await drain()

    def _requeue_unfinished(self, current, failures):
        accounted = {item.id for item, _ in failures}
        with self.finished_lock:
            done = set(self.finished_ids)
        return [
            item
            for item in current
            if item.id not in done and item.id not in accounted
        ]

    async def run(self, items):
        loop = asyncio.get_running_loop()
        self.total = len(items)
        current = []
        try:
            pending = self._plan(items)
            self._flush(force=True)
            current = pending
            attempt = 0
            while current and attempt <= self.s.max_retries:
                if stop_event.is_set():
                    break
                prepared = sort_by_length(
                    prepare(current, self.engine.tokenizer, self.s, attempt=attempt)
                )
                futures = []
                failures = []
                try:
                    await self._generate_decode(prepared, futures, failures)
                finally:
                    await loop.run_in_executor(None, self._settle, futures, failures)
                if attempt > 0:
                    self.tel.add("retried", len(current))
                unfinished = self._requeue_unfinished(current, failures)
                for _, reason in failures:
                    self.tel.fail(reason)
                current = [item for item, _ in failures] + unfinished
                with self.finished_lock:
                    done = set(self.finished_ids)
                current = [item for item in current if item.id not in done]
                attempt += 1
        finally:
            with self.finished_lock:
                done = set(self.finished_ids)
            leftover = [item for item in current if item.id not in done]
            reason = "exhausted_retries"
            if stop_event.is_set():
                reason = self.tripped or "stopped"
            for item in leftover:
                self._record(item, "failed", error=reason)
            self._flush(force=True)
            self.pool.shutdown(wait=True)
        return len(leftover), self.tripped