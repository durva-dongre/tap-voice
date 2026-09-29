import asyncio
import contextlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import audio
from .control import ClientError, Stop, stop_event
from .prompt import fits_context, prepare, sort_by_length

log = logging.getLogger("pipeline")


class Pipeline:
    """Generate -> decode -> encode -> upload, with results reported clip by clip.

    Bookkeeping happens in the upload thread the moment a clip is safely stored, so
    progress, the idle watchdog and the heartbeat all move while the batch is running.
    A background thread sends the records to the server in small batches.
    """

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
        self.send_lock = threading.Lock()
        self.last_flush = time.time()
        self.flush_stop = threading.Event()
        self.finished_ids = set()
        self.finished_lock = threading.Lock()
        self.total = 0
        self.state_lock = threading.RLock()
        self.last_progress = time.time()
        self.consecutive_failures = 0
        self.tripped = None
        self.attempt_failures = []
        self.last_reason = {}

    # ---- bookkeeping -------------------------------------------------------------

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

    def _success(self):
        with self.state_lock:
            self.consecutive_failures = 0
            self.last_progress = time.time()

    def _trip(self, reason):
        with self.state_lock:
            if self.tripped is None:
                self.tripped = reason
        stop_event.set()

    def _add_failure(self, item, reason, count=True):
        self.tel.fail(reason)
        with self.state_lock:
            self.attempt_failures.append((item, reason))
            self.last_reason[item.id] = reason
            if count:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.s.max_consecutive_failures:
                    self._trip("consecutive_failures")

    def _guards(self):
        with self.state_lock:
            idle = time.time() - self.last_progress
        if idle > self.s.idle_timeout_seconds:
            self._trip("idle_timeout")
        if self.tel.spend(self.s.gpu_hourly_rate) > self.s.max_spend_usd:
            self._trip("spend_cap")

    # ---- progress reporting ------------------------------------------------------

    def _flush(self, force=False):
        """Send pending records. Returns True when nothing is left pending."""
        with self.send_lock:
            with self.records_lock:
                if not self.records:
                    return True
                due = (
                    len(self.records) >= self.s.progress_every_items
                    or time.time() - self.last_flush >= self.s.progress_every_seconds
                )
                if not (due or force):
                    return True
                batch = self.records
                self.records = []
                self.last_flush = time.time()
            try:
                self.control.progress(batch)
                return True
            except Stop:
                stop_event.set()
            except ClientError as exc:
                log.warning("progress_rejected %s", exc)
            except Exception as exc:
                log.warning("progress_failed %s", type(exc).__name__)
            with self.records_lock:
                self.records = batch + self.records
            return False

    def _flush_loop(self):
        tick = max(0.05, min(1.0, self.s.progress_every_seconds / 2.0))
        while not self.flush_stop.wait(tick):
            try:
                self._flush()
            except Exception as exc:
                log.warning("flush_loop_error %s", type(exc).__name__)

    async def _monitor(self):
        while True:
            await asyncio.sleep(2.0)
            self.hb.set("generating", self.remaining())
            self._guards()

    # ---- work --------------------------------------------------------------------

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

    def _encode_upload(self, item, wave):
        try:
            processed, reason = audio.process(wave, self.s)
            if processed is None:
                self._add_failure(item, reason)
                return
            data, content_type, _ = audio.encode(processed, item.fmt, self.s)
            key = self.store.key_for(item)
            url = self.store.upload(key, data, content_type)
            self.tel.add("uploaded")
            self.tel.add("audio_seconds", processed.size / self.s.source_sample_rate)
            self._record(item, "done", url=url)
            self._mark_finished(item.id)
            self._success()
        except Stop:
            stop_event.set()
            self._add_failure(item, "stopped", count=False)
        except Exception as exc:
            self._add_failure(item, type(exc).__name__)
        finally:
            self.slots.release()

    async def _submit(self, loop, futures, item, wave):
        if not self.slots.acquire(blocking=False):
            acquired = await loop.run_in_executor(
                None, lambda: self.slots.acquire(timeout=self.s.slot_timeout_seconds)
            )
            if not acquired:
                return False
        try:
            futures.append(self.pool.submit(self._encode_upload, item, wave))
        except Exception:
            self.slots.release()
            raise
        return True

    @staticmethod
    def _wait(futures):
        for future in futures:
            try:
                future.result()
            except Exception:
                pass

    async def _generate_decode(self, prepared, futures):
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
                self._add_failure(by_id[key], reason)
            for key, wave in decoded:
                self.tel.add("decoded")
                if not await self._submit(loop, futures, by_id[key], wave):
                    self._add_failure(by_id[key], "slot_timeout")

        async for finished in self.engine.generate(prepared, stop_event.is_set):
            self.tel.add("generated")
            if finished.error:
                self._add_failure(by_id[finished.key], finished.error)
            elif finished.finish_reason == "length":
                # Ran out of tokens before the model said it was done: the audio is cut off.
                self.tel.add("truncated")
                self._add_failure(by_id[finished.key], "truncated")
            else:
                self.tel.add("tokens", len(finished.tokens))
                buffer.append(finished)
            if (
                len(buffer) >= self.s.decode_microbatch
                or time.time() - last_drain >= self.s.decode_flush_seconds
            ):
                await drain()
        await drain()

    def _reject_oversize(self, prepared):
        """Items whose prompt leaves too little context for the audio: fail them for good."""
        keep = []
        for entry in prepared:
            if fits_context(entry, self.s):
                keep.append(entry)
                continue
            self.tel.fail("prompt_too_long")
            self.tel.add("failed")
            self._record(entry.item, "failed", error="prompt_too_long")
            self._mark_finished(entry.item.id)
        return keep

    async def run(self, items):
        loop = asyncio.get_running_loop()
        self.total = len(items)
        current = []
        leftover = []
        flusher = threading.Thread(target=self._flush_loop, daemon=True)
        flusher.start()
        monitor = asyncio.create_task(self._monitor())
        try:
            current = self._plan(items)
            attempt = 0
            while current and attempt <= self.s.max_retries:
                if stop_event.is_set():
                    break
                with self.state_lock:
                    self.last_progress = time.time()
                    self.attempt_failures = []
                if attempt > 0:
                    self.tel.add("retried", len(current))
                prepared = prepare(current, self.engine.tokenizer, self.s, attempt=attempt)
                prepared = sort_by_length(self._reject_oversize(prepared))
                futures = []
                try:
                    await self._generate_decode(prepared, futures)
                finally:
                    await loop.run_in_executor(None, self._wait, futures)
                with self.finished_lock:
                    done = set(self.finished_ids)
                current = [item for item in current if item.id not in done]
                attempt += 1
        finally:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.flush_stop.set()
            with self.finished_lock:
                done = set(self.finished_ids)
            leftover = [item for item in current if item.id not in done]
            for item in leftover:
                if stop_event.is_set():
                    error = self.tripped or "stopped"
                else:
                    error = self.last_reason.get(item.id) or "exhausted_retries"
                self._record(item, "failed", error=error)
            self.tel.add("failed", len(leftover))
            for _ in range(2):
                if self._flush(force=True) or stop_event.is_set():
                    break
        return len(leftover), self.tripped