import asyncio
import collections
import contextlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import audio
from .control import ClientError, Stop, stop_event
from .prompt import fits_context, prepare, sort_by_length

log = logging.getLogger("pipeline")

CONTENT_REASONS = frozenset(
    {
        "empty",
        "non_finite",
        "silent",
        "too_short",
        "too_long",
        "pace_short",
        "pace_long",
        "too_few_tokens",
        "bad_codes",
        "truncated",
        "prompt_too_long",
    }
)

PRUNE_AT = 1024


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
        self.decoder_pool = ThreadPoolExecutor(max_workers=1)
        self.loop = None
        self.slots = None
        self.futures = []
        self.futures_lock = threading.Lock()
        self.records = []
        self.records_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.last_flush = time.time()
        self.flush_stop = threading.Event()
        self.finished_ids = set()
        self.finished_lock = threading.Lock()
        self.total = 0
        self.final_failed = 0
        self.state_lock = threading.RLock()
        self.last_progress = time.time()
        self.consecutive_failures = 0
        self.window = collections.deque(maxlen=max(10, settings.breaker_window))
        self.tripped = None
        self.last_reason = {}

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

    def _touch(self):
        with self.state_lock:
            self.last_progress = time.time()

    def _success(self):
        with self.state_lock:
            self.consecutive_failures = 0
            self.last_progress = time.time()
            self.window.append(True)

    def _trip(self, reason):
        with self.state_lock:
            if self.tripped is None:
                self.tripped = reason
        stop_event.set()

    def _check_window(self):
        window = self.window
        if len(window) < window.maxlen:
            return
        failures = sum(1 for ok in window if not ok)
        if failures / float(len(window)) > self.s.breaker_fail_ratio:
            self._trip("failure_rate")

    def _add_failure(self, item, reason, count=True):
        self.tel.fail(reason)
        with self.state_lock:
            self.last_reason[item.id] = reason
            if not count:
                return
            self.window.append(False)
            if reason not in CONTENT_REASONS:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.s.max_consecutive_failures:
                    self._trip("consecutive_failures")
            self._check_window()

    def _fail_final(self, item, reason, count=True):
        self.tel.fail(reason)
        self.tel.add("failed")
        self._record(item, "failed", error=reason)
        self._mark_finished(item.id)
        with self.state_lock:
            self.final_failed += 1
            self.last_reason[item.id] = reason
            if count:
                self.window.append(False)
                self._check_window()

    def _guards(self):
        with self.state_lock:
            idle = time.time() - self.last_progress
        if idle > self.s.idle_timeout_seconds:
            self._trip("idle_timeout")
        if self.tel.spend(self.s.gpu_hourly_rate) > self.s.max_spend_usd:
            self._trip("spend_cap")

    def _flush(self, force=False):
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

    def _release_slot(self):
        try:
            self.loop.call_soon_threadsafe(self.slots.release)
        except RuntimeError:
            pass

    def _encode_upload(self, item, wave):
        try:
            processed, reason = audio.process(wave, self.s, len(item.text))
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
            self._release_slot()

    async def _submit(self, item, wave):
        try:
            await asyncio.wait_for(self.slots.acquire(), timeout=self.s.slot_timeout_seconds)
        except asyncio.TimeoutError:
            return False
        try:
            future = self.pool.submit(self._encode_upload, item, wave)
        except Exception:
            self.slots.release()
            raise
        with self.futures_lock:
            if len(self.futures) >= PRUNE_AT:
                self.futures = [f for f in self.futures if not f.done()]
            self.futures.append(future)
        return True

    def _wait_encodes(self):
        with self.futures_lock:
            pending = list(self.futures)
            self.futures = []
        for future in pending:
            try:
                future.result()
            except Exception:
                pass

    async def _decode_and_submit(self, batch, by_id):
        entries = [(f.key, f.tokens) for f in batch]
        try:
            decoded, bad = await self.loop.run_in_executor(
                self.decoder_pool, self.codec.decode_batch, entries
            )
        except Exception as exc:
            for f in batch:
                self._add_failure(by_id[f.key], type(exc).__name__)
            return
        for key, reason in bad:
            self._add_failure(by_id[key], reason)
        for key, wave in decoded:
            self.tel.add("decoded")
            if not await self._submit(by_id[key], wave):
                self._add_failure(by_id[key], "slot_timeout")

    async def _decoder(self, queue, by_id):
        loop = self.loop
        micro = max(1, self.s.decode_microbatch)
        while True:
            first = await queue.get()
            if first is None:
                return
            batch = [first]
            closed = False
            deadline = loop.time() + self.s.decode_flush_seconds
            while len(batch) < micro and not closed:
                try:
                    nxt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    wait = deadline - loop.time()
                    if wait <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(queue.get(), wait)
                    except asyncio.TimeoutError:
                        break
                if nxt is None:
                    closed = True
                else:
                    batch.append(nxt)
            await self._decode_and_submit(batch, by_id)
            if closed:
                return

    async def _stream(self, prepared):
        if not prepared:
            return
        by_id = {p.item.id: p.item for p in prepared}
        queue = asyncio.Queue()
        decoder = asyncio.create_task(self._decoder(queue, by_id))
        try:
            async for finished in self.engine.generate(prepared, stop_event.is_set):
                self.tel.add("generated", finished.generations)
                self.tel.add("tokens", finished.spent_tokens)
                if finished.truncations:
                    self.tel.add("truncated", finished.truncations)
                if finished.generations > 1:
                    self.tel.add("retried", finished.generations - 1)
                self._touch()
                item = by_id[finished.key]
                if finished.error:
                    self._add_failure(item, finished.error)
                elif finished.finish_reason == "length":
                    self._fail_final(item, "truncated")
                else:
                    queue.put_nowait(finished)
        finally:
            queue.put_nowait(None)
            await asyncio.gather(decoder, return_exceptions=True)

    def _reject_oversize(self, prepared):
        keep = []
        for entry in prepared:
            if fits_context(entry, self.s):
                keep.append(entry)
                continue
            self._fail_final(entry.item, "prompt_too_long", count=False)
        return keep

    async def run(self, items):
        self.loop = asyncio.get_running_loop()
        self.slots = asyncio.Semaphore(max(1, self.s.queue_depth))
        self.total = len(items)
        current = []
        flusher = threading.Thread(target=self._flush_loop, daemon=True)
        flusher.start()
        monitor = asyncio.create_task(self._monitor())
        try:
            current = self._plan(items)
            attempt = 0
            while current and attempt <= self.s.max_retries:
                if stop_event.is_set():
                    break
                self._touch()
                if attempt > 0:
                    self.tel.add("retried", len(current))
                prepared = prepare(current, self.engine.tokenizer, self.s, attempt=attempt)
                prepared = sort_by_length(self._reject_oversize(prepared))
                try:
                    await self._stream(prepared)
                finally:
                    await self.loop.run_in_executor(None, self._wait_encodes)
                with self.finished_lock:
                    done = set(self.finished_ids)
                current = [item for item in current if item.id not in done]
                attempt += 1
        finally:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.decoder_pool.shutdown(wait=False)
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
        return len(leftover) + self.final_failed, self.tripped