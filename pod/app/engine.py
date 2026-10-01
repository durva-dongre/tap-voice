import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import List, Optional

from .logits import sampling_restrictions
from .prompt import EOS, Prepared, attempt_cap, bos_id, build_ids, seed_from_hash

log = logging.getLogger("engine")


class FatalEngineError(Exception):
    pass


@dataclass
class Finished:
    key: str
    tokens: List[int] = field(default_factory=list)
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    generations: int = 1
    truncations: int = 0
    spent_tokens: int = 0


@dataclass(frozen=True)
class WarmupItem:
    id: str


def resolve_quantization(model_dir, override):
    if override:
        return None if override == "auto" else override
    try:
        with open(os.path.join(model_dir, "config.json")) as handle:
            config = json.load(handle)
    except Exception:
        return "fp8"
    return None if config.get("quantization_config") else "fp8"


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.llm = None
        self.tokenizer = None
        self.kv_dtype = None
        self.extras = {}
        self._params_cls = None

    def _gpu_total_mb(self):
        try:
            import torch

            return int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except Exception:
            return self.settings.gpu_total_mb

    def utilization(self):
        s = self.settings
        reserve = s.codec_vram_reserve_mb / float(self._gpu_total_mb())
        return max(0.5, min(0.95, s.gpu_memory_utilization - reserve))

    def resolve_kv_dtype(self):
        requested = (self.settings.kv_cache_dtype or "auto").strip().lower()
        if not requested.startswith("fp8"):
            return requested
        try:
            import torch

            capability = torch.cuda.get_device_capability(0)
        except Exception:
            capability = (0, 0)
        if capability < (8, 9):
            log.warning("kv_cache_dtype %s unsupported on arch %s, using auto", requested, capability)
            return "auto"
        return requested

    def _probe_extras(self):
        from vllm import SamplingParams

        s = self.settings
        extras = {}
        if s.skip_detokenize:
            try:
                SamplingParams(max_tokens=1, detokenize=False)
                extras["detokenize"] = False
            except (TypeError, ValueError):
                log.warning("detokenize=False not supported by this vLLM")
        if s.final_only_outputs:
            try:
                from vllm.sampling_params import RequestOutputKind

                SamplingParams(max_tokens=1, output_kind=RequestOutputKind.FINAL_ONLY)
                extras["output_kind"] = RequestOutputKind.FINAL_ONLY
            except (ImportError, TypeError, ValueError):
                log.warning("output_kind=FINAL_ONLY not supported by this vLLM")
        return extras

    async def start(self):
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

        s = self.settings
        self._params_cls = SamplingParams
        self.kv_dtype = self.resolve_kv_dtype()
        steps = max(1, s.num_scheduler_steps)
        kwargs = dict(
            model=s.model_dir,
            tokenizer=s.model_dir,
            dtype="bfloat16",
            quantization=resolve_quantization(s.model_dir, s.quantization),
            kv_cache_dtype=self.kv_dtype,
            max_model_len=s.max_model_len,
            max_num_seqs=s.max_num_seqs,
            gpu_memory_utilization=self.utilization(),
            enable_prefix_caching=s.enable_prefix_caching,
            enable_chunked_prefill=s.enable_chunked_prefill and steps == 1,
            swap_space=0,
            enforce_eager=False,
            disable_log_stats=True,
            disable_log_requests=True,
        )
        if steps > 1:
            kwargs["num_scheduler_steps"] = steps
        if s.max_num_batched_tokens > 0:
            kwargs["max_num_batched_tokens"] = s.max_num_batched_tokens
        self.llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
        self.tokenizer = await self.llm.get_tokenizer()
        self.extras = self._probe_extras()
        log.info(
            "engine_ready kv=%s util=%.3f steps=%d extras=%s",
            self.kv_dtype,
            kwargs["gpu_memory_utilization"],
            steps,
            sorted(self.extras),
        )
        return self.tokenizer

    def sampling(self, prepared):
        s = self.settings
        attempt = getattr(prepared, "attempt", 0)
        return self._params_cls(
            temperature=max(0.3, s.temperature - 0.08 * attempt),
            top_p=s.top_p,
            top_k=s.top_k,
            repetition_penalty=s.repetition_penalty + 0.03 * attempt,
            max_tokens=prepared.max_tokens,
            stop_token_ids=[EOS],
            seed=prepared.seed if s.use_seeds else None,
            **self.extras,
            **sampling_restrictions(s.logits_mask, allow_header=s.allow_header_tokens),
        )

    def is_dead(self):
        errored = getattr(self.llm, "errored", None)
        if errored is None:
            return False
        return bool(errored() if callable(errored) else errored)

    async def _once(self, prepared):
        key = prepared.item.id
        request_id = f"{key}-{uuid.uuid4().hex[:8]}"
        last = None
        try:
            async for output in self.llm.generate(
                {"prompt_token_ids": prepared.prompt_ids},
                self.sampling(prepared),
                request_id,
            ):
                last = output
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.is_dead():
                raise FatalEngineError("engine_dead") from exc
            return Finished(key=key, error=type(exc).__name__)
        if last is None or not last.outputs:
            return Finished(key=key, error="no_output")
        out = last.outputs[0]
        tokens = list(out.token_ids)
        return Finished(
            key=key, tokens=tokens, finish_reason=out.finish_reason, spent_tokens=len(tokens)
        )

    def _retryable(self, prepared):
        if getattr(prepared.item, "content_hash", None) is None:
            return False
        return prepared.attempt < self.settings.max_retries

    def _next_attempt(self, prepared):
        attempt = prepared.attempt + 1
        item = prepared.item
        return replace(
            prepared,
            max_tokens=attempt_cap(item.text, prepared.room, self.settings, attempt),
            seed=seed_from_hash(item.content_hash, self.settings.base_seed, attempt),
            attempt=attempt,
        )

    async def run_one(self, prepared, stop_flag=None):
        current = prepared
        generations = 0
        truncations = 0
        spent = 0
        while True:
            result = await self._once(current)
            generations += 1
            spent += len(result.tokens)
            cut = result.error is None and result.finish_reason == "length"
            if cut:
                truncations += 1
            stopping = stop_flag is not None and stop_flag()
            if not cut or stopping or not self._retryable(current):
                result.generations = generations
                result.truncations = truncations
                result.spent_tokens = spent
                return result
            current = self._next_attempt(current)

    async def generate(self, prepared_list, stop_flag):
        if self.llm is None:
            raise FatalEngineError("engine_not_started")
        prepared_list = list(prepared_list)
        if not prepared_list:
            return
        workers_count = max(1, min(self.settings.max_num_seqs * 2, len(prepared_list)))
        grace = getattr(self.settings, "stop_grace_seconds", 20.0)
        source = iter(prepared_list)
        queue = asyncio.Queue()

        async def worker():
            try:
                while not stop_flag():
                    try:
                        prepared = next(source)
                    except StopIteration:
                        return
                    queue.put_nowait(await self.run_one(prepared, stop_flag))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                queue.put_nowait(exc)
            finally:
                queue.put_nowait(None)

        workers = [asyncio.create_task(worker()) for _ in range(workers_count)]
        alive = workers_count
        stopped_at = None
        try:
            while alive:
                if stopped_at is None and stop_flag():
                    stopped_at = time.time()
                if stopped_at is not None and time.time() - stopped_at > grace:
                    return
                try:
                    got = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if self.is_dead():
                        raise FatalEngineError("engine_dead")
                    continue
                if got is None:
                    alive -= 1
                    continue
                if isinstance(got, BaseException):
                    raise got
                yield got
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def warmup(self, voices=None):
        chosen = list(voices or [])[: max(1, self.settings.warmup_voices)] or ["English (Female)"]
        bos = bos_id(self.tokenizer)
        prepared = []
        for index, voice in enumerate(chosen):
            ids = build_ids(self.tokenizer, voice, "Warmup.", None, bos)
            prepared.append(
                Prepared(item=WarmupItem(id=f"warmup-{index}"), prompt_ids=ids, max_tokens=48, seed=1)
            )
        async for _ in self.generate(prepared, lambda: False):
            pass

    async def shutdown(self):
        if self.llm is not None:
            try:
                self.llm.shutdown_background_loop()
            except Exception:
                pass