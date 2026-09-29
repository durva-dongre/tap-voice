import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from .logits import sampling_restrictions
from .prompt import EOS, Prepared, bos_id, build_ids

log = logging.getLogger("engine")


class FatalEngineError(Exception):
    pass


@dataclass
class Finished:
    key: str
    tokens: List[int] = field(default_factory=list)
    error: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class WarmupItem:
    id: str


def resolve_quantization(model_dir, override):
    """Choose the value for vLLM's quantization argument.

    llm-compressor checkpoints declare their own method in config.json, and vLLM 0.6.x
    raises at startup if the argument disagrees with it. So pass nothing when the
    checkpoint already declares one, and "fp8" only when it does not.
    """
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

    # ---- configuration -----------------------------------------------------------

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
        """FP8 KV cache needs CUDA arch 8.9+ (Ada/Hopper). On Ampere it crashes at the first
        request inside a Triton kernel, after the model has already been paid for."""
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
        """Optional CPU savers. Each is used only if this vLLM build accepts it."""
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
            # Multi-step scheduling and chunked prefill do not combine in vLLM 0.6.x.
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

    # ---- generation --------------------------------------------------------------

    def sampling(self, prepared):
        s = self.settings
        return self._params_cls(
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            repetition_penalty=s.repetition_penalty,
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

    async def run_one(self, prepared):
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
        return Finished(key=key, tokens=list(out.token_ids), finish_reason=out.finish_reason)

    async def generate(self, prepared_list, stop_flag):
        """Yield Finished results. After stop_flag() turns true, no new requests are started
        and in-flight ones get stop_grace_seconds to finish before being cancelled."""
        if self.llm is None:
            raise FatalEngineError("engine_not_started")
        limit = max(1, self.settings.max_num_seqs * 2)
        grace = getattr(self.settings, "stop_grace_seconds", 20.0)
        iterator = iter(prepared_list)
        in_flight = set()
        exhausted = False
        stopped_at = None
        try:
            while True:
                stopping = stop_flag()
                if stopping and stopped_at is None:
                    stopped_at = time.time()
                while not exhausted and not stopping and len(in_flight) < limit:
                    try:
                        prepared = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    in_flight.add(asyncio.create_task(self.run_one(prepared)))
                if not in_flight:
                    if exhausted or stopping:
                        return
                    continue
                if stopped_at is not None and time.time() - stopped_at > grace:
                    return
                done, in_flight = await asyncio.wait(
                    in_flight, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                )
                if not done and self.is_dead():
                    raise FatalEngineError("engine_dead")
                for task in done:
                    yield task.result()
        finally:
            for task in in_flight:
                task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)

    async def warmup(self, voices=None):
        """One tiny request is enough to trigger kernel setup; more only add billed seconds."""
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