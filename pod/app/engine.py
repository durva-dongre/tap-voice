import asyncio
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from .logits import sampling_restrictions
from .prompt import EOS, Prepared, bos_id, build_ids


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


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.llm = None
        self.tokenizer = None

    def utilization(self):
        s = self.settings
        reserve = s.codec_vram_reserve_mb / float(s.gpu_total_mb)
        return max(0.5, min(0.95, s.gpu_memory_utilization - reserve))

    async def start(self):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        s = self.settings
        args = AsyncEngineArgs(
            model=s.model_dir,
            tokenizer=s.model_dir,
            dtype="bfloat16",
            quantization="fp8",
            kv_cache_dtype=s.kv_cache_dtype,
            max_model_len=s.max_model_len,
            max_num_seqs=s.max_num_seqs,
            gpu_memory_utilization=self.utilization(),
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            swap_space=0,
            enforce_eager=False,
            disable_log_stats=True,
        )
        self.llm = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = await self.llm.get_tokenizer()
        return self.tokenizer

    def sampling(self, prepared):
        from vllm import SamplingParams

        s = self.settings
        return SamplingParams(
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            repetition_penalty=s.repetition_penalty,
            max_tokens=prepared.max_tokens,
            stop_token_ids=[EOS],
            seed=prepared.seed,
            **sampling_restrictions(s.logits_mask),
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
        if self.llm is None:
            raise FatalEngineError("engine_not_started")
        limit = max(1, self.settings.max_num_seqs * 2)
        iterator = iter(prepared_list)
        in_flight = set()
        exhausted = False
        try:
            while True:
                while not exhausted and not stop_flag() and len(in_flight) < limit:
                    try:
                        prepared = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    in_flight.add(asyncio.create_task(self.run_one(prepared)))
                if not in_flight:
                    if exhausted or stop_flag():
                        return
                    continue
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

    async def warmup(self, voices):
        bos = bos_id(self.tokenizer)
        prepared = []
        for index, voice in enumerate(voices):
            ids = build_ids(self.tokenizer, voice, "Warmup.", None, bos)
            prepared.append(
                Prepared(item=WarmupItem(id=f"warmup-{index}"), prompt_ids=ids, max_tokens=64, seed=1)
            )
        async for _ in self.generate(prepared, lambda: False):
            pass

    async def shutdown(self):
        if self.llm is not None:
            try:
                self.llm.shutdown_background_loop()
            except Exception:
                pass