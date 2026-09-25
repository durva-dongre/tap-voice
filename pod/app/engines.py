import logging, re
import numpy as np
from . import config as C

log = logging.getLogger("engines")
SR = 24000
VOICE_RE = re.compile(r"^[a-z0-9_]{1,40}$")

def _tone(sr=SR):
    return (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr, False))).astype(np.float32)

class Kokoro:
    sr = SR

    def __init__(s):
        if not C.MOCK:
            from kokoro import KPipeline
            s.p = KPipeline(lang_code="a")

    def __call__(s, job):
        if C.MOCK:
            return _tone(s.sr)
        voice = job.get("voice_id") if VOICE_RE.match(job.get("voice_id") or "") else C.KOKORO_VOICE

        def run(v):
            return [np.asarray(a.detach().cpu() if hasattr(a, "detach") else a, dtype=np.float32)
                    for _, _, a in s.p(job["text"], voice=v)]

        try:
            parts = run(voice)
        except Exception:
            if voice == C.KOKORO_VOICE:
                raise
            parts = run(C.KOKORO_VOICE)
        return np.concatenate(parts)

def _resolve_dtype(torch):
    if C.INDIC_DTYPE == "float16":
        return torch.float16
    if C.INDIC_DTYPE == "float32":
        return torch.float32
    log.warning("unrecognized INDIC_DTYPE=%r, falling back to float32", C.INDIC_DTYPE)
    return torch.float32

class Indic:
    def __init__(s):
        s.sr = SR
        if C.MOCK:
            return
        import torch
        from transformers import AutoTokenizer
        from parler_tts import ParlerTTSForConditionalGeneration

        s.torch = torch
        s.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _resolve_dtype(torch)
        with torch.inference_mode():
            s.model = ParlerTTSForConditionalGeneration.from_pretrained(C.INDIC_REPO, torch_dtype=dtype).to(s.device)
            s.model.eval()
            s.prompt_tok = AutoTokenizer.from_pretrained(C.INDIC_REPO)
            s.desc_tok = AutoTokenizer.from_pretrained(s.model.config.text_encoder._name_or_path)
            s.sr = s.model.config.sampling_rate
            desc = s.desc_tok(C.INDIC_SPEAKER, return_tensors="pt").to(s.device)
            s.desc_ids, s.desc_mask = desc.input_ids, desc.attention_mask

        try:
            enc_dtype = next(s.model.text_encoder.parameters()).dtype
            log.info("indic text_encoder dtype=%s requested=%s", enc_dtype, dtype)
        except Exception as e:
            log.warning("could not inspect text_encoder dtype: %s", e)

        if s.device == "cuda":
            try:
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                log.info("indic load vram allocated=%.2fGB total=%.2fGB headroom=%.2fGB",
                          allocated / 1e9, total / 1e9, (total - allocated) / 1e9)
            except Exception as e:
                log.warning("vram headroom check failed: %s", e)

    def __call__(s, job):
        if C.MOCK:
            return _tone(s.sr)
        prompt = s.prompt_tok(job["text"], return_tensors="pt").to(s.device)
        with s.torch.inference_mode():
            out = s.model.generate(
                input_ids=s.desc_ids,
                attention_mask=s.desc_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
            )
        return out.to(s.torch.float32).cpu().numpy().squeeze()