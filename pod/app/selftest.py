import base64, time
import numpy as np
from . import config as C
from .engines import Kokoro, Indic
from .audio import to_ogg

def vram_report(torch, label):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    print(f"{label} vram allocated={alloc/1e9:.2f}GB peak={peak/1e9:.2f}GB")

for name, E, job in (
    ("kokoro", Kokoro, {"text": "Hello from the self test.", "language": "en"}),
    ("indic", Indic, {"text": "नमस्ते, यह एक परीक्षण है।", "language": "hi"}),
):
    t = time.time()
    e = E()
    t1 = time.time()
    a = e(job)
    t2 = time.time()
    print(name, f"load {t1-t:.1f}s gen {t2-t1:.1f}s dtype {a.dtype} shape {a.shape} sr {e.sr} "
                f"dur {len(a)/e.sr:.1f}s peak {np.abs(a).max():.2f} ogg_bytes {len(to_ogg(a, e.sr))}")
    if not C.MOCK and hasattr(e, "torch"):
        vram_report(e.torch, name)

if not C.MOCK:
    import torch as _torch

    ab_job = {"text": "नमस्ते, यह एक परीक्षण है।", "language": "hi"}
    punjabi_job = {"text": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਇੱਕ ਟੈਸਟ ਹੈ।", "language": "pa"}
    long_hindi_text = ("नमस्ते, यह एक लंबा परीक्षण वाक्य है जो यह जांचने के लिए बनाया गया है कि "
                        "मॉडल लंबे इनपुट पाठ पर कैसा प्रदर्शन करता है। " * 12)[:990]
    long_hindi_job = {"text": long_hindi_text, "language": "hi"}

    print("indic extra checks: building second engine instance for A/B and language/length probes")

    dtypes_to_test = ["float16", "float32"]
    ab_results = {}
    for dtype in dtypes_to_test:
        original = C.INDIC_DTYPE
        C.INDIC_DTYPE = dtype
        try:
            eng = Indic()
            t0 = time.time()
            audio = eng(ab_job)
            elapsed = time.time() - t0
            ogg = to_ogg(audio, eng.sr)
            ab_results[dtype] = (elapsed, audio.dtype, len(ogg))
            print(f"indic_ab dtype={dtype} gen={elapsed:.1f}s array_dtype={audio.dtype} ogg_bytes={len(ogg)}")
            with open(f"/tmp/indic_ab_{dtype}.ogg", "wb") as f:
                f.write(ogg)
            print(f"indic_ab dtype={dtype} written to /tmp/indic_ab_{dtype}.ogg ({len(ogg)} bytes)")
            print(f"indic_ab dtype={dtype} ogg_base64={base64.b64encode(ogg).decode()}")
            vram_report(_torch, f"indic_ab_{dtype}")
            del eng
            import gc
            gc.collect()
            _torch.cuda.empty_cache()
        finally:
            C.INDIC_DTYPE = original

    punjabi_eng = Indic()
    t0 = time.time()
    punjabi_audio = punjabi_eng(punjabi_job)
    punjabi_elapsed = time.time() - t0
    punjabi_ogg = to_ogg(punjabi_audio, punjabi_eng.sr)
    with open("/tmp/indic_punjabi.ogg", "wb") as f:
        f.write(punjabi_ogg)
    print(f"indic_punjabi gen={punjabi_elapsed:.1f}s dur={len(punjabi_audio)/punjabi_eng.sr:.1f}s "
          f"ogg_bytes={len(punjabi_ogg)} written to /tmp/indic_punjabi.ogg")
    vram_report(_torch, "indic_punjabi")

    t0 = time.time()
    long_audio = punjabi_eng(long_hindi_job)
    long_elapsed = time.time() - t0
    long_ogg = to_ogg(long_audio, punjabi_eng.sr)
    with open("/tmp/indic_long_hindi.ogg", "wb") as f:
        f.write(long_ogg)
    print(f"indic_long_hindi chars={len(long_hindi_text)} gen={long_elapsed:.1f}s "
          f"dur={len(long_audio)/punjabi_eng.sr:.1f}s ogg_bytes={len(long_ogg)} written to /tmp/indic_long_hindi.ogg")
    vram_report(_torch, "indic_long_hindi")

    del punjabi_eng
    import gc
    gc.collect()
    _torch.cuda.empty_cache()