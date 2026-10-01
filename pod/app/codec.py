import logging

import numpy as np

from .prompt import AUDIO_END, AUDIO_OFFSET

log = logging.getLogger("codec")

FRAME = 7
BAND = 4096
HOP = 2048


def clean_tokens(tokens):
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.size == 0:
        return arr
    audio = arr[(arr >= AUDIO_OFFSET) & (arr < AUDIO_END)]
    usable = (audio.size // FRAME) * FRAME
    return audio[:usable]


def tokens_to_layers(tokens, tolerance=0.0):
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.size == 0 or arr.size % FRAME:
        return None
    slots = np.arange(arr.size) % FRAME
    codes = arr - AUDIO_OFFSET - slots * BAND
    bad = (codes < 0) | (codes >= BAND)
    if bad.any():
        if bad.mean() > tolerance:
            return None
        codes[bad] = codes[bad] % BAND
    frames = codes.reshape(-1, FRAME)
    layer0 = frames[:, 0]
    layer1 = frames[:, [1, 4]].reshape(-1)
    layer2 = frames[:, [2, 3, 5, 6]].reshape(-1)
    return layer0, layer1, layer2


def valid_codes(layers):
    for layer in layers:
        if layer.size == 0 or layer.min() < 0 or layer.max() >= BAND:
            return False
    return True


def expected_samples(frames):
    return frames * HOP


def pad_layer(dst, src, per_frame):
    dst[: src.size] = src
    missing = dst.size - src.size
    if missing <= 0:
        return
    if src.size >= per_frame:
        dst[src.size :] = np.tile(src[-per_frame:], missing // per_frame)
        return
    tail = src[-per_frame:] if src.size >= per_frame else src
    reps = (missing + tail.size - 1) // tail.size
    dst[src.size :] = np.tile(tail, reps)[:missing]


def is_oom(exc):
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


class Codec:
    def __init__(self, snac_dir, device="cuda", half=True, tolerance=0.0):
        import torch
        from snac import SNAC

        self.torch = torch
        self.device = device
        self.tolerance = tolerance
        model = SNAC.from_pretrained(snac_dir).to(device).eval()
        self.model = model.half() if half else model
        self.stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None

    def warmup(self):
        frames = 12
        tokens = AUDIO_OFFSET + (np.arange(frames * FRAME) % FRAME) * BAND
        layers = tokens_to_layers(tokens)
        self._decode_group([("warmup", tokens, layers)])

    def _release(self):
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def _decode_group(self, group):
        torch = self.torch
        n = len(group)
        frame_counts = [tokens.size // FRAME for _, tokens, _ in group]
        max_frames = max(frame_counts)
        host0 = np.zeros((n, max_frames), dtype=np.int64)
        host1 = np.zeros((n, max_frames * 2), dtype=np.int64)
        host2 = np.zeros((n, max_frames * 4), dtype=np.int64)
        for row, (_, _, layers) in enumerate(group):
            pad_layer(host0[row], layers[0], 1)
            pad_layer(host1[row], layers[1], 2)
            pad_layer(host2[row], layers[2], 4)
        with torch.inference_mode():
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    tensors = [torch.from_numpy(h).to(self.device) for h in (host0, host1, host2)]
                    audio = self.model.decode(tensors).float().squeeze(1).cpu().numpy()
                self.stream.synchronize()
            else:
                tensors = [torch.from_numpy(h).to(self.device) for h in (host0, host1, host2)]
                audio = self.model.decode(tensors).float().squeeze(1).cpu().numpy()
        results = []
        for row, (key, _, _) in enumerate(group):
            samples = expected_samples(frame_counts[row])
            results.append((key, np.ascontiguousarray(audio[row, :samples], dtype=np.float32)))
        return results

    def _decode_safe(self, group):
        try:
            return self._decode_group(group), []
        except Exception as exc:
            oom = is_oom(exc)
            if oom:
                self._release()
            if len(group) > 1:
                log.warning("decode_split size=%d oom=%s %s", len(group), oom, type(exc).__name__)
                mid = len(group) // 2
                ok_a, bad_a = self._decode_safe(group[:mid])
                ok_b, bad_b = self._decode_safe(group[mid:])
                return ok_a + ok_b, bad_a + bad_b
            reason = "decode_oom" if oom else "decode_error"
            log.warning("decode_failed key=%s %s", group[0][0], type(exc).__name__)
            return [], [(group[0][0], reason)]

    def decode_batch(self, entries):
        prepared = []
        failed = []
        for key, tokens in entries:
            cleaned = clean_tokens(tokens)
            if cleaned.size < FRAME:
                failed.append((key, "too_few_tokens"))
                continue
            layers = tokens_to_layers(cleaned, self.tolerance)
            if layers is None or not valid_codes(layers):
                failed.append((key, "bad_codes"))
                continue
            prepared.append((key, cleaned, layers))
        if not prepared:
            return [], failed
        prepared.sort(key=lambda entry: entry[1].size)
        decoded, bad = self._decode_safe(prepared)
        return decoded, failed + bad