import numpy as np

from .prompt import AUDIO_END, AUDIO_OFFSET

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


def tokens_to_layers(tokens):
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.size == 0 or arr.size % FRAME:
        return None
    slots = np.arange(arr.size) % FRAME
    codes = arr - AUDIO_OFFSET - slots * BAND
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


class Codec:
    def __init__(self, snac_dir, device="cuda"):
        import torch
        from snac import SNAC

        self.torch = torch
        self.device = device
        model = SNAC.from_pretrained(snac_dir)
        self.model = model.to(device).half().eval()
        self.stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None

    def _decode_group(self, group):
        torch = self.torch
        n = len(group)
        frame_counts = [tokens.size // FRAME for _, tokens, _ in group]
        max_frames = max(frame_counts)
        host0 = np.zeros((n, max_frames), dtype=np.int64)
        host1 = np.zeros((n, max_frames * 2), dtype=np.int64)
        host2 = np.zeros((n, max_frames * 4), dtype=np.int64)
        for row, (_, _, layers) in enumerate(group):
            host0[row, : layers[0].size] = layers[0]
            host1[row, : layers[1].size] = layers[1]
            host2[row, : layers[2].size] = layers[2]
        pinned = [
            torch.from_numpy(host0).pin_memory(),
            torch.from_numpy(host1).pin_memory(),
            torch.from_numpy(host2).pin_memory(),
        ]
        with torch.inference_mode():
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    tensors = [t.to(self.device, non_blocking=True) for t in pinned]
                    audio = self.model.decode(tensors)
                    audio = audio.float().squeeze(1).cpu().numpy()
                self.stream.synchronize()
            else:
                tensors = [t.to(self.device) for t in pinned]
                audio = self.model.decode(tensors).float().squeeze(1).cpu().numpy()
        results = []
        for row, (key, _, _) in enumerate(group):
            samples = expected_samples(frame_counts[row])
            results.append((key, np.ascontiguousarray(audio[row, :samples], dtype=np.float32)))
        return results

    def decode_batch(self, entries):
        prepared = []
        failed = []
        for key, tokens in entries:
            cleaned = clean_tokens(tokens)
            if cleaned.size < FRAME:
                failed.append((key, "too_few_tokens"))
                continue
            layers = tokens_to_layers(cleaned)
            if layers is None or not valid_codes(layers):
                failed.append((key, "bad_codes"))
                continue
            prepared.append((key, cleaned, layers))
        if not prepared:
            return [], failed
        prepared.sort(key=lambda entry: entry[1].size)
        return self._decode_group(prepared), failed