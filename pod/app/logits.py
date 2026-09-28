from typing import List

from .prompt import AUDIO_END, AUDIO_OFFSET, EOS

FRAME = 7
BAND = 4096

_PROCESSOR_CACHE = {}


class RangeMask:
    def __init__(self, vocab_size=None):
        import torch

        self.torch = torch
        self.vocab_size = vocab_size
        self.mask = None

    def _get(self, logits):
        if self.mask is None or self.mask.device != logits.device or self.mask.dtype != logits.dtype:
            mask = self.torch.full(
                (logits.shape[-1],), float("-inf"), device=logits.device, dtype=logits.dtype
            )
            mask[AUDIO_OFFSET:AUDIO_END] = 0.0
            mask[EOS] = 0.0
            self.mask = mask
        return self.mask

    def __call__(self, output_ids: List[int], logits):
        return logits + self._get(logits)


class FrameMask:
    def __init__(self, vocab_size=None):
        import torch

        self.torch = torch
        self.vocab_size = vocab_size
        self.masks = None

    def _get(self, logits):
        if (
            self.masks is None
            or self.masks[0].device != logits.device
            or self.masks[0].dtype != logits.dtype
        ):
            masks = []
            for slot in range(FRAME):
                mask = self.torch.full(
                    (logits.shape[-1],),
                    float("-inf"),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                start = AUDIO_OFFSET + slot * BAND
                mask[start : start + BAND] = 0.0
                mask[EOS] = 0.0
                masks.append(mask)
            self.masks = masks
        return self.masks

    def __call__(self, output_ids: List[int], logits):
        slot = len(output_ids) % FRAME
        return logits + self._get(logits)[slot]


def allowed_token_ids():
    ids = list(range(AUDIO_OFFSET, AUDIO_END))
    ids.append(EOS)
    return ids


def build_processors(kind, vocab_size=None):
    if kind == "frame":
        return [FrameMask(vocab_size)]
    if kind == "range":
        return [RangeMask(vocab_size)]
    return []


def sampling_restrictions(kind, vocab_size=None):
    if kind == "allowed":
        return {"allowed_token_ids": allowed_token_ids()}
    if kind in ("range", "frame"):
        if kind not in _PROCESSOR_CACHE:
            _PROCESSOR_CACHE[kind] = build_processors(kind, vocab_size)
        return {"logits_processors": _PROCESSOR_CACHE[kind]}
    return {}