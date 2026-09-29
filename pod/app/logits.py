from typing import List

from .prompt import AUDIO_END, AUDIO_OFFSET, EOS, HEADER_TOKENS

FRAME = 7
BAND = 4096

_PROCESSOR_CACHE = {}


class RangeMask:
    """Allow audio tokens, EOS and (optionally) the two header tokens; forbid everything else.

    Runs once per sequence per decode step inside vLLM, so it must be cheap: the mask is
    built once per device/dtype and added in place (no new vocab-sized tensor per call).
    """

    def __init__(self, allow_header=True):
        self.allow_header = allow_header
        self.mask = None

    def _get(self, logits):
        mask = self.mask
        if mask is None or mask.device != logits.device or mask.dtype != logits.dtype:
            import torch

            mask = torch.full(
                (logits.shape[-1],), float("-inf"), device=logits.device, dtype=logits.dtype
            )
            mask[AUDIO_OFFSET:AUDIO_END] = 0.0
            mask[EOS] = 0.0
            if self.allow_header:
                for token in HEADER_TOKENS:
                    mask[token] = 0.0
            self.mask = mask
        return mask

    def __call__(self, output_ids: List[int], logits):
        return logits.add_(self._get(logits))


class FrameMask:
    """Strict 7-slot mask. Does not allow header tokens, so use it only for models without them."""

    def __init__(self):
        self.masks = None

    def _get(self, logits):
        first = self.masks[0] if self.masks else None
        if first is None or first.device != logits.device or first.dtype != logits.dtype:
            import torch

            masks = []
            for slot in range(FRAME):
                mask = torch.full(
                    (logits.shape[-1],), float("-inf"), device=logits.device, dtype=logits.dtype
                )
                start = AUDIO_OFFSET + slot * BAND
                mask[start : start + BAND] = 0.0
                mask[EOS] = 0.0
                masks.append(mask)
            self.masks = masks
        return self.masks

    def __call__(self, output_ids: List[int], logits):
        return logits.add_(self._get(logits)[len(output_ids) % FRAME])


def allowed_token_ids(allow_header=True):
    ids = list(range(AUDIO_OFFSET, AUDIO_END))
    ids.append(EOS)
    if allow_header:
        ids.extend(HEADER_TOKENS)
    return ids


def build_processors(kind, allow_header=True):
    if kind == "frame":
        return [FrameMask()]
    if kind == "range":
        return [RangeMask(allow_header)]
    return []


def sampling_restrictions(kind, vocab_size=None, allow_header=True):
    if kind == "allowed":
        return {"allowed_token_ids": allowed_token_ids(allow_header)}
    if kind in ("range", "frame"):
        key = (kind, allow_header)
        if key not in _PROCESSOR_CACHE:
            _PROCESSOR_CACHE[key] = build_processors(kind, allow_header)
        return {"logits_processors": _PROCESSOR_CACHE[key]}
    return {}