from typing import List

from .prompt import AUDIO_END, AUDIO_OFFSET, EOS, HEADER_TOKENS

FRAME = 7
BAND = 4096
HEADER_POSITIONS = 2

_PROCESSOR_CACHE = {}


class RangeMask:
    def __init__(self, allow_header=True):
        self.allow_header = allow_header
        self.base = None
        self.head = None

    def _build(self, logits):
        import torch

        base = torch.full(
            (logits.shape[-1],), float("-inf"), device=logits.device, dtype=logits.dtype
        )
        base[AUDIO_OFFSET:AUDIO_END] = 0.0
        base[EOS] = 0.0
        head = base.clone()
        for token in HEADER_TOKENS:
            head[token] = 0.0
        self.base = base
        self.head = head

    def __call__(self, output_ids: List[int], logits):
        base = self.base
        if (
            base is None
            or base.device != logits.device
            or base.dtype != logits.dtype
            or base.shape[-1] != logits.shape[-1]
        ):
            self._build(logits)
        if self.allow_header and len(output_ids) < HEADER_POSITIONS:
            return logits.add_(self.head)
        return logits.add_(self.base)


class FrameMask:
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