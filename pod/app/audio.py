import io
import os
import subprocess
import threading

import numpy as np
import soundfile as sf
import soxr

_LOCAL = threading.local()
_OGG_MAGIC = b"OggS"


def ffmpeg_binary():
    return os.environ.get("FFMPEG_BIN") or "ffmpeg"


def trim_edges(audio, sample_rate, threshold, pad_seconds):
    if audio.size == 0:
        return audio
    active = np.flatnonzero(np.abs(audio) > threshold)
    if active.size == 0:
        return audio[:0]
    pad = int(pad_seconds * sample_rate)
    start = max(0, int(active[0]) - pad)
    end = min(audio.size, int(active[-1]) + pad + 1)
    return audio[start:end]


def peak_normalize(audio, target):
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio
    return audio * (target / peak)


def rms(audio):
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def loudness_normalize(audio, target_db, peak_target):
    level = rms(audio)
    if level <= 1e-8:
        return audio
    scaled = audio * ((10.0 ** (target_db / 20.0)) / level)
    peak = float(np.max(np.abs(scaled)))
    if peak > peak_target:
        scaled = scaled * (peak_target / peak)
    return scaled


def validate(audio, sample_rate, settings):
    if audio is None or audio.size == 0:
        return "empty"
    if not np.isfinite(audio).all():
        return "non_finite"
    duration = audio.size / float(sample_rate)
    if duration < settings.min_duration:
        return "too_short"
    if duration > settings.max_duration:
        return "too_long"
    if rms(audio) < settings.rms_gate:
        return "silent"
    return None


def process(audio, settings):
    sr = settings.source_sample_rate
    trimmed = trim_edges(audio, sr, settings.trim_threshold, settings.trim_pad_seconds)
    normalized = peak_normalize(trimmed, settings.peak_target)
    reason = validate(normalized, sr, settings)
    if reason is not None:
        return None, reason
    if settings.loudness_normalize:
        normalized = loudness_normalize(
            normalized, settings.loudness_target_db, settings.peak_target
        )
    return normalized.astype(np.float32, copy=False), None


def encode_wav(audio, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class OpusEncoder:
    def __init__(self, sample_rate, bitrate, ffmpeg=None):
        self.sample_rate = sample_rate
        self.bitrate = bitrate
        self.ffmpeg = ffmpeg or ffmpeg_binary()

    def encode(self, pcm):
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            self.bitrate,
            "-vbr",
            "constrained",
            "-application",
            "voip",
            "-f",
            "ogg",
            "pipe:1",
        ]
        result = subprocess.run(
            command,
            input=pcm.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.startswith(_OGG_MAGIC):
            raise RuntimeError("opus_encode_failed")
        return result.stdout


def to_int16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def encode_ogg(audio, settings):
    resampled = soxr.resample(
        audio.astype(np.float32, copy=False),
        settings.source_sample_rate,
        settings.opus_sample_rate,
        quality="HQ",
    )
    encoder = getattr(_LOCAL, "encoder", None)
    if encoder is None:
        encoder = OpusEncoder(
            settings.opus_sample_rate,
            settings.opus_bitrate,
            ffmpeg_binary(),
        )
        _LOCAL.encoder = encoder
    return encoder.encode(to_int16(resampled))


def encode(audio, fmt, settings):
    if fmt == "wav":
        return encode_wav(audio, settings.source_sample_rate), "audio/wav", "wav"
    return encode_ogg(audio, settings), "audio/ogg", "ogg"


def bitrate_bps(value):
    text = str(value).strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1000)
    return int(text)