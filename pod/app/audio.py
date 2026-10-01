import io
import os
import subprocess
import threading

import numpy as np
import soundfile as sf
import soxr

_LOCAL = threading.local()
_OGG_MAGIC = b"OggS"
FFMPEG_TIMEOUT_SECONDS = 30
SIZE_SLACK_BYTES = 2048


class EncodeError(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def ffmpeg_binary():
    return os.environ.get("FFMPEG_BIN") or "ffmpeg"


def peak_of(audio):
    return float(max(audio.max(), -audio.min()))


def trim_edges(audio, sample_rate, threshold, pad_seconds):
    if audio.size == 0:
        return audio
    active = (audio > threshold) | (audio < -threshold)
    if not active.any():
        return audio[:0]
    first = int(active.argmax())
    last = audio.size - 1 - int(active[::-1].argmax())
    pad = int(pad_seconds * sample_rate)
    start = max(0, first - pad)
    end = min(audio.size, last + pad + 1)
    return audio[start:end]


def peak_normalize(audio, target):
    if audio.size == 0:
        return audio
    peak = peak_of(audio)
    if peak <= 0.0:
        return audio
    return audio * (target / peak)


def rms(audio):
    if audio.size == 0:
        return 0.0
    flat = audio.reshape(-1)
    return float(np.sqrt(np.einsum("i,i->", flat, flat, dtype=np.float64) / flat.size))


def loudness_normalize(audio, target_db, peak_target, inplace=False):
    level = rms(audio)
    if level <= 1e-8:
        return audio
    gain = (10.0 ** (target_db / 20.0)) / level
    peak = peak_of(audio) * gain
    if peak > peak_target:
        gain *= peak_target / peak
    if inplace:
        audio *= gain
        return audio
    return audio * gain


def fade_edges(audio, sample_rate, seconds, inplace=False):
    n = min(int(seconds * sample_rate), audio.size // 2)
    if n <= 0:
        return audio
    out = audio if inplace else np.array(audio, dtype=np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def check_shape(audio, sample_rate, settings, text_chars):
    duration = audio.size / float(sample_rate)
    if duration < settings.min_duration:
        return "too_short"
    if duration > settings.max_duration:
        return "too_long"
    if settings.quality_pace_gate and text_chars:
        if duration < text_chars * settings.min_sec_per_char:
            return "pace_short"
        if duration > text_chars * settings.max_sec_per_char + settings.pace_slack_seconds:
            return "pace_long"
    return None


def process(audio, settings, text_chars=None):
    if audio is None or audio.size == 0:
        return None, "empty"
    if not np.isfinite(np.sum(audio, dtype=np.float64)):
        return None, "non_finite"
    sr = settings.source_sample_rate
    trimmed = trim_edges(audio, sr, settings.trim_threshold, settings.trim_pad_seconds)
    if trimmed.size == 0 or rms(trimmed) < settings.rms_gate:
        return None, "silent"
    reason = check_shape(trimmed, sr, settings, text_chars)
    if reason is not None:
        return None, reason
    out = peak_normalize(trimmed, settings.peak_target).astype(np.float32, copy=False)
    if np.may_share_memory(out, audio):
        out = out.copy()
    if settings.loudness_normalize:
        out = loudness_normalize(out, settings.loudness_target_db, settings.peak_target, inplace=True)
    out = fade_edges(out, sr, settings.fade_seconds, inplace=True)
    return out, None


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
        try:
            result = subprocess.run(
                command,
                input=pcm.tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise EncodeError("ffmpeg_timeout") from exc
        except OSError as exc:
            raise EncodeError("ffmpeg_unavailable") from exc
        if result.returncode != 0 or not result.stdout.startswith(_OGG_MAGIC):
            raise EncodeError("opus_encode_failed")
        return result.stdout


def to_int16(audio):
    scaled = np.clip(audio, -1.0, 1.0)
    scaled *= 32767.0
    return scaled.astype(np.int16)


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


def bitrate_bps(value):
    text = str(value).strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1000)
    return int(text)


def encoded_size_ok(size, seconds, settings):
    expected = bitrate_bps(settings.opus_bitrate) / 8.0 * seconds
    low = expected * settings.opus_size_low
    high = expected * settings.opus_size_high + SIZE_SLACK_BYTES
    return low <= size <= high


def encode(audio, fmt, settings):
    if fmt == "wav":
        data = encode_wav(audio, settings.source_sample_rate)
        if not data:
            raise EncodeError("empty_encode")
        return data, "audio/wav", "wav"
    data = encode_ogg(audio, settings)
    seconds = audio.size / float(settings.source_sample_rate)
    if not encoded_size_ok(len(data), seconds, settings):
        raise EncodeError("opus_size")
    return data, "audio/ogg", "ogg"