import subprocess
import numpy as np

def to_ogg(audio, sr):
    if audio.size < sr * 0.2 or not np.isfinite(audio).all(): raise ValueError("empty/invalid audio")
    pcm = np.clip(audio, -1, 1).astype("<f4").tobytes()
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-c:a", "libopus", "-b:a", "24k", "-application", "voip", "-ar", "16000", "-f", "ogg", "pipe:1"],
                       input=pcm, capture_output=True, timeout=30)
    if p.returncode or not p.stdout: raise RuntimeError("ffmpeg: " + p.stderr.decode()[:150])
    return p.stdout
