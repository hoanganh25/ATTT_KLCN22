"""Read WAV/FLAC -> mono waveform @ target sample rate."""
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MelConfig


def read_wave(path: str | Path, cfg: MelConfig) -> np.ndarray:
    """Load PCM/float WAV, downmix (mean of channels), resample if needed."""
    data, source_sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError("Audio file is empty")
    mono = np.mean(data, axis=1, dtype=np.float32)
    if source_sr != cfg.sr:
        import librosa
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=cfg.sr)
    if not np.isfinite(mono).all():
        raise ValueError("Input contains NaN or infinity")
    # Deliberately DO NOT peak-normalize: amplitude must be preserved for metrics.
    return np.asarray(mono, dtype=np.float32)

