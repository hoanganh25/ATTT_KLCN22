"""MFCC transform for the *existing* V0.1 natural-log power-Mel representation.

Unlike librosa.feature.mfcc defaults (often dB-scaled Mel), this uses DCT-II
on EXACTLY the cached raw ln(MelPower + 1e-8), keeping V0.1 DSP definitions.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import dct


def logmel_to_mfcc(logmel: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
    """[96,T] raw natural-log Mel -> [n_mfcc,T] MFCC (DCT-II, orthonormal)."""
    x = np.asarray(logmel)
    if x.ndim != 2 or x.shape[0] != 96:
        raise ValueError(f"Expected raw log-Mel [96,T], got {x.shape}")
    if not 1 <= n_mfcc <= 96:
        raise ValueError("n_mfcc must be between 1 and 96")
    if x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Empty or non-finite log-Mel")
    mfcc = dct(x.astype(np.float64), type=2, axis=0, norm="ortho")[:n_mfcc]
    return np.ascontiguousarray(mfcc, dtype=np.float32)
