"""Mốc 3: periodic (DFT-even) Hann window, denominator N not N-1."""
import numpy as np


def periodic_hann(n_fft: int) -> np.ndarray:
    n = np.arange(n_fft, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)


def apply_hann(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 2:
        raise ValueError("Expected frames of shape [N, T]")
    return frames * periodic_hann(frames.shape[0])[:, None]
