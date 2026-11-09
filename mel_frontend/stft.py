"""Mốc 4+5: real FFT -> complex STFT -> magnitude/power."""
import numpy as np


def complex_stft(windowed_frames: np.ndarray) -> np.ndarray:
    if windowed_frames.ndim != 2:
        raise ValueError("Expected windowed frames [N, T]")
    return np.fft.rfft(windowed_frames, axis=0)


def power_spectrogram(X: np.ndarray) -> np.ndarray:
    return np.abs(X) ** 2   # power=2, no FFT energy normalization
