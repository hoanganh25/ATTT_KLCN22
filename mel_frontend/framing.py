"""Mốc 2: centered framing; equivalent to librosa.stft center=True constant pad."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def frame_wave(x: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Return frames [n_fft, T], center-pad n_fft//2 on each side with zeros."""
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Expected nonempty mono waveform")
    padded = np.pad(x, (n_fft // 2, n_fft // 2), mode="constant")
    # [T, N] -> [N, T], no accidental Hann/FFT handled at this stage.
    views = sliding_window_view(padded, n_fft)[::hop_length]
    return np.ascontiguousarray(views.T, dtype=np.float64)
