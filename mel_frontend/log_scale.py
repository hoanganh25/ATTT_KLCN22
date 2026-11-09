"""Mốc 7: natural log (NOT librosa.power_to_db)."""
import numpy as np


def mel_to_logmel(mel_power: np.ndarray, eps: float) -> np.ndarray:
    if np.any(mel_power < -1e-10):
        raise ValueError("Power Mel must be nonnegative")
    return np.log(np.maximum(mel_power, 0.0) + eps)

