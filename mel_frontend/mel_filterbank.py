"""Mốc 6: Slaney Mel bank maps linear-frequency power -> Mel power."""
import librosa
import numpy as np

from .config import MelConfig


def make_mel_bank(cfg: MelConfig) -> np.ndarray:
    return librosa.filters.mel(
        sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
        fmin=cfg.fmin, fmax=cfg.fmax,
        htk=False, norm="slaney", dtype=np.float64,
    )


def to_mel_power(power: np.ndarray, bank: np.ndarray) -> np.ndarray:
    if power.ndim != 2 or bank.shape[1] != power.shape[0]:
        raise ValueError("Expected bank [n_mels, n_fft//2+1] and power [F, T]")
    return bank @ power
