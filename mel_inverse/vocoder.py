"""Non-trainable V0.1 log-Mel power inversion via NNLS and Griffin-Lim.

Exact forward convention in mel_frontend:
  L = ln(W_slaney @ abs(STFT(x))**2 + eps), n_fft=1024, hop=256, sr=16k.
The inverse is APPROXIMATE: Mel projection and phase loss are not invertible.
Do not include the reference phase or peak normalization in receiver synthesis.
"""
from __future__ import annotations

import numpy as np
from mel_frontend.config import MelConfig


def inverse_log(logmel: np.ndarray, cfg: MelConfig,
                clip_min: float = -30.0, clip_max: float = 25.0) -> tuple[np.ndarray, float]:
    """Undo natural-log power Mel; report fraction requiring numeric clipping."""
    a = np.asarray(logmel, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != cfg.n_mels or a.shape[1] < 1:
        raise ValueError(f'Expected [{cfg.n_mels}, T>0] log-Mel, got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError('log-Mel contains NaN or Inf')
    if not clip_min < clip_max:
        raise ValueError('Invalid clamp interval')
    clipped_fraction = float(np.mean((a < clip_min) | (a > clip_max)))
    mel_power = np.maximum(np.exp(np.clip(a, clip_min, clip_max)) - cfg.eps, 0.0)
    return mel_power, clipped_fraction


def inverse_mel_magnitude(mel_power: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """Non-negative least-squares inverse of 96-bin Slaney power Mel.

    librosa.feature.inverse.mel_to_stft returns approximate magnitude, since
    input Mel was derived from |STFT|**2 (power=2), NOT decibels or magnitudes.
    """
    import librosa
    mel_power = np.asarray(mel_power, dtype=np.float64)
    if mel_power.ndim != 2 or mel_power.shape[0] != cfg.n_mels:
        raise ValueError('Bad Mel power dimensions')
    if np.any(mel_power < 0) or not np.isfinite(mel_power).all():
        raise ValueError('Mel power must be finite and non-negative')
    out = librosa.feature.inverse.mel_to_stft(
        mel_power, sr=cfg.sr, n_fft=cfg.n_fft, power=2.0,
        fmin=cfg.fmin, fmax=cfg.fmax, htk=False, norm='slaney')
    if out.shape != (cfg.n_fft // 2 + 1, mel_power.shape[1]):
        raise ValueError(f'Unexpected STFT magnitude shape {out.shape}')
    return out


def griffin_lim_wave(magnitude: np.ndarray, cfg: MelConfig, n_samples: int,
                     *, n_iter: int = 64, seed: int = 42,
                     momentum: float = 0.99) -> np.ndarray:
    """Generate full waveform using random-phase Griffin-Lim, exact length."""
    import librosa
    if n_samples <= 0 or n_iter < 1 or not (0 <= momentum < 1):
        raise ValueError('Require n_samples>0, n_iter>=1 and 0<=momentum<1')
    # Use old/new librosa RNG parameter compatibility without assuming version.
    kw = dict(n_iter=n_iter, hop_length=cfg.hop_length, win_length=cfg.n_fft,
              window='hann', center=True, length=n_samples,
              init='random', momentum=momentum)
    from inspect import signature
    if 'rng' in signature(librosa.griffinlim).parameters:
        kw['rng'] = seed
    else:
        kw['random_state'] = seed
    wave = np.asarray(librosa.griffinlim(magnitude, **kw), dtype=np.float32)
    if wave.shape != (n_samples,) or not np.isfinite(wave).all():
        raise RuntimeError('Waveform inverse failed: bad length/non-finite samples')
    return wave


def synthesize(logmel: np.ndarray, cfg: MelConfig, n_samples: int, *,
               n_iter: int = 64, seed: int = 42, clip_min: float = -30.0,
               clip_max: float = 25.0) -> tuple[np.ndarray, float]:
    """Full log-Mel -> inverse-log -> NNLS inverse Mel -> Griffin-Lim.

    Call ONLY after denormalizing V0.1 decoder's z-scored output back to
    natural-log Mel POWER, and AFTER stitching all [96,96] patches.
    Returns (audio_float32, fraction_of_clipped_logmel_values).
    """
    power, clipped = inverse_log(logmel, cfg, clip_min, clip_max)
    mag = inverse_mel_magnitude(power, cfg)
    wave = griffin_lim_wave(mag, cfg, n_samples, n_iter=n_iter, seed=seed)
    return wave, clipped
