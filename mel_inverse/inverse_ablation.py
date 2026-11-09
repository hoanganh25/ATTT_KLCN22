"""TEST-only diagnostic controls for inverse Mel / phase recovery.

NEVER use `oracle_phase_wave` at the receiver: it deliberately uses the
reference STFT phase, and therefore leaks TEST-ground-truth information. It
exists solely to isolate the magnitude inversion error in an ablation.

Forward STFT is taken from the unchanged V0.1 frontend, not a substitute
front-end. All syntheses use the same periodic Hann / centered iSTFT convention.
"""
from __future__ import annotations

import time
import numpy as np

from mel_frontend.config import MelConfig
from mel_frontend.framing import frame_wave
from mel_frontend.windowing import apply_hann
from mel_frontend.stft import complex_stft
from mel_frontend.mel_filterbank import make_mel_bank
from mel_inverse.vocoder import inverse_log, inverse_mel_magnitude, griffin_lim_wave
from mel_inverse.dm_fgla import dm_fgla_wave


def reference_stft(waveform: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """Exactly the V0.1 frontend's centered/constant-pad periodic-Hann STFT."""
    x = np.asarray(waveform)
    spec = complex_stft(apply_hann(frame_wave(x, cfg.n_fft, cfg.hop_length)))
    if spec.shape != (cfg.n_fft // 2 + 1, 1 + len(x) // cfg.hop_length):
        raise ValueError(f'Unexpected frontend STFT shape: {spec.shape}')
    return spec


def istft_same_frontend(spec: np.ndarray, cfg: MelConfig, n_samples: int) -> np.ndarray:
    import librosa
    y = librosa.istft(spec, hop_length=cfg.hop_length, win_length=cfg.n_fft,
                      window='hann', center=True, length=n_samples)
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (n_samples,) or not np.isfinite(y).all():
        raise RuntimeError('Invalid diagnostic iSTFT waveform')
    return y


def oracle_phase_wave(magnitude: np.ndarray, reference_complex_stft: np.ndarray,
                      cfg: MelConfig, n_samples: int) -> np.ndarray:
    """Use GROUND-TRUTH PHASE with a chosen magnitude. Diagnostic ONLY."""
    mag = np.asarray(magnitude, dtype=np.float64)
    ref = np.asarray(reference_complex_stft)
    if mag.shape != ref.shape or np.any(mag < 0) or not np.isfinite(mag).all():
        raise ValueError('Magnitude must match reference complex STFT and be nonnegative')
    phase = np.divide(ref, np.abs(ref), out=np.ones_like(ref), where=np.abs(ref) > 1e-12)
    return istft_same_frontend(mag * phase, cfg, n_samples)


def nnls_magnitude(logmel: np.ndarray, cfg: MelConfig):
    start = time.perf_counter()
    power, clipped = inverse_log(logmel, cfg)  # unchanged clipping and inverse-log
    magnitude = inverse_mel_magnitude(power, cfg)  # unchanged librosa NNLS
    elapsed = time.perf_counter() - start
    return magnitude, float(clipped), float(elapsed)


def magnitude_diagnostics(mag: np.ndarray, true_mag: np.ndarray, mel_log: np.ndarray,
                          cfg: MelConfig, mel_bank=None) -> dict:
    """Disentangle linear-STFT inversion error from log-Mel reprojection error."""
    a = np.asarray(mag, np.float64)
    b = np.asarray(true_mag, np.float64)
    if a.shape != b.shape:
        raise ValueError(f'NNLS magnitude {a.shape} != true STFT magnitude {b.shape}')
    bank = make_mel_bank(cfg) if mel_bank is None else mel_bank
    power_est = bank @ np.square(a)
    mel_reproj = np.log(power_est + cfg.eps)
    delta = a - b
    return {
        'nnls_stft_magnitude_mse': float(np.mean(delta ** 2)),
        'nnls_stft_magnitude_nmse': float(np.sum(delta ** 2) / max(np.sum(b ** 2), 1e-12)),
        'nnls_logmel_cycle_mse': float(np.mean((mel_reproj - np.asarray(mel_log)) ** 2)),
    }


def recover_phase(mag: np.ndarray, cfg: MelConfig, n_samples: int, method: str,
                  *, seed: int, griffin_iters: int = 64, dm_iters: int = 24,
                  fgla_iters: int = 40, dm_beta: float = 1.0,
                  fgla_momentum: float = 0.99) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if method == 'griffin_lim':
        y = griffin_lim_wave(mag, cfg, n_samples, n_iter=griffin_iters, seed=seed)
    elif method == 'dm_fgla':
        y = dm_fgla_wave(mag, cfg, n_samples, dm_iters=dm_iters,
                         fgla_iters=fgla_iters, beta=dm_beta,
                         momentum=fgla_momentum, seed=seed)
    else:
        raise ValueError(f'Unknown phase recovery: {method}')
    return y, float(time.perf_counter() - start)
