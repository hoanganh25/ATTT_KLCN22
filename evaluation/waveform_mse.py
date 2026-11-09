"""Waveform MSE diagnostics (no implicit alignment, gain fit, or normalization).

For comparison with original speech, use the original 16-kHz mono waveform and
whole-utterance waveform reconstructed with exactly the same number of samples.
Waveform MSE is phase/time-sensitive: see clean-Mel Griffin-Lim control.
"""
import math
import numpy as np


def score_waveform_mse(reference, estimate):
    """Return raw waveform MSE, RMSE, energy-normalized MSE, reference power.

    NMSE = mean((reference - estimate)^2) / mean(reference^2). It is None
    for (near-)silent references. Does not time-align, gain-match or clip.
    """
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(estimate, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError('Reference and estimate must have equal, nonzero length')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Waveform contains NaN or infinity')
    mse = float(np.mean(np.square(a - b)))
    power = float(np.mean(np.square(a)))
    return {
        'waveform_mse': mse,
        'waveform_rmse': math.sqrt(mse),
        'waveform_nmse': mse / power if power > 1e-12 else None,
        'reference_mean_power': power,
    }
