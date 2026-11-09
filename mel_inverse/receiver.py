"""Bridge from the unchanged V0.1 decoder to inference-only waveform synthesis.

For full utterances: split into 96-frame patches, apply encoder/channel/decoder,
average overlapping decoded patches in ORIGINAL log-Mel units, then vocode
ONCE (avoid independent per-patch Griffin-Lim artifacts).
"""
from __future__ import annotations
import numpy as np
import torch
from mel_frontend.config import MelConfig
from prepare_dataset import positions
from .vocoder import synthesize


def recover_full_logmel(logmel: np.ndarray, encoder, decoder, channel,
                        device, mean, std, batch_size: int = 12) -> np.ndarray:
    logmel = np.asarray(logmel, dtype=np.float32)
    if logmel.ndim != 2 or logmel.shape[0] != 96 or logmel.shape[1] < 1:
        raise ValueError('V0.1 requires log-Mel [96, T>0]')
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.shape != (96,) or std.shape != (96,) or np.any(std <= 0):
        raise ValueError('Expected train-only per-bin mean/std of length 96')
    if batch_size < 1:
        raise ValueError('batch_size must be >=1')
    T = logmel.shape[1]
    starts = positions(T)
    sums = np.zeros((96, T), dtype=np.float64)
    counts = np.zeros(T, dtype=np.int32)
    with torch.inference_mode():
        for j in range(0, len(starts), batch_size):
            group = starts[j:j+batch_size]
            batch = np.zeros((len(group), 1, 96, 96), dtype=np.float32)
            for k, start in enumerate(group):
                valid = min(96, T-start)
                batch[k, 0, :, :valid] = ((logmel[:, start:start+valid]
                                         - mean[:, None]) / std[:, None])
            x = torch.from_numpy(batch).to(device)
            est = decoder(channel(encoder(x)))
            if est.shape != x.shape:
                raise RuntimeError(f'Decoder output {est.shape} != input {x.shape}')
            est = est[:, 0].detach().cpu().numpy()
            for k, start in enumerate(group):
                valid = min(96, T-start)
                sums[:, start:start+valid] += (est[k, :, :valid].astype(np.float64)
                                                 * std[:, None] + mean[:, None])
                counts[start:start+valid] += 1
    if np.any(counts == 0):
        raise RuntimeError('Reconstructed log-Mel has missing frames')
    return (sums / counts[None, :]).astype(np.float32)


def reconstruct_received_speech(logmel: np.ndarray, encoder, decoder, channel,
                                device, mean, std, cfg: MelConfig, n_samples: int,
                                *, batch_size: int = 12, griffin_iters: int = 64,
                                seed: int = 42):
    """Return (received_logmel [96,T], reconstructed_waveform [N], clip_fraction)."""
    reconstructed = recover_full_logmel(logmel, encoder, decoder, channel,
                                        device, mean, std, batch_size)
    audio, clip = synthesize(reconstructed, cfg, n_samples, n_iter=griffin_iters, seed=seed)
    return reconstructed, audio, clip
