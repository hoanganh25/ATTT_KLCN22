"""Torch backend for the V0.1 POWER-Mel ADMM-inspired joint solver.

This is NOT Algorithm 2 of Masuyama et al. (ICASSP 2025): their linear
magnitude-Mel proximal is replaced by a nonconvex power-Mel proximal.
Reproduces the splitting and projected-gradient/backtracking of the existing
power_mel_joint/solver.py while allowing tensors to reside on CUDA.
No weights, no training, no use of original waveform or original STFT phase.

The function ``invert_power_mel_joint_torch`` can snapshot multiple iteration
counts in one pass (e.g. 32 and 64), using exactly the same initialization.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import torch

from mel_frontend.config import MelConfig
from mel_frontend.mel_filterbank import make_mel_bank
from mel_inverse.vocoder import inverse_log
from mel_inverse.power_joint_cpu import PowerMelJointConfig


def _select_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install a CUDA PyTorch build and check torch.cuda.is_available().')
    if dev.type not in ('cuda', 'cpu'):
        raise ValueError('Only cpu or cuda devices supported')
    return dev


def _torch_stft(wave: torch.Tensor, cfg: MelConfig, window: torch.Tensor) -> torch.Tensor:
    # center=True and pad_mode='constant' match the original V0.1 analysis.
    return torch.stft(wave, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                      win_length=cfg.n_fft, window=window, center=True,
                      pad_mode='constant', normalized=False, onesided=True,
                      return_complex=True)


def _torch_istft(z: torch.Tensor, cfg: MelConfig, window: torch.Tensor, n_samples: int) -> torch.Tensor:
    return torch.istft(z, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                       win_length=cfg.n_fft, window=window, center=True,
                       normalized=False, onesided=True, length=n_samples)


def _prox_power(w: torch.Tensor, phi: torch.Tensor, E: torch.Tensor,
                M: torch.Tensor, lam: float, rho: float, steps: int,
                E_fro2: torch.Tensor) -> torch.Tensor:
    """The existing NumPy solver's nonnegative proximal with backtracking.

    Line search uses a small number of GPU scalar synchronizations; it is
    retained for numerical comparability and monotone inner updates.
    """
    w = w.clamp_min(0)
    for _ in range(steps):
        resid = E @ w.square() - M
        etr = E.T @ resid
        grad = 2.0 * lam * w * etr + rho * (w - phi)
        if not bool(torch.isfinite(grad).all().item()):
            raise FloatingPointError('Non-finite W gradient')
        norm_inf = etr.abs().max()
        maxw2 = w.square().max()
        lips = rho + 2.0 * lam * norm_inf + 4.0 * lam * E_fro2 * maxw2
        step = 1.0 / torch.clamp(lips, min=1e-12)
        old_loss = (0.5*lam*resid.square().sum() + 0.5*rho*(w-phi).square().sum())
        accepted = False
        for _ in range(12):
            trial = (w - step * grad).clamp_min(0)
            r_trial = E @ trial.square() - M
            loss = 0.5*lam*r_trial.square().sum() + 0.5*rho*(trial-phi).square().sum()
            if bool((torch.isfinite(loss) & (loss <= old_loss + 1e-9*torch.clamp(old_loss, min=1.0))).item()):
                w = trial
                accepted = True
                break
            step = step * 0.5
        if not accepted:
            break
    return w


@torch.no_grad()
def invert_power_mel_joint_torch(
        logmel: np.ndarray, cfg: MelConfig, n_samples: int,
        options: PowerMelJointConfig | None = None, *,
        device: str = 'cuda', precision: str = 'float32',
        snapshots: Iterable[int] | None = None,
) -> dict[int, tuple[np.ndarray, dict]]:
    """Invert V0.1 raw log-Mel power [96,T] to waveforms at given iterations.

    Args:
      snapshots: 1..options.iterations; if None, only final iteration.
      precision: float64 recommended for CPU-vs-CUDA parity;
                 float32 is generally faster on RTX 3050 but may differ.
    Returns: {iteration: (float32_waveform, diagnostics)}.

    Time includes preprocessing/initialization + all iterations up to snapshot,
    including GPU -> CPU waveform transfer; CUDA is synchronized at snapshots.
    """
    opts = options or PowerMelJointConfig()
    opts.validate()
    if n_samples < 1:
        raise ValueError('n_samples must be >=1')
    if precision not in ('float32', 'float64'):
        raise ValueError('precision must be float32 or float64')
    dev = _select_device(device)
    points = tuple(sorted(set(snapshots if snapshots is not None else [opts.iterations])))
    if not points or points[0] < 1 or points[-1] > opts.iterations:
        raise ValueError('snapshots must be between 1 and options.iterations')
    dtype = torch.float32 if precision=='float32' else torch.float64
    # Keep inverse_log exactly as V0.1 (same eps and clipping).
    mel, clip_fraction = inverse_log(logmel, cfg)
    t0=time.perf_counter()
    mel_t=torch.as_tensor(np.asarray(mel),dtype=dtype,device=dev)
    bank=torch.as_tensor(np.asarray(make_mel_bank(cfg)),dtype=dtype,device=dev)
    if opts.init == 'nnls':
        from mel_inverse.vocoder import inverse_mel_magnitude
        # Optional NNLS initializes on CPU; not a fully GPU-only path.
        magnitude=torch.as_tensor(inverse_mel_magnitude(mel,cfg),dtype=dtype,device=dev)
    else:
        ht_m=(bank.T @ mel_t).clamp_min(0)
        row_sq=bank.square().sum(dim=1)
        power=(bank.T @ (mel_t / row_sq.clamp_min(1e-24).unsqueeze(1))).clamp_min(0)
        for _ in range(opts.init_iterations):
            denominator=(bank.T @ (bank @ power)).clamp_min(1e-24)
            power=(power*(ht_m/denominator)).clamp_min(0)
        magnitude=power.sqrt()
    if not bool(torch.isfinite(magnitude).all().item()):
        raise RuntimeError('Non-finite magnitude initialization')
    amplitude_scale=torch.sqrt(magnitude.square().mean()).clamp_min(1e-6)
    mel_scale=mel_t.mean().clamp_min(1e-10)
    E=bank*(amplitude_scale.square()/mel_scale)
    M=mel_t/mel_scale
    y=magnitude/amplitude_scale
    w=y.clone()
    # Match original numpy solver's per-file seeded initial phase precisely.
    random=np.random.RandomState(opts.seed)
    phases_np=np.exp(1j*(2.0*np.pi*random.random(size=tuple(y.shape))))
    phases=torch.as_tensor(phases_np,device=dev,
                           dtype=torch.complex64 if precision=='float32' else torch.complex128)
    z=y*phases
    v=torch.zeros_like(z)
    u=torch.zeros_like(y)
    window=torch.hann_window(cfg.n_fft,periodic=True,device=dev,dtype=dtype)
    e_fro2=E.square().sum()
    outputs={}
    trace=[]
    for k in range(1,opts.iterations+1):
        psi=z+v
        psi_phase=psi/psi.abs().clamp_min(1e-12)
        x=((y+opts.rho*psi.abs())/(1+opts.rho))*psi_phase
        phi=y+u
        w=_prox_power(w,phi,E,M,opts.lambda_mel,opts.rho,opts.inner_steps,e_fro2)
        z=_torch_stft(_torch_istft(x-v,cfg,window,n_samples),cfg,window)
        if z.shape != x.shape:
            raise RuntimeError('STFT shape drift')
        y=((x.abs()+opts.rho*(w-u))/(1+opts.rho)).clamp_min(0)
        v=v+z-x
        u=u+y-w
        if k % opts.trace_every==0 or k==1 or k in points:
            if not bool((torch.isfinite(y).all() & torch.isfinite(z).all() &
                         torch.isfinite(u).all() & torch.isfinite(v).all()).item()):
                raise FloatingPointError(f'ADMM diverged at iteration {k}')
            consistency=bank @ (z.abs()*amplitude_scale).square()
            residual=torch.linalg.vector_norm(consistency-mel_t)/torch.linalg.vector_norm(mel_t).clamp_min(1e-12)
            rec={'iteration':k,
                 'relative_mel_power_residual':float(residual.item()),
                 'relative_admm_primal_X':float((torch.linalg.vector_norm(z-x)/torch.linalg.vector_norm(z).clamp_min(1e-12)).item()),
                 'relative_admm_primal_Y':float((torch.linalg.vector_norm(y-w)/torch.linalg.vector_norm(y).clamp_min(1e-12)).item())}
            trace.append(rec)
        if k in points:
            wave_t=_torch_istft(z*amplitude_scale,cfg,window,n_samples)
            if dev.type=='cuda': torch.cuda.synchronize(dev)
            wave=wave_t.detach().cpu().numpy().astype(np.float32)
            if wave.shape!=(n_samples,) or not np.isfinite(wave).all():
                raise RuntimeError('Invalid synthesized waveform')
            outputs[k]=(wave,{
                'algorithm':'power_mel_admm_joint_adaptation_torch',
                'not_published_magnitude_algorithm':True,
                'device_joint':str(dev),'precision':precision,
                'initialization':opts.init,'initialization_is_nnls':opts.init=='nnls',
                'iterations':k,'lambda_mel':opts.lambda_mel,'rho':opts.rho,
                'inner_steps':opts.inner_steps,'init_iterations':opts.init_iterations,
                'clip_fraction':clip_fraction,'time_s':time.perf_counter()-t0,
                'final_mel_power_residual':trace[-1]['relative_mel_power_residual'],
                'trace':list(trace)})
    return outputs
