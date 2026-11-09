"""Power-Mel ADMM-inspired JOINT inversion (not the published Algorithm 2).

Masuyama/Ueno/Ono, ICASSP 2025, Algorithm 2 applies to E|STFT|, while
Semantic Voice V0.1 transmits ln(E|STFT|**2 + eps). This implementation
retains the scaled-ADMM splitting but replaces its closed-form W proximal
update with a NONCONVEX projected-gradient update fitting power-Mel.
No learned weights, no reference magnitude/phase, no alteration to the DSP.

Variables X,Z : complex STFT; Y,W : nonnegative STFT magnitude;
V,U : scaled dual variables. In *normalized* amplitude coordinates:
   min 0.5|| |X|-Y ||² + lam/2||H (W**2)-M ||²,
   s.t. Z=X, Y=W, Z in STFT-consistent set, Y>=0.
The chosen (H,M) normalization merely rescales the Mel residual objective.

Each ADMM iteration:
  X = ((Y+rho*|Z+V|)/(1+rho))*phase(Z+V)
  W ≈ prox_{lam*power_mel_loss/rho}(Y+U)  [projected gradient]
  Z = STFT(ISTFT(X-V))
  Y = maximum((|X|+rho*(W-U))/(1+rho),0)
  V = V+Z-X ; U=U+Y-W.

Unlike the linear magnitude-Mel paper, W has NO closed-form linear solve.
Convergence/perceptual improvement are NOT guaranteed for this nonconvex
power-Mel adaptation: compare with unchanged NNLS+GL baseline on DEV.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse

from mel_frontend.config import MelConfig
from mel_frontend.mel_filterbank import make_mel_bank
from mel_inverse.vocoder import inverse_log


@dataclass(frozen=True)
class PowerMelJointConfig:
    iterations: int = 64
    lambda_mel: float = 1.0
    rho: float = 1.0
    inner_steps: int = 5
    init_iterations: int = 40
    seed: int = 42
    # NNLS optional control only; 'multiplicative' never calls inverse Mel NNLS.
    init: str = 'multiplicative'
    trace_every: int = 8

    def validate(self):
        if self.iterations < 1 or self.inner_steps < 1 or self.init_iterations < 0:
            raise ValueError('iterations/inner_steps must be positive; init_iterations >=0')
        if not np.isfinite([self.lambda_mel, self.rho]).all() or self.lambda_mel <= 0 or self.rho <= 0:
            raise ValueError('lambda_mel and rho must be positive finite')
        if self.init not in ('multiplicative', 'nnls'):
            raise ValueError('init must be multiplicative or nnls')
        if self.trace_every < 1:
            raise ValueError('trace_every must >= 1')


def _project_consistent(z, cfg, n_samples):
    import librosa
    wav = librosa.istft(z, hop_length=cfg.hop_length, win_length=cfg.n_fft,
                        window='hann', center=True, length=n_samples)
    return librosa.stft(wav, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                       win_length=cfg.n_fft, window='hann', center=True,
                       pad_mode='constant')


def _initial_amplitude(mel_power, h, cfg, iters, method):
    """Positive multiplicative power-domain Mel fit (never uses clean wave)."""
    if method == 'nnls':
        from mel_inverse.vocoder import inverse_mel_magnitude
        return np.asarray(inverse_mel_magnitude(mel_power, cfg), dtype=np.float64)
    # H sparse CSR, shape [B,F]. All entries nonnegative.
    ht_m = np.maximum(h.T @ mel_power, 0.)
    row_sq = np.asarray(h.multiply(h).sum(axis=1)).ravel()
    # Simple positive transpose redistribution, calibrated to energy scale.
    p = np.maximum(h.T @ (mel_power / np.maximum(row_sq[:, None], 1e-24)), 0.)
    # Iterative multiplicative nonnegative least-squares in POWER domain.
    for _ in range(iters):
        den = np.maximum(h.T @ (h @ p), 1e-24)
        p *= ht_m / den
        # For frequencies outside Mel bank, stay at zero (not an error).
        p = np.maximum(p, 0.)
    return np.sqrt(p)


def _power_loss(w, e, target, lam, rho, phi):
    r = (e @ (w * w)) - target
    d = w - phi
    return 0.5 * lam * float(np.sum(r * r)) + 0.5 * rho * float(np.sum(d * d))


def _w_prox_power(w0, phi, e, target, lam, rho, steps):
    """Approx. nonnegative W proximal by projected gradient + backtracking.

    For f(W)=lam/2||E W² - M||² + rho/2||W-Phi||²:
      gradient = 2 lam W ⊙ Eᵀ(E W² - M) + rho (W - Phi).
    A backtracking test rejects objective-increasing projected updates.
    """
    w = np.maximum(w0, 0.)
    for _ in range(steps):
        residual = (e @ (w*w)) - target
        et_resid = e.T @ residual
        grad = 2.0 * lam * w * et_resid + rho * (w - phi)
        if not np.isfinite(grad).all():
            raise FloatingPointError('Non-finite W gradient (try smaller lambda or rho)')
        norm_inf = float(np.max(np.abs(et_resid)))
        # Lipschitz-style conservative step, then line-search to ensure descent.
        # The sparse operator norm is bounded by ||E||_F.
        e2 = float(e.multiply(e).sum())
        maxw2 = float(np.max(w*w))
        lips = rho + 2.0*lam*norm_inf + 4.0*lam*e2*maxw2
        step = 1.0 / max(lips, 1e-12)
        old_loss = _power_loss(w, e, target, lam, rho, phi)
        accepted = False
        for _ in range(12):
            trial = np.maximum(w - step*grad, 0.)
            loss = _power_loss(trial, e, target, lam, rho, phi)
            if np.isfinite(loss) and loss <= old_loss + 1e-9*max(1., old_loss):
                w = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return w


def invert_power_mel_joint(logmel, cfg: MelConfig, n_samples: int,
                           options: PowerMelJointConfig | None = None):
    """Take unnormalized ORIGINAL natural-log Mel power [96,T] -> waveform.

    Returns (waveform_float32 [n_samples], diagnostic_dict). No ground-truth
    waveform/phase used; no training. Actual CNN input already denormalized.
    """
    import librosa
    opts = options or PowerMelJointConfig()
    opts.validate()
    if n_samples < 1:
        raise ValueError('n_samples must be >=1')
    mel, clip_fraction = inverse_log(logmel, cfg)  # exact V0.1 inverse log
    t0 = time.perf_counter()
    H = sparse.csr_matrix(make_mel_bank(cfg))
    init_mag = _initial_amplitude(mel, H, cfg, opts.init_iterations, opts.init)
    if not np.isfinite(init_mag).all() or init_mag.shape != (cfg.n_fft//2+1, mel.shape[1]):
        raise RuntimeError('Invalid initialization magnitude')
    # Condition variables to O(1); do NOT change the input Mel or front-end.
    a_scale = max(float(np.sqrt(np.mean(init_mag*init_mag))), 1e-6)
    mel_scale = max(float(np.mean(mel)), 1e-10)
    E = H * (a_scale*a_scale / mel_scale)
    M = mel / mel_scale
    y = init_mag / a_scale
    w = y.copy()
    # Match legacy librosa Griffin-Lim random-angle generator when using
    # random_state=int. The SAME seed does not align magnitude estimates.
    rng = np.random.RandomState(opts.seed)
    phases = np.exp(1j*(2.0*np.pi*rng.random(size=y.shape)))
    z = y * phases
    v = np.zeros_like(z)
    u = np.zeros_like(y)
    trace = []
    for k in range(opts.iterations):
        psi = z + v
        # Zero-bin phase is arbitrary but determinate, and 0/0 is avoided.
        psi_phase = np.divide(psi, np.maximum(np.abs(psi), 1e-12))
        x = ((y + opts.rho*np.abs(psi))/(1+opts.rho)) * psi_phase
        phi = y + u
        w = _w_prox_power(w, phi, E, M, opts.lambda_mel, opts.rho,
                          opts.inner_steps)
        z = _project_consistent(x-v, cfg, n_samples)
        if z.shape != x.shape:
            raise RuntimeError('STFT shape drift; center/window mismatch')
        y = np.maximum((np.abs(x)+opts.rho*(w-u))/(1+opts.rho), 0.)
        v = v + z - x
        u = u + y - w
        if not (np.isfinite(y).all() and np.isfinite(z).all() and np.isfinite(u).all() and np.isfinite(v).all()):
            raise FloatingPointError(f'ADMM diverged at iteration {k+1}; adjust lambda/rho')
        if ((k+1) % opts.trace_every == 0 or k == 0 or (k+1) == opts.iterations):
            consistency = (H @ (np.abs(z*a_scale)**2))
            relative = float(np.linalg.norm(consistency-mel)/max(np.linalg.norm(mel),1e-12))
            trace.append({'iteration':k+1,
                          'relative_mel_power_residual':relative,
                          'relative_admm_primal_X':float(np.linalg.norm(z-x)/max(np.linalg.norm(z),1e-12)),
                          'relative_admm_primal_Y':float(np.linalg.norm(y-w)/max(np.linalg.norm(y),1e-12))})
    # Return waveform from STFT-consistent z; original signal not involved.
    wave = librosa.istft(z*a_scale, hop_length=cfg.hop_length,
                        win_length=cfg.n_fft, window='hann', center=True,
                        length=n_samples).astype(np.float32)
    if wave.shape != (n_samples,) or not np.isfinite(wave).all():
        raise RuntimeError('Invalid output waveform')
    return wave, {
        'algorithm':'power_mel_admm_joint_adaptation',
        'not_published_magnitude_algorithm':True,
        'initialization':opts.init,
        'initialization_is_nnls':opts.init=='nnls',
        'iterations':opts.iterations,
        'lambda_mel':opts.lambda_mel,'rho':opts.rho,
        'inner_steps':opts.inner_steps,'init_iterations':opts.init_iterations,
        'clip_fraction':clip_fraction,'time_s':time.perf_counter()-t0,
        'final_mel_power_residual':trace[-1]['relative_mel_power_residual'],
        'trace':trace,
    }
