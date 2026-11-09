"""Single official synthesis API: raw natural-log power Mel -> waveform.
Default: CUDA when available, otherwise NumPy CPU, same objective/config.
"""
from mel_inverse.power_joint_cpu import PowerMelJointConfig, invert_power_mel_joint
from mel_inverse.power_joint_cuda import invert_power_mel_joint_torch


def synthesize(logmel, cfg, n_samples, *, iterations=(32, 64), device='auto',
               precision='float32', lambda_mel=1.0, rho=1.0,
               inner_steps=5, init_iterations=40, init='multiplicative', seed=42):
    import torch
    points=sorted(set(int(x) for x in iterations))
    if not points or points[0]<1:raise ValueError('iterations must be positive')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    opts=PowerMelJointConfig(iterations=points[-1],lambda_mel=lambda_mel,rho=rho,
            inner_steps=inner_steps,init_iterations=init_iterations,init=init,seed=seed)
    if device=='cuda' or device=='torch-cpu':
        return invert_power_mel_joint_torch(logmel,cfg,n_samples,opts,
                 device='cpu' if device=='torch-cpu' else 'cuda',precision=precision,snapshots=points)
    if device=='cpu':
        # CPU NumPy uses the same objective but computes each snapshot separately.
        from dataclasses import replace
        return {n:invert_power_mel_joint(logmel,cfg,n_samples,replace(opts,iterations=n)) for n in points}
    raise ValueError('device must be auto, cuda, cpu or torch-cpu')
