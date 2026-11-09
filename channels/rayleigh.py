"""Complex baseband quasi-static Rayleigh flat-fading channel for real CNN latent.

Pair adjacent latent channels as I/Q; with 8 real channels => 4 complex streams.
Normalize the *real-valued* latent just as AWGNChannel, so E[z_I²]≈E[z_Q²]≈1
in aggregate. Each patch experiences ONE independent h~CN(0,1).

  r = h*s + w, h~CN(0,1), Re(w), Im(w)~N(0, 10^(-SNR/10))
  r_eq = conj(h)/(abs(h)^2 + sigma^2) * r  (perfect-CSI LMMSE)

Nominal SNR refers to average pre-fade power; instantaneous SNR varies with |h|².
Perfect CSI is an *optimistic assumption*, not channel estimation or physical RF.
MMSE equalization has no learnable parameters and occurs before CNN Decoder.
"""
import torch
from torch import nn


class RayleighChannel(nn.Module):
    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = float(snr_db)

    def forward(self, latent: torch.Tensor, snr_db=None):
        if latent.ndim != 4 or latent.shape[1] % 2:
            raise ValueError('Rayleigh needs float [B, EVEN_channels, H, W] for I/Q pairing')
        if not latent.dtype.is_floating_point:
            raise TypeError('Latent tensor must be floating point')
        snr = self.snr_db if snr_db is None else float(snr_db)
        sigma = 10.0 ** (-snr / 20.0)
        dims = tuple(range(1, latent.ndim))
        base = latent.float()  # complex64, also safe under CUDA autocast / float16 CNN activations
        normed = base / torch.sqrt(base.square().mean(dim=dims, keepdim=True) + 1e-8)
        # z order is [I0,Q0,I1,Q1,...] along channel dim
        s = torch.complex(normed[:, 0::2], normed[:, 1::2])
        b = latent.shape[0]
        h = (torch.randn((b, 1, 1, 1), device=latent.device, dtype=base.dtype)
             + 1j * torch.randn((b, 1, 1, 1), device=latent.device, dtype=base.dtype)) / (2 ** .5)
        w = sigma * (torch.randn_like(s.real) + 1j * torch.randn_like(s.real))
        received = h * s + w
        # For unit mean power per real component (Es=2), noise has E|w|²=2*sigma².
        # Coherent complex MMSE factor = h* /( |h|² + sigma² ).
        eq = received * h.conj() / (h.abs().square() + sigma * sigma)
        recovered = torch.stack((eq.real, eq.imag), dim=2).flatten(1, 2)
        if recovered.shape != latent.shape or not torch.isfinite(recovered).all():
            raise RuntimeError('Rayleigh equalization produced invalid real latent')
        return recovered
