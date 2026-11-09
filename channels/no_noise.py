"""Control channel: original power normalization without channel noise."""
import torch
from torch import nn


class NoNoiseChannel(nn.Module):
    def forward(self, latent):
        dims = tuple(range(1, latent.ndim))
        return latent / torch.sqrt(latent.square().mean(dim=dims, keepdim=True) + 1e-8)
