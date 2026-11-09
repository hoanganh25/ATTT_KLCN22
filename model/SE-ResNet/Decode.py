"""DeepSC-S-inspired SE-ResNet decoder for V0.1 normalized log-Mel.

This lives in model/SE-ResNet/Decode.py and uses shared SE blocks
from Encoder.py to avoid copy-pasting and ensure exact architecture.
"""
from __future__ import annotations
import torch
from torch import nn
from .Encoder import SEConfig, _conv_bn, _deconv_bn, _SEStack


class Decoder(nn.Module):
    """V0.1-compatible latent [B,C,24,24] -> normalized log-Mel [B,1,96,96]."""
    def __init__(self, config: SEConfig | None = None):
        super().__init__()
        self.config = config or SEConfig()
        features, card, width, blocks = self.config.resolve()
        self.channel_decoder = _conv_bn(self.config.latent_channels, features,
                                        kernel=5, activation=True)
        self.semantic = _SEStack(features, card, width, blocks,
                                 self.config.checkpoint_blocks)
        up_channels = features if self.config.profile == 'repo' else 32
        self.up = nn.Sequential(
            _deconv_bn(features, up_channels),
            _deconv_bn(up_channels, 32),
        )
        self.output = nn.Conv2d(32, 1, kernel_size=1, bias=False)

    def forward(self, z):
        expected = (self.config.latent_channels, 24, 24)
        if z.ndim != 4 or z.shape[1:] != expected:
            raise ValueError(f'Expect [B,{expected[0]},24,24], got {tuple(z.shape)}')
        return self.output(self.up(self.semantic(self.channel_decoder(z))))


