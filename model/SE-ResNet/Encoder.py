"""PyTorch adaptation of the SE-ResNet *building blocks* in official DeepSC-S.

Official reference: https://github.com/Zhenzi-Weng/DeepSC-S/blob/main/models.py
This project adapts waveform frames -> normalized log-Mel [B,1,96,96] and
keeps the original V0.1 channel interface [B,8,24,24]. This is NOT a
bitwise reproduction of the original TensorFlow waveform DeepSC-S system.

The default 'repo' profile reproduces official-repo hyperparameters:
    5 SE blocks per side, 4 parallel 5x5 Conv+BN+ReLU branches of 128 filters,
    concat -> 1x1 Conv+BN -> channel SE (reduction=4) -> residual add -> ReLU.
The 'paper' profile is an optional lighter approximation of the *published
telephone-system description*: six blocks, two branches with 16 filters,
transition to 32. Source paper and public repo are not identical.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SEConfig:
    profile: str = 'repo'
    latent_channels: int = 8
    blocks: int = 0     # 0 = source-profile default
    checkpoint_blocks: bool = False

    def resolve(self):
        if self.profile not in {'repo', 'paper'}:
            raise ValueError('profile must be repo or paper')
        if self.latent_channels < 2 or self.latent_channels % 2:
            raise ValueError('latent_channels must be positive and even (Rayleigh I/Q)')
        cfg = {'repo': (128, 4, 128, 5), 'paper': (32, 2, 16, 6)}[self.profile]
        features, cardinality, branch_width, nblocks = cfg
        nblocks = self.blocks if self.blocks else nblocks
        if nblocks < 1:
            raise ValueError('blocks must be >= 1')
        return features, cardinality, branch_width, nblocks


def _bn(ch):
    # TF Keras BatchNormalization defaults: epsilon=1e-3, momentum=0.99;
    # PyTorch uses the complement, i.e. momentum=0.01 for running statistics.
    return nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01)


def _conv_bn(cin, cout, stride=1, kernel=5, activation=True):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=kernel, stride=stride,
                  padding=kernel // 2, bias=False),
        _bn(cout),
        *( [nn.ReLU(inplace=False)] if activation else []),
    )


def _deconv_bn(cin, cout):
    # ConvTranspose2d(5,stride2,pad2,output_pad1) maps 24->48->96.
    # Border convention approximates TF 'same' transpose convolution.
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, kernel_size=5, stride=2, padding=2,
                           output_padding=1, bias=False),
        _bn(cout), nn.ReLU(inplace=False),
    )


class OfficialSEBlock(nn.Module):
    """Official DeepSC-S split/transition/SE residual topology (no post BN)."""
    def __init__(self, features: int, cardinality: int, branch_width: int,
                 reduction: int = 4):
        super().__init__()
        if features % reduction:
            raise ValueError('SE features must divide reduction')
        self.split = nn.ModuleList([
            _conv_bn(features, branch_width, kernel=5)
            for _ in range(cardinality)
        ])
        self.transition = _conv_bn(cardinality * branch_width, features,
                                    kernel=1, activation=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, features // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(features // reduction, features, bias=False),
            nn.Sigmoid(),
        )
        self.out_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        s = torch.cat([branch(x) for branch in self.split], dim=1)
        t = self.transition(s)
        weights = self.excite(self.pool(t)).view(x.size(0), x.size(1), 1, 1)
        return self.out_relu(x + t * weights)


class _SEStack(nn.Module):
    def __init__(self, features, cardinality, branch_width, count, use_checkpoint):
        super().__init__()
        self.blocks = nn.ModuleList([
            OfficialSEBlock(features, cardinality, branch_width)
            for _ in range(count)
        ])
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class Encoder(nn.Module):
    """normalized log-Mel [B,1,96,96] -> V0.1-compatible latent [B,C,24,24]."""
    def __init__(self, config: SEConfig | None = None):
        super().__init__()
        self.config = config or SEConfig()
        features, card, width, blocks = self.config.resolve()
        stem_width = 32  # official semantic encoder first Conv=32
        self.stem = nn.Sequential(
            _conv_bn(1, stem_width, stride=2),
            _conv_bn(stem_width, features, stride=2),
        )
        self.semantic = _SEStack(features, card, width, blocks,
                                 self.config.checkpoint_blocks)
        # Only necessary V0.1 adaptation: retain channel symbol count=8 instead
        # of official code's 128; no change to V0.1 AWGN/Rayleigh equations.
        self.channel_encoder = _conv_bn(features, self.config.latent_channels,
                                        kernel=5, activation=False)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, 96, 96):
            raise ValueError(f'Expect [B,1,96,96], got {tuple(x.shape)}')
        z = self.channel_encoder(self.semantic(self.stem(x)))
        return z


