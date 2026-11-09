"""Reproducible, channel-agnostic SNR sampling for train batches.

This changes *only* noise severity, not channel type or the V0.1 architecture.
One SNR is sampled per batch; all samples in that batch experience the same
nominal SNR, but their Gaussian noise/fading draws are independent.

The sampler has its own Python RNG: it does not consume Torch's RNG (which is
needed for neural initialization, AWGN noise, and Rayleigh fading).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class MultiSNRSampler:
    min_db: float = 0.0
    max_db: float = 20.0
    seed: int = 42
    mode: str = 'uniform'
    grid: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0)
    _rng: random.Random = field(init=False, repr=False)
    _samples: list[float] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self):
        if not all(map(math.isfinite, (self.min_db, self.max_db))):
            raise ValueError('SNR endpoints must be finite')
        if self.min_db >= self.max_db:
            raise ValueError('--snr-min must be smaller than --snr-max')
        if self.mode not in ('uniform', 'grid'):
            raise ValueError('SNR mode must be uniform or grid')
        if self.mode == 'grid' and (
            not self.grid or
            any(not math.isfinite(v) or v < self.min_db or v > self.max_db for v in self.grid)
        ):
            raise ValueError('Grid SNRs must be nonempty, finite and inside [min_db, max_db]')
        self._rng = random.Random(self.seed)

    def sample(self) -> float:
        if self.mode == 'uniform':
            db = self._rng.uniform(self.min_db, self.max_db)
        else:
            db = float(self._rng.choice(self.grid))
        self._samples.append(db)
        return db

    def summary(self) -> dict[str, float | int]:
        if not self._samples:
            raise ValueError('No SNR was sampled')
        return {
            'train_snr_batches': len(self._samples),
            'train_snr_mean_db': sum(self._samples) / len(self._samples),
            'train_snr_min_db': min(self._samples),
            'train_snr_max_db': max(self._samples),
        }


def parse_snr_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in value.split(',') if token.strip())
    except ValueError as exc:
        raise ValueError('SNR grid must contain numeric values, comma-separated') from exc
    if not values or not all(map(math.isfinite, values)) or len(set(values)) != len(values):
        raise ValueError('SNR grid must be nonempty, finite and without duplicates')
    return values
