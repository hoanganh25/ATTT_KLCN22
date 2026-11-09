from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MelConfig:
    # Lock all preprocessing settings for reproducible V0.1.
    sr: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 96
    fmin: float = 0.0
    fmax: float | None = None   # None: sr/2
    eps: float = 1e-8
    random_seed: int = 0

    def __post_init__(self):
        if self.sr <= 0 or self.n_fft <= 0 or self.n_fft % 2:
            raise ValueError("sr must be positive and n_fft must be positive and even")
        if not (0 < self.hop_length <= self.n_fft):
            raise ValueError("0 < hop_length <= n_fft required")
        if self.n_mels <= 0 or self.eps <= 0:
            raise ValueError("n_mels and eps must be positive")
        upper = self.sr / 2 if self.fmax is None else self.fmax
        if self.fmin < 0 or not (self.fmin < upper <= self.sr / 2):
            raise ValueError("fmin/fmax must lie in 0..Nyquist")

    def as_dict(self) -> dict:
        return asdict(self)
