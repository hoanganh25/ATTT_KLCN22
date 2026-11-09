
import torch
from torch import nn


class AWGNChannel(nn.Module):
    def __init__(self, snr_db=10.0):
        super().__init__()
        self.snr_db = float(snr_db)

    def forward(self, z, snr_db=None):
        if snr_db is None:
            snr_db = self.snr_db

        dims = tuple(range(1, z.ndim))
        power = z.pow(2).mean(dim=dims, keepdim=True)

        z_normalized = z / torch.sqrt(power + 1e-8)

        noise_std = 10.0 ** (-float(snr_db) / 20.0)

        noise = torch.randn_like(z_normalized) * noise_std

        return z_normalized + noise


if __name__ == "__main__":
    channel = AWGNChannel(snr_db=10)

    z = torch.randn(2, 8, 24, 24)
    y = channel(z)

    print("Before channel:", z.shape)
    print("After channel:", y.shape)
