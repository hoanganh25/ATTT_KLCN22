
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                8, 32, kernel_size=3,
                stride=1, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 16, kernel_size=4,
                stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                16, 1, kernel_size=4,
                stride=2, padding=1
            )
        )

    def forward(self, z):
        return self.network(z)


if __name__ == "__main__":
    model = Decoder()

    z = torch.randn(2, 8, 24, 24)
    reconstructed = model(z)

    print("Encoded:", z.shape)
    print("Reconstructed:", reconstructed.shape)
