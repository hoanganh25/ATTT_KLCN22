
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3,
                stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16, 32, kernel_size=3,
                stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 8, kernel_size=3,
                stride=1, padding=1
            )
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = Encoder()

    x = torch.rand(2, 1, 96, 96)
    z = model(x)

    print("Input:", x.shape)
    print("Encoded:", z.shape)
