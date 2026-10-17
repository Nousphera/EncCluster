"""ConvMixer-256/8: official CIFAR configuration (patch 2, kernel 5).

Adapted from locuslab/convmixer; MIT notice in THIRD_PARTY_LICENSES.
Modified to expose pooled features for the common classifier wrapper.
"""

import torch


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer256_8(torch.nn.Sequential):
    embeddings_dim = 256

    def __init__(self):
        nn = torch.nn
        super().__init__(
            nn.Conv2d(3, 256, kernel_size=2, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(256),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(256, 256, 5, groups=256, padding="same"),
                            nn.GELU(),
                            nn.BatchNorm2d(256),
                        )
                    ),
                    nn.Conv2d(256, 256, 1),
                    nn.GELU(),
                    nn.BatchNorm2d(256),
                )
                for _ in range(8)
            ],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
