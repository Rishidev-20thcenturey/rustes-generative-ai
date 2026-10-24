"""RAY-IMAGE VAE v3 for 512x512 training (100M @ 512 pivot).

Copied from vae_v2.py (which stays untouched for the shipped 256px model).

The explicit contract for this model is:
    image:  [B, 3, 512, 512]
    latent: [B, 16, 64, 64]

Three stride-2 encoder transitions (512 -> 256 -> 128 -> 64) and three
matching decoder upsamples (64 -> 128 -> 256 -> 512) keep the same five-block
layout as v2. latent_channels=16 and base_channels=64 are unchanged.
"""

import torch
from torch import nn

IMAGE_SIZE = 512
LATENT_SIZE = 64


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool):
        super().__init__()
        self.up = (
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
            if upsample else nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.block = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(self.up(x))


class RAYVAE_v3(nn.Module):
    """Wider 512px VAE with 16-channel 64x64 latents."""

    def __init__(self, latent_channels: int = 16, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # 512 -> 256 -> 128 -> 64, then two refinement blocks at 64x64.
        self.enc1 = ConvBlock(3, c, stride=2)
        self.enc2 = ConvBlock(c, c * 2, stride=2)
        self.enc3 = ConvBlock(c * 2, c * 4, stride=2)
        self.enc4 = ConvBlock(c * 4, c * 4, stride=1)
        self.enc5 = ConvBlock(c * 4, c * 4, stride=1)
        self.to_stats = nn.Conv2d(c * 4, latent_channels * 2, 3, padding=1)

        # 64 -> 128 -> 256 -> 512, then two refinement blocks at 512x512.
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, c * 4, 3, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.SiLU(inplace=True),
        )
        self.dec1 = UpBlock(c * 4, c * 4, upsample=False)
        self.dec2 = UpBlock(c * 4, c * 2, upsample=True)
        self.dec3 = UpBlock(c * 2, c, upsample=True)
        self.dec4 = UpBlock(c, c // 2, upsample=True)
        self.dec5 = ConvBlock(c // 2, c // 2, stride=1)
        self.out = nn.Sequential(
            nn.Conv2d(c // 2, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.latent_channels = latent_channels
        self.base_channels = base_channels

    def encode(self, x):
        if tuple(x.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"RAYVAE_v3 expects {IMAGE_SIZE}x{IMAGE_SIZE} images, got {tuple(x.shape[-2:])}")
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        stats = self.to_stats(h)
        mean, logvar = stats.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar.clamp(-30, 20))
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar

    def decode(self, z):
        if tuple(z.shape[-2:]) != (LATENT_SIZE, LATENT_SIZE):
            raise ValueError(f"RAYVAE_v3 expects {LATENT_SIZE}x{LATENT_SIZE} latents, got {tuple(z.shape[-2:])}")
        h = self.from_latent(z)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        h = self.dec5(h)
        return self.out(h)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, z, mean, logvar
