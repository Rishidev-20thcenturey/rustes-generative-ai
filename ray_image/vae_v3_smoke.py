"""CPU smoke test for VAE v3 (512 px, 64x64 latents).

Checks the explicit contract: [B,3,512,512] -> latent [B,16,64,64] -> [B,3,512,512],
with output in [0, 1]; and that wrong resolutions are refused instead of silently
producing a wrongly sized latent.
"""

import torch

from .vae_v3 import IMAGE_SIZE, LATENT_SIZE, RAYVAE_v3


def main() -> None:
    torch.manual_seed(0)
    vae = RAYVAE_v3(latent_channels=16, base_channels=64).eval()
    with torch.no_grad():
        x = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        recon, z, mean, logvar = vae(x)
    assert tuple(z.shape) == (1, 16, LATENT_SIZE, LATENT_SIZE), tuple(z.shape)
    assert tuple(mean.shape) == tuple(z.shape) and tuple(logvar.shape) == tuple(z.shape)
    assert tuple(recon.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE), tuple(recon.shape)
    assert float(recon.min()) >= 0.0 and float(recon.max()) <= 1.0
    print(f"[PASS] VAE v3 forward: image {tuple(x.shape)} -> latent {tuple(z.shape)} -> recon {tuple(recon.shape)}")

    for bad in (256, 128):
        try:
            with torch.no_grad():
                vae.encode(torch.rand(1, 3, bad, bad))
        except ValueError:
            continue
        raise AssertionError(f"encode accepted a {bad}px image")
    try:
        with torch.no_grad():
            vae.decode(torch.randn(1, 16, 32, 32))
    except ValueError:
        pass
    else:
        raise AssertionError("decode accepted a 32x32 latent")
    print("[PASS] VAE v3 refuses 256px images and 32x32 latents")
    print("[PASS] vae_v3 smoke test")


if __name__ == "__main__":
    main()
