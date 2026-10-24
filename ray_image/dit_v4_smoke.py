"""CPU smoke test for DiT v4 (100M @ 512 pivot): 2D axial RoPE + shape math.

Checks:
  (a) 2D RoPE is the identity at position (0, 0) and is a pure rotation (norm kept);
  (b) relative-position invariance on the 2D grid: a q/k pair at offset (dy, dx)
      scores the same wherever it sits on the grid;
  (c) a flat 1D RoPE would score (row 0, col 15) vs (row 1, col 0) as relative
      offset 1 (the seam); 2D axial scores them by their real (dy, dx) = (1, -15);
  (d) RAYDiT_v4 default config: 16x16 = 256 tokens, patch embed 16*4*4=256 -> 640,
      output shape [1, 16, 64, 64] on a full-size forward.
"""

import torch

from .dit_v4 import LATENT_CHANNELS, LATENT_SIZE, MODEL_DIM, MODEL_PATCH, TOKEN_GRID, RAYDiT_v4, apply_rope_2d


def _token(row: int, col: int, grid: int) -> int:
    return row * grid + col


def _score(q_vec: torch.Tensor, k_vec: torch.Tensor, grid: int, p: int, p2: int) -> float:
    """Rotated dot product of one q at token p with one k at token p2 (single head)."""
    n = grid * grid
    q = q_vec.view(1, 1, 1, -1).expand(1, 1, n, -1).clone()
    k = k_vec.view(1, 1, 1, -1).expand(1, 1, n, -1).clone()
    q_rot, k_rot = apply_rope_2d(q, k, grid, grid)
    return float((q_rot[0, 0, p] * k_rot[0, 0, p2]).sum())


def check_rope_identity_and_norm() -> None:
    torch.manual_seed(0)
    grid, head_dim = 16, 64
    q = torch.randn(1, 2, grid * grid, head_dim)
    k = torch.randn(1, 2, grid * grid, head_dim)
    q_rot, k_rot = apply_rope_2d(q, k, grid, grid)
    assert torch.allclose(q_rot[..., 0, :], q[..., 0, :], atol=1e-6), "position (0,0) must be unrotated"
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4), "RoPE must preserve norms"
    print("[PASS] 2D RoPE: identity at (0,0), norm-preserving")


def check_relative_invariance() -> None:
    torch.manual_seed(1)
    grid, head_dim = 16, 64
    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)
    # Same relative offset (dy, dx) = (2, 3) at two different absolute positions.
    a = _score(q_vec, k_vec, grid, _token(0, 0, grid), _token(2, 3, grid))
    b = _score(q_vec, k_vec, grid, _token(5, 7, grid), _token(7, 10, grid))
    assert abs(a - b) < 1e-3, f"relative invariance failed: {a} vs {b}"
    # Different offset must (generically) score differently.
    c = _score(q_vec, k_vec, grid, _token(0, 0, grid), _token(2, 4, grid))
    assert abs(a - c) > 1e-3, "different relative offsets should not collide"
    print(f"[PASS] 2D RoPE relative invariance: (2,3) at two positions {a:.5f} == {b:.5f}")


def check_seam_pair_uses_true_offset() -> None:
    torch.manual_seed(2)
    grid, head_dim = 16, 64
    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)
    # Flat-index 1D RoPE sees index 15 vs 16 (offset 1) for (0,15)->(1,0).
    # 2D axial must see offset (dy, dx) = (1, -15), which is not the same as (0, 1).
    seam = _score(q_vec, k_vec, grid, _token(0, 15, grid), _token(1, 0, grid))
    flat_neighbour = _score(q_vec, k_vec, grid, _token(0, 0, grid), _token(0, 1, grid))
    assert abs(seam - flat_neighbour) > 1e-3, "seam pair must not look like a horizontal neighbour"
    print(f"[PASS] seam pair (0,15)->(1,0) scored {seam:.5f}, unlike horizontal neighbour {flat_neighbour:.5f}")


def check_shape_math_and_forward() -> None:
    model = RAYDiT_v4()
    assert model.grid == TOKEN_GRID == 16, model.grid
    weight = model.in_proj.weight
    assert tuple(weight.shape) == (MODEL_DIM, LATENT_CHANNELS, MODEL_PATCH, MODEL_PATCH), tuple(weight.shape)
    assert LATENT_CHANNELS * MODEL_PATCH * MODEL_PATCH == 256
    n_params = sum(p.numel() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
        t = torch.rand(1)
        text = torch.randn(1, 8, 2560)
        out = model(z, t, text, text_mask=torch.zeros(1, 8, dtype=torch.bool))
    assert tuple(out.shape) == (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), tuple(out.shape)
    assert torch.isfinite(out).all().item()
    print(f"[PASS] RAYDiT_v4 forward: out={tuple(out.shape)} params={n_params:,} ({n_params / 1e6:.1f}M)")


def main() -> None:
    check_rope_identity_and_norm()
    check_relative_invariance()
    check_seam_pair_uses_true_offset()
    check_shape_math_and_forward()
    print("[PASS] dit_v4 smoke test")


if __name__ == "__main__":
    main()
