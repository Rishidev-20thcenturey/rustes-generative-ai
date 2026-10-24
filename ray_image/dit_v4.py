"""RAY-IMAGE DiT v4 (100M @ 512 pivot). Copied from dit_v3.py.

Model for 16x64x64 latents with patch=4 -> a 16x16 token grid (256 tokens):
  - dim=640, depth=14, heads=10, patch=4
  - RMSNorm + SwiGLU FFN
  - 2D axial RoPE self-attention (row and column frequencies, see below)
  - gated text cross-attention enabled by default

2D axial RoPE: each attention head's channels are split in half. The first half
is rotated by the token's ROW index (y), the second half by its COLUMN index (x).
The 1D RoPE used in v3 (one position index over the flattened raster) is replaced,
so a token's positional encoding no longer depends on a single flat index that
wraps from the end of one row to the start of the next.

Patch embedding: Conv2d(16, 640, kernel=4, stride=4). Each 4x4 patch has
16 channels x 4 x 4 = 256 input values, mapped to dim=640.

Expected text context: Qwen3-4B hidden states with dim=2560.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

LATENT_CHANNELS = 16
LATENT_SIZE = 64
MODEL_DIM = 640
MODEL_DEPTH = 14
MODEL_HEADS = 10
MODEL_PATCH = 4
TOKEN_GRID = LATENT_SIZE // MODEL_PATCH  # 16 -> 256 tokens


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(dtype=self.weight.dtype) * self.weight


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if self.dim % 2 != 0:
            raise ValueError("timestep embedding dim must be even")
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        x = t.float()[:, None] * freq[None, :]
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.mlp(emb)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _axial_angles(positions: torch.Tensor, axis_dim: int) -> torch.Tensor:
    """Rotary angles for one axis: [N, axis_dim] (pairs of frequencies, duplicated)."""
    pairs = axis_dim // 2
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(pairs, device=positions.device, dtype=torch.float32)
        / max(pairs, 1)
    )
    angles = positions[:, None] * freq[None, :]
    return torch.cat((angles, angles), dim=-1)


def apply_rope_2d(q: torch.Tensor, k: torch.Tensor, grid_h: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """2D axial RoPE for a row-major grid_h x grid_w token layout.

    q/k: [B, heads, N, head_dim] with N = grid_h * grid_w and token n at
    row = n // grid_w, column = n % grid_w.
    The first head_dim/2 channels are rotated by the row index; the second half
    by the column index. Each half uses its own rotate-half, so row and column
    frequencies never mix.
    """
    head_dim = q.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError("2D axial RoPE requires head_dim divisible by 4")
    n = grid_h * grid_w
    if q.shape[-2] != n or k.shape[-2] != n:
        raise ValueError(f"token count {q.shape[-2]} does not match grid {grid_h}x{grid_w}")
    device = q.device
    rows = torch.arange(grid_h, device=device, dtype=torch.float32).repeat_interleave(grid_w)
    cols = torch.arange(grid_w, device=device, dtype=torch.float32).repeat(grid_h)
    axis_dim = head_dim // 2
    cos = torch.cat((_axial_angles(rows, axis_dim).cos(), _axial_angles(cols, axis_dim).cos()), dim=-1)
    sin = torch.cat((_axial_angles(rows, axis_dim).sin(), _axial_angles(cols, axis_dim).sin()), dim=-1)
    cos = cos.to(dtype=q.dtype)[None, None, :, :]
    sin = sin.to(dtype=q.dtype)[None, None, :, :]

    def rotate(t: torch.Tensor) -> torch.Tensor:
        t_row, t_col = t.chunk(2, dim=-1)
        return torch.cat((_rotate_half(t_row), _rotate_half(t_col)), dim=-1)

    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(round((4 * dim * 2 / 3) / 64) * 64)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SelfAttentionRoPE(nn.Module):
    """Multi-head self-attention with 2D axial RoPE over a grid x grid token layout."""

    def __init__(self, dim: int, heads: int, grid: int = TOKEN_GRID):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.grid = grid
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope_2d(q, k, self.grid, self.grid)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Token-to-text cross-attention with optional learned output gate."""

    def __init__(self, dim: int, heads: int, context_dim: int, cross_gate: bool = True):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.cross_gate = nn.Parameter(torch.ones(1)) if cross_gate else None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        l = context.shape[1]
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if context_mask is not None:
            # v2's mask convention is True=padding/ignore. SDPA's bool mask is
            # True=keep, so invert the same mask before passing it to attention.
            attn_mask = (~context_mask).to(device=q.device)[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.proj(out)
        if self.cross_gate is not None:
            out = self.cross_gate * out
        return out


class RAYDiTBlock_v4(nn.Module):
    """Pre-norm DiT block with RMSNorm, 2D axial RoPE self-attention and gated cross-attention."""

    def __init__(self, dim: int, heads: int, context_dim: int, cross_gate: bool = True, grid: int = TOKEN_GRID):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttentionRoPE(dim, heads, grid=grid)
        self.norm_cross = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, heads, context_dim, cross_gate=cross_gate)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)
        self.cond = nn.Linear(dim, dim * 2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        x = x + self.self_attn(h)
        x = x + self.cross_attn(
            self.norm_cross(x),
            text,
            context_mask=text_mask,
        )
        x = x + self.ffn(self.norm2(x))
        return x


class RAYDiT_v4(nn.Module):
    """100M-class RAY-IMAGE DiT v4 for 16x64x64 latents with patch=4 (256 tokens)."""

    def __init__(
        self,
        latent_channels: int = LATENT_CHANNELS,
        latent_size: int = LATENT_SIZE,
        dim: int = MODEL_DIM,
        depth: int = MODEL_DEPTH,
        heads: int = MODEL_HEADS,
        patch: int = MODEL_PATCH,
        cond_dim: int = 2560,
        cross_gate: bool = True,
    ):
        super().__init__()
        if latent_size % patch != 0:
            raise ValueError("latent_size must be divisible by patch")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.patch = patch
        self.cond_dim = cond_dim
        self.grid = latent_size // patch

        self.in_proj = nn.Conv2d(latent_channels, dim, kernel_size=patch, stride=patch)
        self.time = TimestepEmbedding(dim)
        self.text_proj = nn.Linear(cond_dim, dim)
        self.text_ctx_proj = nn.Linear(cond_dim, dim) if cond_dim != dim else nn.Identity()
        self.blocks = nn.ModuleList(
            [RAYDiTBlock_v4(dim, heads, context_dim=dim, cross_gate=cross_gate, grid=self.grid) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, patch * patch * latent_channels)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"expected latent tensor [B,C,H,W], got {tuple(z.shape)}")
        b, c, h, w = z.shape
        if (c, h, w) != (self.latent_channels, self.latent_size, self.latent_size):
            raise ValueError(
                f"expected latent [{self.latent_channels},{self.latent_size},{self.latent_size}], got [{c},{h},{w}]"
            )
        if text.ndim != 3 or text.shape[0] != b or text.shape[2] != self.cond_dim:
            raise ValueError(f"expected text [B,L,{self.cond_dim}], got {tuple(text.shape)}")
        if t.ndim != 1 or t.shape[0] != b:
            raise ValueError(f"expected timestep [B], got {tuple(t.shape)}")
        if text_mask is not None and text_mask.shape != text.shape[:2]:
            raise ValueError(f"expected text_mask {tuple(text.shape[:2])}, got {tuple(text_mask.shape)}")

        x = self.in_proj(z)
        ph, pw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        text = text.to(dtype=x.dtype)
        if text_mask is None:
            pooled = text.mean(dim=1)
        else:
            valid = (~text_mask).to(dtype=text.dtype).unsqueeze(-1)
            pooled = (text * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        cond = self.time(t).to(dtype=x.dtype) + self.text_proj(pooled)
        text_ctx = self.text_ctx_proj(text)
        for block in self.blocks:
            x = block(x, cond, text_ctx, text_mask=text_mask)

        x = self.out(self.norm(x))
        x = x.view(b, ph, pw, self.patch, self.patch, self.latent_channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.latent_channels, h, w)
