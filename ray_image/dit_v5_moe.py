"""DiT v5: sparse Mixture-of-Experts DiT for 16x64x64 latents (RAY-IMAGE).

Built from dit_v4.py. Unchanged from v4 and imported from it: RMSNorm, timestep
embedding, 2D axial RoPE self-attention, gated cross-attention, SwiGLU, patch
embed, and the output head. The only change is the feed-forward of each block.
A dense SwiGLU becomes MoELayer: one shared SwiGLU expert that always runs, plus
num_experts routed SwiGLU experts of which each token uses top_k.

Warm start: warm_start_from_v4() copies a v4 block's FFN (gate/up/down) into every
routed expert and every shared expert. The hidden size is 1728 in both models, so
the copy is exact. The MoE output is not identical to the dense FFN at init (see
MoELayer.forward notes).

forward() returns (prediction, load_balancing_loss, router_z_loss). The two losses
are averaged over layers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .dit_v4 import (
    LATENT_CHANNELS,
    LATENT_SIZE,
    MODEL_DEPTH,
    MODEL_DIM,
    MODEL_HEADS,
    MODEL_PATCH,
    TOKEN_GRID,
    CrossAttention,
    RMSNorm,
    SelfAttentionRoPE,
    SwiGLU,
    TimestepEmbedding,
)

MOE_NUM_EXPERTS = 8
MOE_TOP_K = 2
MOE_EXPERT_HIDDEN = 1728
MOE_NUM_SHARED = 1
MOE_CAPACITY_FACTOR = 1.25
MOE_ROUTER_JITTER = 0.01


class MoELayer(nn.Module):
    """Shared expert(s) plus top-k routed SwiGLU experts with a per-expert capacity.

    Routing:
      * logits = router(x) in fp32. Training adds uniform noise in [-jitter, +jitter].
      * torch.topk picks top_k logits per token; softmax over those gives the weights.
      * Capacity per expert = ceil(capacity_factor * tokens * top_k / num_experts).
        Assignments are handled first-choice-first (all tokens' top-1 before any top-2).
        Assignments past capacity are dropped. A dropped routed assignment contributes 0,
        so the token still gets the shared expert and the residual path.

    Auxiliary losses (fp32, returned to the caller):
      * Switch load-balancing loss: E * sum_i f_i * P_i, where f_i is the fraction of
        tokens whose top-1 expert is i and P_i is the mean router probability for i.
      * Router z-loss: mean over tokens of logsumexp(logits)^2.

    Notes:
      * last_dropped_fraction holds the fraction of routed assignments dropped in the
        most recent forward pass. It is for logging only.
      * The top_k weights are a softmax over the selected logits, so they sum to 1.
        With the warm start (down-projections scaled by WARM_START_DOWN_SCALE = 0.5),
        the routed part is 0.5 * FFN(x) and the shared expert adds 0.5 * FFN(x). So at
        init the MoE output equals FFN(x), whatever the router, unless assignments are
        dropped. Dropped assignments reduce it.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = MOE_NUM_EXPERTS,
        top_k: int = MOE_TOP_K,
        expert_hidden_dim: int = MOE_EXPERT_HIDDEN,
        num_shared_experts: int = MOE_NUM_SHARED,
        capacity_factor: float = MOE_CAPACITY_FACTOR,
        router_jitter_noise: float = MOE_ROUTER_JITTER,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, num_experts={num_experts}], got {top_k}")
        if num_shared_experts < 0:
            raise ValueError("num_shared_experts must be >= 0")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be > 0")
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router_jitter_noise = router_jitter_noise
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([SwiGLU(dim, expert_hidden_dim) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList(
            [SwiGLU(dim, expert_hidden_dim) for _ in range(num_shared_experts)]
        )
        self.last_dropped_fraction = 0.0
        # Off by default. When on (set by RAYDiT_v5.forward(capture=True)), forward() keeps the
        # router probabilities, the top-k expert indices and the MoE output for the regularisers.
        self.record_routing = False
        self.last_probs = None
        self.last_top_idx = None
        self.last_out = None

    def forward(self, x: torch.Tensor):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        n = flat.shape[0]
        num_e, k = self.num_experts, self.top_k

        # Routing, auxiliary losses: fp32 regardless of the autocast context.
        with torch.autocast(device_type=flat.device.type, enabled=False):
            logits = self.router(flat.float())
            if self.training and self.router_jitter_noise > 0:
                logits = logits + (torch.rand_like(logits) * 2.0 - 1.0) * self.router_jitter_noise
            probs = torch.softmax(logits, dim=-1)
            top_logits, top_idx = torch.topk(logits, k, dim=-1)
            top_w = torch.softmax(top_logits, dim=-1)
            router_z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
            dispatch_frac = F.one_hot(top_idx[:, 0], num_e).float().mean(dim=0)
            lb_loss = num_e * torch.sum(dispatch_frac * probs.mean(dim=0))

        # Capacity and token dropping. k-major order: all first choices come before second choices.
        capacity = max(1, math.ceil(self.capacity_factor * n * k / num_e))
        flat_expert = top_idx.t().reshape(-1)
        flat_weight = top_w.t().reshape(-1)
        flat_token = torch.arange(n, device=flat.device).repeat(k)
        onehot = F.one_hot(flat_expert, num_e)
        position = (torch.cumsum(onehot, dim=0) - 1).gather(1, flat_expert[:, None]).squeeze(1)
        keep = position < capacity
        self.last_dropped_fraction = float((~keep).float().mean())

        out = torch.zeros(n, d, device=flat.device, dtype=torch.float32)
        for e in range(num_e):
            idx = torch.nonzero(keep & (flat_expert == e), as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            tokens = flat_token[idx]
            h = self.experts[e](flat[tokens])
            out.index_add_(0, tokens, h.float() * flat_weight[idx, None])
        for shared in self.shared_experts:
            out = out + shared(flat).float()
        if self.record_routing:
            self.last_probs = probs
            self.last_top_idx = top_idx.detach()
            self.last_out = out
        return out.to(x.dtype).view(b, t, d), lb_loss, router_z_loss


def router_similarity_loss(probs: torch.Tensor, top_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Router similarity loss (Expert Race, arXiv 2503.16057, Sec. 5.2). Flag-gated, off by default.

    probs:   [n, E] softmax router probabilities (differentiable).
    top_idx: [n, k] selected expert indices (hard, no gradient).
    M is the hard indicator of the selection. P' = P^T P, M' = M^T M, and
    L = (1/n) * sum_ij W(i,j) * P'(i,j). W is built from M' (no gradient):
      diagonal:     M'_ii / (E * sum_i M'_ii)
      off-diagonal: M'_ij / ((E^2 - E) * sum_{i!=j} M'_ij)
    The off-diagonal terms are penalised by how often two experts are picked together.
    """
    n, e = probs.shape
    m = torch.zeros(n, e, device=probs.device, dtype=probs.dtype)
    m.scatter_(1, top_idx, 1.0)
    with torch.no_grad():
        m_pair = m.t() @ m
        diag_mask = torch.eye(e, device=probs.device, dtype=torch.bool)
        diag_sum = m_pair.masked_select(diag_mask).sum().clamp_min(1e-6)
        off_sum = m_pair.masked_select(~diag_mask).sum().clamp_min(1e-6)
        weight = torch.where(
            diag_mask,
            m_pair / (diag_sum * e),
            m_pair / (off_sum * (e * e - e)),
        )
    p_pair = probs.t() @ probs
    return (weight * p_pair).sum() / n


def sage_loss(f_c: torch.Tensor, f_u: torch.Tensor, k_split: int = 2) -> torch.Tensor:
    """SAGE subspace-alignment loss (arXiv 2609.34525, Eq. 9 and Algorithm 1). Flag-gated.

    f_c, f_u: [n, d] MoE outputs for the conditional and the paired unconditional pass
    (n = B*T tokens). Rows are tokens, so the paper's F (d x n) is the transpose here.
    A column subset of size m = min(n, floor(d / k_split)) is taken from the stop-gradient
    conditional output. Its thin QR gives an orthonormal basis Q. The loss is
    ||F_u - Q Q^T F_u||_F^2 / ||F_u||_F^2. Gradients flow through f_u only.
    """
    n, d = f_u.shape
    m = min(n, d // max(1, k_split))
    if m < 1:
        raise ValueError("sage_loss needs at least one token and a non-zero split")
    fc = f_c.detach().float()
    fu = f_u.float()
    idx = torch.randperm(n, device=fc.device)[:m]
    with torch.no_grad():
        q, _ = torch.linalg.qr(fc[idx].t(), mode="reduced")  # [d, m], orthonormal columns
    proj = (fu @ q) @ q.t()
    residual = fu - proj
    denom = fu.pow(2).sum().clamp_min(1e-8)
    return residual.pow(2).sum() / denom


class RAYDiTBlock_v5(nn.Module):
    """v4 block with the dense FFN replaced by MoELayer. Returns (x, lb_loss, z_loss)."""

    def __init__(self, dim: int, heads: int, context_dim: int, cross_gate: bool = True,
                 grid: int = TOKEN_GRID, moe_kwargs: dict | None = None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttentionRoPE(dim, heads, grid=grid)
        self.norm_cross = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, heads, context_dim, cross_gate=cross_gate)
        self.norm2 = RMSNorm(dim)
        self.moe = MoELayer(dim, **(moe_kwargs or {}))
        self.cond = nn.Linear(dim, dim * 2)

    def forward(self, x, cond, text, text_mask=None):
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        x = x + self.self_attn(h)
        x = x + self.cross_attn(self.norm_cross(x), text, context_mask=text_mask)
        moe_out, lb_loss, z_loss = self.moe(self.norm2(x))
        return x + moe_out, lb_loss, z_loss


def _run_block(block, x, cond, text_ctx, text_mask):
    return block(x, cond, text_ctx, text_mask=text_mask)


class RAYDiT_v5(nn.Module):
    """Sparse-MoE RAY-IMAGE DiT for 16x64x64 latents with patch=4 (256 tokens)."""

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
        num_experts: int = MOE_NUM_EXPERTS,
        top_k: int = MOE_TOP_K,
        expert_hidden_dim: int = MOE_EXPERT_HIDDEN,
        num_shared_experts: int = MOE_NUM_SHARED,
        capacity_factor: float = MOE_CAPACITY_FACTOR,
        router_jitter_noise: float = MOE_ROUTER_JITTER,
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
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim

        moe_kwargs = dict(
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            num_shared_experts=num_shared_experts,
            capacity_factor=capacity_factor,
            router_jitter_noise=router_jitter_noise,
        )
        self.in_proj = nn.Conv2d(latent_channels, dim, kernel_size=patch, stride=patch)
        self.time = TimestepEmbedding(dim)
        self.text_proj = nn.Linear(cond_dim, dim)
        self.text_ctx_proj = nn.Linear(cond_dim, dim) if cond_dim != dim else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                RAYDiTBlock_v5(dim, heads, context_dim=dim, cross_gate=cross_gate, grid=self.grid,
                               moe_kwargs=moe_kwargs)
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, patch * patch * latent_channels)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        checkpoint_blocks: bool = False,
        return_losses: bool = True,
        capture: bool = False,
    ):
        """Training forward. Returns (prediction, load_balancing_loss, router_z_loss) by default.

        return_losses=False returns only the prediction tensor, the same shape as v4's forward.
        Use predict_velocity() for inference rather than passing this flag by hand.
        checkpoint_blocks=True recomputes each block in the backward pass (training only).
        capture=True stores, per layer, (router probs, top-k indices, MoE output) in
        self.last_captures for the SAGE and router-similarity regularisers. Default off.
        """
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
        lb_total = x.new_zeros((), dtype=torch.float32)
        z_total = x.new_zeros((), dtype=torch.float32)
        if capture:
            for block in self.blocks:
                block.moe.record_routing = True
        for block in self.blocks:
            if checkpoint_blocks and self.training:
                x, lb, zl = checkpoint(_run_block, block, x, cond, text_ctx, text_mask, use_reentrant=False)
            else:
                x, lb, zl = block(x, cond, text_ctx, text_mask=text_mask)
            lb_total = lb_total + lb
            z_total = z_total + zl
        if capture:
            self.last_captures = []
            for block in self.blocks:
                moe = block.moe
                self.last_captures.append((moe.last_probs, moe.last_top_idx, moe.last_out))
                moe.record_routing = False
                moe.last_probs = moe.last_top_idx = moe.last_out = None

        x = self.out(self.norm(x))
        x = x.view(b, ph, pw, self.patch, self.patch, self.latent_channels)
        prediction = x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.latent_channels, h, w)
        if not return_losses:
            return prediction
        return prediction, lb_total / self.depth, z_total / self.depth

    def predict_velocity(self, z: torch.Tensor, t: torch.Tensor, text: torch.Tensor,
                         text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Inference entry point: predicted velocity tensor only, no auxiliary losses.

        Call it in eval mode (no router jitter, no block checkpointing).
        """
        return self.forward(z, t, text, text_mask=text_mask, return_losses=False)

    def dropped_fraction(self) -> float:
        """Mean fraction of routed assignments dropped by capacity in the last forward pass."""
        values = [block.moe.last_dropped_fraction for block in self.blocks]
        return sum(values) / max(len(values), 1)


WARM_START_DOWN_SCALE = 0.5  # applied to every copied down-projection (see warm_start_from_v4)


def warm_start_from_v4(model: RAYDiT_v5, v4_state: dict) -> None:
    """Load a RAYDiT_v4 state dict into RAYDiT_v5.

    * Every non-FFN key is copied one-to-one.
    * Each block's dense FFN (blocks.i.ffn.{gate,up,down}.weight) is copied into every routed
      expert (blocks.i.moe.experts.e.*) and every shared expert (blocks.i.moe.shared_experts.s.*).
    * The down-projection is multiplied by WARM_START_DOWN_SCALE (0.5) in every copy. The
      routed top-k weights sum to 1, so the routed part is 0.5*FFN(x) and the shared expert
      adds 0.5*FFN(x). Together they give FFN(x) at init (no capacity drops), so the
      residual branch keeps its v4 scale instead of doubling.
    * The router (blocks.i.moe.router.*) keeps its fresh initialisation.

    The key sets are checked exactly and the load is strict. A mismatch raises.
    """
    target = model.state_dict()
    router_keys = {k for k in target if ".moe.router." in k}
    new_state = {}
    for key, value in v4_state.items():
        if ".ffn." in key:
            prefix, rest = key.split(".ffn.", 1)
            scaled = value * WARM_START_DOWN_SCALE if rest.startswith("down.") else value
            for e in range(model.num_experts):
                new_state[f"{prefix}.moe.experts.{e}.{rest}"] = scaled.clone()
            for s in range(model.num_shared_experts):
                new_state[f"{prefix}.moe.shared_experts.{s}.{rest}"] = scaled.clone()
        else:
            new_state[key] = value
    for key in router_keys:
        new_state[key] = target[key]

    missing = sorted(set(target) - set(new_state))
    unexpected = sorted(set(new_state) - set(target))
    if missing or unexpected:
        raise RuntimeError(f"warm start key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.load_state_dict(new_state, strict=True)
