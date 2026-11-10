"""DiT v5 (sparse MoE, 16x64x64 latents) training on the rolling-window MONET Qwen cache.

Data pipeline, windows, replay buffer, checkpoint policy and recipe are the same as
train_dit_v4.py. Those helpers are imported from it. Changes in this file:
  * RAYDiT_v5 (dit_v5_moe.py): MoE feed-forwards.
  * Loss: total = mse + aux_loss_weight * load_balancing + z_loss_weight * router_z.
  * --warm-start-from: copy a v4 checkpoint into all experts; training continues at the v4 step.
  * --resume: loads a v5 checkpoint with strict=True. Architecture checked (RB-11).
  * v5 file names and HF repos (ray-image-dit-v5-moe).

Checkpoint files are written as a pair, as in v4:
  ray_dit_v5_step_XXXXXXX.pt      DiT + EMA + optimizer + step (resume)
  ray_dit_v5_ema_step_XXXXXXX.pt  EMA weights under key "dit" (inference)
"""

import argparse
import copy
import functools
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, create_repo
from torch.optim import AdamW

from .cfg import (
    NULL_PROMPT,
    apply_caption_dropout,
    ema_drift,
    fit_null_to_length,
    load_or_build_null_prompt,
    update_ema,
)
from .dit_v4 import LATENT_CHANNELS, LATENT_SIZE, MODEL_DEPTH, MODEL_DIM, MODEL_HEADS, MODEL_PATCH
from .dit_v5_moe import (
    MOE_CAPACITY_FACTOR,
    MOE_EXPERT_HIDDEN,
    MOE_NUM_EXPERTS,
    MOE_NUM_SHARED,
    MOE_ROUTER_JITTER,
    MOE_TOP_K,
    RAYDiT_v5,
    WARM_START_DOWN_SCALE,
    router_similarity_loss,
    sage_loss,
    warm_start_from_v4,
)
from .train_dit_v4 import (
    CHECKPOINT_EVERY,
    DISK_REFUSE_SAVE_PCT,
    GRAD_CLIP,
    HF_KEEP,
    HF_UPLOAD_EVERY,
    LATENT_SPACE,
    LOCAL_CHECKPOINT_DIR,
    LOG_EVERY,
    MILESTONE_EVERY,
    REPLAY_FILENAME,
    TEXT_DIM,
    TRAIN_LR,
    VAE_DEFAULT_CHECKPOINT,
    VAE_DEFAULT_REPO,
    WARMUP_STEPS,
    WINDOW_STEPS,
    Window,
    ReplayBuffer,
    _encode_null_with_qwen,
    _step_of,
    check_resume_config,
    checkpoint_has_corruption,
    load_vae,
    log_disk_usage,
    sample_training_batch,
    seed_everything,
    set_learning_rate,
)

HF_DEFAULT_REPO_V5 = "Rishix500/ray-image-dit-v5-moe"
MILESTONE_DEFAULT_REPO_V5 = "Rishix500/ray-image-dit-v5-moe-milestones"
# Multi-repo cache (see repo_for_chunk). 500k samples = 50 chunks of 10k.
CACHE_DEFAULT_REPOS = "Rishix500/monet-qwen-cache-200k-v4,Rishix500/mixed-qwen-cache-250k-v5"
CHUNKS_PER_REPO_DEFAULT = 25
TOTAL_CHUNKS_DEFAULT = 50
CKPT_PREFIX = "ray_dit_v5_step_"
EMA_CKPT_PREFIX = "ray_dit_v5_ema_step_"
CKPT_PREFIXES = (CKPT_PREFIX, EMA_CKPT_PREFIX)
STAGE_NAME = "dit_v5_moe"
ARCH_NAME = "dit_v5_moe"
# Local retention for v5 (decision): 2 full + 2 EMA = about 19.2 GB at 482M params.
LOCAL_KEEP_V5 = 2

# Set in main() from the CLI (read by the checkpoint provenance block).
CAPTION_DROP = 0.10
EMA_DECAY = 0.999
VAE_CHECKPOINT = None
RUN_ID = None


def build_model(args) -> RAYDiT_v5:
    return RAYDiT_v5(
        latent_channels=LATENT_CHANNELS,
        latent_size=LATENT_SIZE,
        dim=MODEL_DIM,
        depth=MODEL_DEPTH,
        heads=MODEL_HEADS,
        patch=MODEL_PATCH,
        cond_dim=TEXT_DIM,
        cross_gate=True,
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_hidden_dim=args.expert_hidden_dim,
        num_shared_experts=MOE_NUM_SHARED,
        capacity_factor=args.capacity_factor,
        router_jitter_noise=MOE_ROUTER_JITTER,
    )


def architecture_config(args) -> dict:
    """RB-11: architecture values a resume must match exactly."""
    return {
        "arch": ARCH_NAME,
        "image_size": 512,
        "latent_channels": LATENT_CHANNELS,
        "latent_size": LATENT_SIZE,
        "dim": MODEL_DIM,
        "depth": MODEL_DEPTH,
        "heads": MODEL_HEADS,
        "patch": MODEL_PATCH,
        "text_dim": TEXT_DIM,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "expert_hidden_dim": args.expert_hidden_dim,
        "num_shared_experts": MOE_NUM_SHARED,
    }


def build_config(args) -> dict:
    config = dict(architecture_config(args))
    config.update({
        "flow_matching": "linear_gaussian",
        "cross_gate": True,
        "text_max_tokens": 128,
        "capacity_factor": args.capacity_factor,
        "aux_loss_weight": args.aux_loss_weight,
        "z_loss_weight": args.z_loss_weight,
        "use_sage": bool(args.use_sage),
        "sage_k_split": args.sage_k_split,
        "sage_weight": args.sage_weight,
        "use_sim_loss": bool(args.use_sim_loss),
        "sim_loss_weight": args.sim_loss_weight,
        # RB-03: provenance so a later run (or generate.py) can tell exactly what this was trained on.
        "latent_space": LATENT_SPACE,
        "vae_checkpoint": VAE_CHECKPOINT,
        "cache_repos": parse_cache_repos(args.cache_repos),
        "chunks_per_repo": args.chunks_per_repo,
        "total_chunks": args.total_chunks,
        "cache_window": args.cache_window,
        "replay_buffer": args.replay_buffer,
        "caption_drop": CAPTION_DROP,
        "ema_decay": EMA_DECAY,
        "null_prompt": NULL_PROMPT,
        "torch": torch.__version__,
        "run": RUN_ID,
    })
    return config


def train_step(dit, ema, optimizer, scaler, batch, null_emb_full, null_mask_full, args, device):
    """One optimiser step.

    Returns (stats dict or None if skipped, number of dropped captions).
    total = mse + aux_loss_weight * load_balancing + z_loss_weight * router_z.
    """
    clean_latents, text_hidden, text_mask = batch
    null_emb_fit, null_mask_fit = fit_null_to_length(null_emb_full, null_mask_full, text_hidden.shape[1])
    text_hidden, text_mask, dropped = apply_caption_dropout(
        text_hidden, text_mask, null_emb_fit, null_mask_fit, args.caption_drop,
    )
    noise = torch.randn_like(clean_latents)
    t = torch.rand(clean_latents.shape[0], device=device)
    t_view = t[:, None, None, None]
    z_t = (1.0 - t_view) * clean_latents + t_view * noise
    target_velocity = noise - clean_latents
    optimizer.zero_grad(set_to_none=True)
    # SAGE and the router similarity loss read the per-layer routing captures. Both are off by default,
    # so the baseline run does the same forward as before.
    capture = bool(args.use_sage or args.use_sim_loss)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        if capture:
            prediction, lb_loss, z_loss = dit(z_t, t, text_hidden, text_mask, checkpoint_blocks=True, capture=True)
            cond_captures = dit.last_captures
        else:
            prediction, lb_loss, z_loss = dit(z_t, t, text_hidden, text_mask, checkpoint_blocks=True)
        mse = F.mse_loss(prediction.float(), target_velocity.float())
    lb_loss = lb_loss.float()
    z_loss = z_loss.float()
    loss = mse + args.aux_loss_weight * lb_loss + args.z_loss_weight * z_loss
    extra = {}
    if args.use_sim_loss:
        sim_loss = torch.stack(
            [router_similarity_loss(probs, idx, dit.num_experts) for probs, idx, _ in cond_captures]
        ).mean()
        loss = loss + args.sim_loss_weight * sim_loss
        extra["sim"] = float(sim_loss.detach().item())
    if args.use_sage:
        # Paired unconditional pass on the same x_t and t, with the null prompt for every sample.
        batch_size = text_hidden.shape[0]
        null_text = null_emb_fit.unsqueeze(0).expand(batch_size, -1, -1)
        null_text_mask = null_mask_fit.unsqueeze(0).expand(batch_size, -1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            dit(z_t, t, null_text, null_text_mask, checkpoint_blocks=True, capture=True)
            uncond_captures = dit.last_captures
        sage_terms = [
            sage_loss(f_c, f_u, k_split=args.sage_k_split)
            for (_, _, f_c), (_, _, f_u) in zip(cond_captures, uncond_captures)
        ]
        sage = torch.stack(sage_terms).mean()
        loss = loss + args.sage_weight * sage
        extra["sage"] = float(sage.detach().item())
    if not torch.isfinite(loss).item():
        optimizer.zero_grad(set_to_none=True)
        return None, int(dropped)
    # fp16 autocast needs loss scaling. GradScaler is a no-op when disabled (CPU).
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # so clipping sees the true gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), GRAD_CLIP)
    # scaler.step skips the update if any gradient is inf/NaN, and scaler.update lowers the scale.
    scaler.step(optimizer)
    scaler.update()
    if not torch.isfinite(grad_norm).item():
        optimizer.zero_grad(set_to_none=True)
        return None, int(dropped)
    update_ema(ema, dit, args.ema_decay)
    stats = {"loss": float(loss.item()), "mse": float(mse.item()),
             "lb": float(lb_loss.item()), "z": float(z_loss.item()), **extra}
    return stats, int(dropped)


def set_v5_learning_rate(optimizer, base_lr: float, step: int, ramp_start, warmup_steps: int) -> float:
    """LR for v5. Without a ramp this is the v4 schedule. With a ramp (warm start, or resume
    inside one), the LR is 0 at ramp_start and rises linearly to base_lr over warmup_steps.
    This keeps the fresh Adam moments from taking full-size steps at once."""
    if ramp_start is None:
        return set_learning_rate(optimizer, base_lr, step)
    if warmup_steps <= 0:
        progress = 1.0
    else:
        progress = min(max((step - ramp_start) / warmup_steps, 0.0), 1.0)
    lr = base_lr * progress
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# --------------------------------------------------------------------------
# Checkpoints (same policy as v4: local keep-4, HF main keep-4 per kind, milestones kept)
# --------------------------------------------------------------------------

def build_full_payload(dit, optimizer, ema, scaler, step: int, seed: int, window_index: int, config: dict,
                       lr_ramp=None) -> dict:
    payload = {
        "stage": STAGE_NAME,
        "step": step,
        "window": window_index,
        "seed": seed,
        "config": config,
        "dit": dit.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    if lr_ramp is not None:
        payload["lr_ramp"] = {"start": int(lr_ramp[0]), "steps": int(lr_ramp[1])}
    return payload


def build_ema_payload(ema, step: int, config: dict) -> dict:
    return {"stage": STAGE_NAME, "step": step, "config": config, "dit": ema.state_dict()}


def prune_local_checkpoints(out_dir: Path, keep: int) -> None:
    """Remove stale *.pt.tmp from crashed saves, then keep the newest `keep` of each kind."""
    for stale in out_dir.glob("*.pt.tmp"):
        try:
            stale.unlink()
            print(f"[local cleanup] removed stale temp file {stale.name}")
        except OSError as exc:
            print(f"[local cleanup] failed to remove {stale.name}: {exc}")
    for prefix in CKPT_PREFIXES:
        files = sorted(out_dir.glob(f"{prefix}*.pt"), key=lambda p: _step_of(p.name))
        for path in files[:-keep]:
            try:
                path.unlink()
                print(f"[local cleanup] deleted {path.name}")
            except OSError as exc:
                print(f"[local cleanup] failed to delete {path.name}: {exc}")


def save_local_checkpoint(out_dir: Path, dit, optimizer, ema, scaler, step: int, seed: int,
                          window_index: int, args, lr_ramp=None):
    """Write both files for this step. Returns (full_path, ema_path), or None if skipped."""
    if log_disk_usage(out_dir) > DISK_REFUSE_SAVE_PCT:
        print(f"[checkpoint] SKIP step={step}: disk above {DISK_REFUSE_SAVE_PCT:.0f}%")
        return None
    if checkpoint_has_corruption(dit) or checkpoint_has_corruption(ema):
        print(f"[checkpoint] SKIP step={step}: corrupted weights")
        return None
    config = build_config(args)
    full_path = out_dir / f"{CKPT_PREFIX}{step:07d}.pt"
    ema_path = out_dir / f"{EMA_CKPT_PREFIX}{step:07d}.pt"
    for path, payload in (
        (full_path, build_full_payload(dit, optimizer, ema, scaler, step, seed, window_index, config, lr_ramp)),
        (ema_path, build_ema_payload(ema, step, config)),
    ):
        tmp = out_dir / (path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
    print(f"[checkpoint] saved {full_path.name} + {ema_path.name}")
    prune_local_checkpoints(out_dir, LOCAL_KEEP_V5)
    return full_path, ema_path


def upload_main_checkpoints(api: HfApi, paths, repo_id: str, step: int) -> None:
    """Upload both files to the main branch, then keep only the newest HF_KEEP of each kind.

    Deletes use HfApi.delete_file. Git history is not rewritten and nothing is force-pushed.
    """
    for path in paths:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=repo_id,
                        repo_type="model", commit_message=f"checkpoint step {step}")
        print(f"[HF] uploaded {path.name} -> {repo_id}")
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    for prefix in CKPT_PREFIXES:
        ckpts = sorted((f for f in files if "/" not in f and f.startswith(prefix) and f.endswith(".pt")),
                       key=lambda f: _step_of(f))
        for old in ckpts[:-HF_KEEP]:
            try:
                api.delete_file(path_in_repo=old, repo_id=repo_id, repo_type="model",
                                commit_message=f"prune old checkpoint {old}")
                print(f"[HF cleanup] deleted {old} from main branch")
            except Exception as exc:
                print(f"[HF cleanup] failed to delete {old}: {exc}")


def upload_milestone_checkpoints(api: HfApi, paths, repo_id: str, step: int) -> None:
    """Milestones are uploaded and never deleted."""
    for path in paths:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=repo_id,
                        repo_type="model", commit_message=f"milestone step {step}")
        print(f"[milestone] uploaded {path.name} -> {repo_id}")


def load_resume(path: Path, dit, optimizer, ema, scaler, args):
    """Returns (step, lr_ramp). lr_ramp is (start, steps) if the checkpoint was saved inside a ramp, else None."""
    """Resume a v5 checkpoint. Loaded to CPU first, so the GPU never holds two full copies."""
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    check_resume_config(data, expected=architecture_config(args))  # RB-11
    dit.load_state_dict(data["dit"], strict=True)
    if "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    else:
        print("[resume] WARNING: checkpoint has no optimizer state; AdamW moments restart from zero")
    if scaler.is_enabled() and "scaler" in data:
        scaler.load_state_dict(data["scaler"])
    elif scaler.is_enabled():
        print("[resume] WARNING: checkpoint has no GradScaler state; loss scale restarts")
    if "ema" in data:
        ema.load_state_dict(data["ema"], strict=True)
    else:
        print("[resume] WARNING: checkpoint has no EMA weights; EMA restarts from the loaded DiT weights")
        ema.load_state_dict(dit.state_dict(), strict=True)
    step = int(data.get("step", 0))
    ramp = data.get("lr_ramp")
    lr_ramp = (int(ramp["start"]), int(ramp["steps"])) if ramp else None
    print(f"[resume] loaded {path.name} at step {step} (strict); lr ramp={lr_ramp}")
    return step, lr_ramp


def load_warm_start(path: Path, dit, ema, warmup_steps: int) -> int:
    """Copy a v4 checkpoint's FFN weights into every expert. Returns the v4 step to continue from.

    The step comes from the checkpoint's "step" field, or from "step_NNNNNNN" in the file name.
    Continuing the v4 count keeps the LR schedule and the milestone step numbers consistent.
    """
    if not path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    step = data.get("step") if isinstance(data, dict) else None
    if step is None:
        match = re.search(r"step_(\d+)", path.name)
        if match is None:
            raise ValueError(f"cannot determine the v4 step for {path.name}: no 'step' field and no step_NNNNNNN in the name")
        step = int(match.group(1))
    v4_state = data["dit"] if "dit" in data else data
    warm_start_from_v4(dit, v4_state)
    ema.load_state_dict(dit.state_dict(), strict=True)
    print(f"[warm start] copied {path.name} (v4 FFN -> all experts, down-proj x{WARM_START_DOWN_SCALE}); "
          f"continuing at step {int(step)}. Fresh optimizer and empty replay buffer.")
    print(f"[warm start] LR ramps from 0 at step {int(step)} to the target LR over {warmup_steps} steps")
    return int(step)


def parse_cache_repos(text: str) -> list:
    """Parse --cache-repos ("org/a,org/b"). Empty entries are ignored. Each entry must be org/name."""
    repos = [item.strip() for item in text.split(",") if item.strip()]
    if not repos:
        raise ValueError("--cache-repos must name at least one Hugging Face dataset repo")
    for repo in repos:
        parts = repo.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"--cache-repos entry {repo!r} must have the form org/name")
    return repos


def repo_for_chunk(chunk_id: int, repos, chunks_per_repo: int) -> str:
    """Return the repo that holds chunk_id: repos[chunk_id // chunks_per_repo].

    With the defaults (25 chunks per repo, two repos) chunks 0-24 come from repo 0 and
    chunks 25-49 from repo 1. A chunk past the last repo raises ValueError.
    """
    if chunk_id < 0:
        raise ValueError(f"chunk id must be >= 0, got {chunk_id}")
    if chunks_per_repo < 1:
        raise ValueError(f"--chunks-per-repo must be >= 1, got {chunks_per_repo}")
    if not repos:
        raise ValueError("no cache repos given")
    repo_index = chunk_id // chunks_per_repo
    if repo_index >= len(repos):
        raise ValueError(f"Requested chunk {chunk_id}, but only {len(repos)} repos provided.")
    return repos[repo_index]


def make_windows_v5(total_chunks: int, cache_window: int):
    """Consecutive groups of cache_window chunks, covering chunks 0..total_chunks-1."""
    indices = list(range(total_chunks))
    return [indices[i:i + cache_window] for i in range(0, len(indices), cache_window)]


def open_window(index: int, chunk_indices, cache_dir: Path, cache_repo):
    """The only place training data is opened. The trainer uses the returned object through
    load(), sample_training_batch(), replay.absorb() and release(). To change the data format
    (for example WebDataset or Parquet streaming), replace this function and keep that interface."""
    return Window(index, chunk_indices, cache_dir, cache_repo)  # cache_repo: callable chunk_id -> repo


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train RAY-IMAGE DiT v5 (sparse MoE) on the rolling-window cache.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--cache-repos", type=str, default=CACHE_DEFAULT_REPOS,
                        help="Comma-separated Hugging Face dataset repos, in chunk order.")
    parser.add_argument("--chunks-per-repo", type=int, default=CHUNKS_PER_REPO_DEFAULT,
                        help="Chunks held by each repo before switching to the next. The last repo takes the rest.")
    parser.add_argument("--total-chunks", type=int, default=TOTAL_CHUNKS_DEFAULT,
                        help="Number of 10k-sample chunks to train over (50 = 500k samples).")
    parser.add_argument("--cache-window", type=int, default=2)
    parser.add_argument("--replay-buffer", type=int, default=5000)
    parser.add_argument("--vae-repo", type=str, default=VAE_DEFAULT_REPO)
    parser.add_argument("--vae-checkpoint", type=str, default=VAE_DEFAULT_CHECKPOINT)
    parser.add_argument("--hf-repo", type=str, default=HF_DEFAULT_REPO_V5)
    parser.add_argument("--milestone-repo", type=str, default=MILESTONE_DEFAULT_REPO_V5)
    parser.add_argument("--resume", type=str, default=None,
                        help="Local v5 checkpoint (.pt) to resume from. Loaded with strict=True.")
    parser.add_argument("--warm-start-from", type=str, default=None,
                        help="Local v4 checkpoint (.pt). Its FFN weights are copied into every expert.")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="With --warm-start-from: LR ramps linearly from 0 to --lr over this many steps.")
    parser.add_argument("--output-dir", type=str, default=LOCAL_CHECKPOINT_DIR)
    parser.add_argument("--no-hf-upload", action="store_true")
    parser.add_argument("--caption-drop", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    # MoE
    parser.add_argument("--num-experts", type=int, default=MOE_NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=MOE_TOP_K)
    parser.add_argument("--expert-hidden-dim", type=int, default=MOE_EXPERT_HIDDEN)
    parser.add_argument("--aux-loss-weight", type=float, default=0.01)
    parser.add_argument("--z-loss-weight", type=float, default=0.001)
    # Off by default. The baseline runs with neither flag.
    parser.add_argument("--use-sim-loss", action="store_true",
                        help="Add the router similarity loss (Expert Race). Default off.")
    parser.add_argument("--sim-loss-weight", type=float, default=1e-4,
                        help="Weight on the router similarity loss, averaged over layers. Default 1e-4.")
    parser.add_argument("--use-sage", action="store_true",
                        help="Add the SAGE subspace loss. Costs an extra unconditional forward. Default off.")
    parser.add_argument("--sage-k-split", type=int, default=2,
                        help="SAGE split factor: basis size is d_model // k_split. Paper value: 2.")
    parser.add_argument("--sage-weight", type=float, default=1.0,
                        help="Weight on SAGE, averaged over layers. Paper value: 1.0.")
    parser.add_argument("--capacity-factor", type=float, default=MOE_CAPACITY_FACTOR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("--steps must be >= 0, --batch-size > 0, --lr > 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.cache_window < 1 or args.replay_buffer < 0:
        raise ValueError("--cache-window must be >= 1 and --replay-buffer >= 0")
    if not 0.0 <= args.caption_drop <= 1.0:
        raise ValueError("--caption-drop must be in [0, 1]")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if args.aux_loss_weight < 0 or args.z_loss_weight < 0:
        raise ValueError("--aux-loss-weight and --z-loss-weight must be >= 0")
    if args.resume and args.warm_start_from:
        raise ValueError("use either --resume (v5 checkpoint) or --warm-start-from (v4 checkpoint), not both")
    repos = parse_cache_repos(args.cache_repos)
    if args.total_chunks < 1:
        raise ValueError("--total-chunks must be >= 1")
    windows = make_windows_v5(args.total_chunks, args.cache_window)
    needed = math.ceil(args.steps / WINDOW_STEPS)
    if needed > len(windows):
        raise ValueError(f"--steps {args.steps} needs {needed} windows of {WINDOW_STEPS} steps, "
                         f"but --cache-window {args.cache_window} with --total-chunks {args.total_chunks} "
                         f"gives only {len(windows)}")
    # Fail before training if any chunk needed by --steps cannot be mapped to a repo.
    for w in range(needed):
        for chunk_id in windows[w]:
            repo_for_chunk(chunk_id, repos, args.chunks_per_repo)
    for repo_index, repo in enumerate(repos):
        first = repo_index * args.chunks_per_repo
        last = min(first + args.chunks_per_repo, args.total_chunks) - 1
        if first < args.total_chunks:
            print(f"[cache] repo {repo_index}: {repo} holds chunks {first}-{last}")
    if not torch.cuda.is_available():
        raise RuntimeError("DiT v5 training requires CUDA")

    global CAPTION_DROP, EMA_DECAY, VAE_CHECKPOINT, RUN_ID
    CAPTION_DROP = args.caption_drop
    EMA_DECAY = args.ema_decay
    VAE_CHECKPOINT = args.vae_checkpoint
    RUN_ID = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-seed{args.seed}"
    seed_everything(args.seed)
    device = torch.device("cuda")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    upload = not args.no_hf_upload
    api = None
    if upload:
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for uploads and cache downloads (or pass --no-hf-upload)")
        api = HfApi(token=hf_token)
        create_repo(repo_id=args.hf_repo, repo_type="model", private=True, exist_ok=True, token=hf_token)
        create_repo(repo_id=args.milestone_repo, repo_type="model", private=True, exist_ok=True, token=hf_token)

    vae = load_vae(device, args.vae_repo, args.vae_checkpoint)
    print("[VAE] verified; latents are precomputed so the loss does not use the VAE")

    dit = build_model(args).to(device)
    optimizer = AdamW(dit.parameters(), lr=0.0, betas=(0.9, 0.99), weight_decay=0.01)
    ema = copy.deepcopy(dit)
    ema.requires_grad_(False)
    ema.eval()
    # fp16 autocast on a T4 needs loss scaling. The scaler is disabled on CPU.
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    step = 0
    lr_ramp = None  # (start_step, steps) while the LR is ramping; None = plain v4 schedule
    if args.resume:
        step, lr_ramp = load_resume(Path(args.resume), dit, optimizer, ema, scaler, args)
    elif args.warm_start_from:
        step = load_warm_start(Path(args.warm_start_from), dit, ema, args.warmup_steps)
        lr_ramp = (step, args.warmup_steps) if args.warmup_steps > 0 else None
    else:
        print("[init] WARNING: no --resume or --warm-start-from; training from random init")
    if step >= args.steps:
        print(f"[done] target steps={args.steps} already reached at step={step}")
        return

    null_payload = load_or_build_null_prompt(
        args.hf_repo, text_dim=TEXT_DIM, build_fn=_encode_null_with_qwen, upload=upload,
    )
    null_emb_full = null_payload["emb"].to(device=device, dtype=torch.float16)
    null_mask_full = null_payload["mask"].to(device=device, dtype=torch.bool)

    replay = ReplayBuffer(args.replay_buffer)
    replay_path = out_dir / REPLAY_FILENAME
    if args.resume and replay_path.is_file():
        replay.load(replay_path)
    elif args.resume:
        print("[replay] WARNING: no replay buffer file next to the resume point; starting empty")

    print(f"[config] arch={ARCH_NAME} experts={args.num_experts} top_k={args.top_k} "
          f"expert_hidden={args.expert_hidden_dim} capacity_factor={args.capacity_factor} "
          f"aux_w={args.aux_loss_weight} z_w={args.z_loss_weight}")
    print(f"[config] batch={args.batch_size} lr={args.lr:.2e} warmup={WARMUP_STEPS} target_steps={args.steps} "
          f"window_steps={WINDOW_STEPS} cache_window={args.cache_window} replay_buffer={args.replay_buffer}")
    print(f"[cfg] caption_drop={args.caption_drop:.3f} null_tokens={null_emb_full.shape[0]} ema_decay={args.ema_decay}")
    print(f"[DiT v5] total parameters: {sum(p.numel() for p in dit.parameters()):,}")
    log_disk_usage(out_dir)

    window = None
    run_total = run_mse = run_lb = run_z = 0.0
    run_reg = {"sage": 0.0, "sim": 0.0}
    run_count = 0
    dropped_total = 0
    seen_total = 0
    while step < args.steps:
        w = step // WINDOW_STEPS
        window_end = min((w + 1) * WINDOW_STEPS, args.steps)
        if window is None:
            resolver = functools.partial(repo_for_chunk, repos=repos, chunks_per_repo=args.chunks_per_repo)
            window = open_window(w, windows[w], cache_dir, resolver)
            window.load()
            print(f"[window {w}] steps {step + 1}..{window_end}")
        while step < window_end:
            step += 1
            set_v5_learning_rate(optimizer, args.lr, step, lr_ramp[0] if lr_ramp else None,
                                 lr_ramp[1] if lr_ramp else 0)
            batch = sample_training_batch(window, replay, args.batch_size, device)
            stats, dropped = train_step(dit, ema, optimizer, scaler, batch, null_emb_full, null_mask_full, args, device)
            seen_total += args.batch_size
            dropped_total += dropped
            if stats is not None:
                run_total += stats["loss"]
                run_mse += stats["mse"]
                run_lb += stats["lb"]
                run_z += stats["z"]
                for key in run_reg:
                    run_reg[key] += stats.get(key, 0.0)
                run_count += 1

            if step % LOG_EVERY == 0:
                n = max(run_count, 1)
                drift = ema_drift(ema, dit)
                dropped_frac = dropped_total / max(seen_total, 1)
                loss_out = run_total / n if run_count else float("nan")
                print(f"step={step} loss={loss_out:.4f} ema_drift={drift:.4f} dropped={dropped_frac:.3f}")
                print(f"[aux] step={step} mse={run_mse / n:.4f} lb={run_lb / n:.4f} z={run_z / n:.4f} "
                      f"route_dropped={dit.dropped_fraction():.4f}")
                if args.use_sage or args.use_sim_loss:
                    print(f"[reg] step={step} sage={run_reg['sage'] / n:.4f} sim={run_reg['sim'] / n:.6f}")
                run_total = run_mse = run_lb = run_z = 0.0
                run_reg = {"sage": 0.0, "sim": 0.0}
                run_count = 0
                dropped_total = 0
                seen_total = 0

            if step % CHECKPOINT_EVERY == 0 or step == args.steps:
                saved = save_local_checkpoint(out_dir, dit, optimizer, ema, scaler, step, args.seed, w, args, lr_ramp)
                if saved is not None and upload:
                    if step % HF_UPLOAD_EVERY == 0 or step == args.steps:
                        try:
                            upload_main_checkpoints(api, saved, args.hf_repo, step)
                        except Exception as exc:
                            print(f"[HF] upload of step {step} to {args.hf_repo} FAILED: {exc}")
                    if step % MILESTONE_EVERY == 0:
                        try:
                            upload_milestone_checkpoints(api, saved, args.milestone_repo, step)
                        except Exception as exc:
                            print(f"[milestone] upload of step {step} FAILED: {exc}")

        # Window finished: refresh the replay buffer, save it, then free its chunk files.
        replay.absorb(window)
        replay.save(replay_path)
        window.release()
        window = None
        log_disk_usage(out_dir)

    print(f"[done] reached step={step}")
    del vae


if __name__ == "__main__":
    main()
