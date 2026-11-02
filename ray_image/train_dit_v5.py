"""DiT v5 (sparse MoE, 16x64x64 latents) training on the rolling-window MONET Qwen cache.

Data pipeline, windows, replay buffer, checkpoint policy and recipe are the same as
train_dit_v4.py. Those helpers are imported from it. Changes in this file:
  * RAYDiT_v5 (dit_v5_moe.py): MoE feed-forwards.
  * Loss: total = mse + aux_loss_weight * load_balancing + z_loss_weight * router_z.
  * --warm-start-from: copy a v4 checkpoint into all experts (fresh step counter).
  * --resume: loads a v5 checkpoint with strict=True. Architecture checked (RB-11).
  * v5 file names and HF repos (ray-image-dit-v5-moe).

Checkpoint files are written as a pair, as in v4:
  ray_dit_v5_step_XXXXXXX.pt      DiT + EMA + optimizer + step (resume)
  ray_dit_v5_ema_step_XXXXXXX.pt  EMA weights under key "dit" (inference)
"""

import argparse
import copy
import math
import os
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
    warm_start_from_v4,
)
from .train_dit_v4 import (
    CACHE_DEFAULT_REPO,
    CHECKPOINT_EVERY,
    DISK_REFUSE_SAVE_PCT,
    GRAD_CLIP,
    HF_KEEP,
    HF_UPLOAD_EVERY,
    LATENT_SPACE,
    LOCAL_CHECKPOINT_DIR,
    LOCAL_KEEP,
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
    make_windows,
    sample_training_batch,
    seed_everything,
    set_learning_rate,
)

HF_DEFAULT_REPO_V5 = "Rishix500/ray-image-dit-v5-moe"
MILESTONE_DEFAULT_REPO_V5 = "Rishix500/ray-image-dit-v5-moe-milestones"
CKPT_PREFIX = "ray_dit_v5_step_"
EMA_CKPT_PREFIX = "ray_dit_v5_ema_step_"
CKPT_PREFIXES = (CKPT_PREFIX, EMA_CKPT_PREFIX)
STAGE_NAME = "dit_v5_moe"
ARCH_NAME = "dit_v5_moe"

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
        # RB-03: provenance so a later run (or generate.py) can tell exactly what this was trained on.
        "latent_space": LATENT_SPACE,
        "vae_checkpoint": VAE_CHECKPOINT,
        "cache_repo": args.cache_repo,
        "cache_window": args.cache_window,
        "replay_buffer": args.replay_buffer,
        "caption_drop": CAPTION_DROP,
        "ema_decay": EMA_DECAY,
        "null_prompt": NULL_PROMPT,
        "torch": torch.__version__,
        "run": RUN_ID,
    })
    return config


def train_step(dit, ema, optimizer, batch, null_emb_full, null_mask_full, args, device):
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
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        prediction, lb_loss, z_loss = dit(z_t, t, text_hidden, text_mask, checkpoint_blocks=True)
        mse = F.mse_loss(prediction.float(), target_velocity.float())
    lb_loss = lb_loss.float()
    z_loss = z_loss.float()
    loss = mse + args.aux_loss_weight * lb_loss + args.z_loss_weight * z_loss
    if not torch.isfinite(loss).item():
        optimizer.zero_grad(set_to_none=True)
        return None, int(dropped)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm).item():
        optimizer.zero_grad(set_to_none=True)
        return None, int(dropped)
    optimizer.step()
    update_ema(ema, dit, args.ema_decay)
    stats = {"loss": float(loss.item()), "mse": float(mse.item()),
             "lb": float(lb_loss.item()), "z": float(z_loss.item())}
    return stats, int(dropped)


# --------------------------------------------------------------------------
# Checkpoints (same policy as v4: local keep-4, HF main keep-4 per kind, milestones kept)
# --------------------------------------------------------------------------

def build_full_payload(dit, optimizer, ema, step: int, seed: int, window_index: int, config: dict) -> dict:
    return {
        "stage": STAGE_NAME,
        "step": step,
        "window": window_index,
        "seed": seed,
        "config": config,
        "dit": dit.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


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


def save_local_checkpoint(out_dir: Path, dit, optimizer, ema, step: int, seed: int,
                          window_index: int, args):
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
        (full_path, build_full_payload(dit, optimizer, ema, step, seed, window_index, config)),
        (ema_path, build_ema_payload(ema, step, config)),
    ):
        tmp = out_dir / (path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
    print(f"[checkpoint] saved {full_path.name} + {ema_path.name}")
    prune_local_checkpoints(out_dir, LOCAL_KEEP)
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


def load_resume(path: Path, dit, optimizer, ema, args) -> int:
    """Resume a v5 checkpoint. Loaded to CPU first, so the GPU never holds two full copies."""
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    check_resume_config(data, expected=architecture_config(args))  # RB-11
    dit.load_state_dict(data["dit"], strict=True)
    if "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    if "ema" in data:
        ema.load_state_dict(data["ema"], strict=True)
    else:
        print("[resume] WARNING: checkpoint has no EMA weights; EMA restarts from the loaded DiT weights")
        ema.load_state_dict(dit.state_dict(), strict=True)
    step = int(data.get("step", 0))
    print(f"[resume] loaded {path.name} at step {step} (strict)")
    return step


def load_warm_start(path: Path, dit, ema) -> None:
    """Copy a v4 checkpoint's FFN weights into every expert (see dit_v5_moe.warm_start_from_v4)."""
    if not path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    v4_state = data["dit"] if "dit" in data else data
    warm_start_from_v4(dit, v4_state)
    ema.load_state_dict(dit.state_dict(), strict=True)
    print(f"[warm start] copied {path.name} (v4 FFN -> all experts); step counter starts at 0")


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
    parser.add_argument("--cache-repo", type=str, default=CACHE_DEFAULT_REPO)
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
    parser.add_argument("--capacity-factor", type=float, default=MOE_CAPACITY_FACTOR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("--steps must be >= 0, --batch-size > 0, --lr > 0")
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
    windows = make_windows(args.cache_window)
    needed = math.ceil(args.steps / WINDOW_STEPS)
    if needed > len(windows):
        raise ValueError(f"--steps {args.steps} needs {needed} windows of {WINDOW_STEPS} steps, "
                         f"but --cache-window {args.cache_window} gives only {len(windows)}")
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
    step = 0
    if args.resume:
        step = load_resume(Path(args.resume), dit, optimizer, ema, args)
    elif args.warm_start_from:
        load_warm_start(Path(args.warm_start_from), dit, ema)
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
    run_count = 0
    dropped_total = 0
    seen_total = 0
    while step < args.steps:
        w = step // WINDOW_STEPS
        window_end = min((w + 1) * WINDOW_STEPS, args.steps)
        if window is None:
            window = Window(w, windows[w], cache_dir, args.cache_repo)
            window.load()
            print(f"[window {w}] steps {step + 1}..{window_end}")
        while step < window_end:
            step += 1
            set_learning_rate(optimizer, args.lr, step)
            batch = sample_training_batch(window, replay, args.batch_size, device)
            stats, dropped = train_step(dit, ema, optimizer, batch, null_emb_full, null_mask_full, args, device)
            seen_total += args.batch_size
            dropped_total += dropped
            if stats is not None:
                run_total += stats["loss"]
                run_mse += stats["mse"]
                run_lb += stats["lb"]
                run_z += stats["z"]
                run_count += 1

            if step % LOG_EVERY == 0:
                n = max(run_count, 1)
                drift = ema_drift(ema, dit)
                dropped_frac = dropped_total / max(seen_total, 1)
                loss_out = run_total / n if run_count else float("nan")
                print(f"step={step} loss={loss_out:.4f} ema_drift={drift:.4f} dropped={dropped_frac:.3f}")
                print(f"[aux] step={step} mse={run_mse / n:.4f} lb={run_lb / n:.4f} z={run_z / n:.4f} "
                      f"route_dropped={dit.dropped_fraction():.4f}")
                run_total = run_mse = run_lb = run_z = 0.0
                run_count = 0
                dropped_total = 0
                seen_total = 0

            if step % CHECKPOINT_EVERY == 0 or step == args.steps:
                saved = save_local_checkpoint(out_dir, dit, optimizer, ema, step, args.seed, w, args)
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
