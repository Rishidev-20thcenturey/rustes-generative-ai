"""DiT v4 (100M @ 512) flow-matching training on the 200k MONET Qwen cache.

Rolling-window training:
  * The 20 cache chunks (10k samples each) are grouped into windows of
    --cache-window chunks (default 2 = ~20k samples): chunks 0-1, 2-3, ..., 18-19.
  * Each window trains for WINDOW_STEPS (50,000) steps. A replay buffer of
    --replay-buffer samples (default 5,000) drawn from earlier windows is mixed
    into every batch to limit forgetting.
  * Missing chunks are downloaded from --cache-repo (HF dataset) into
    --cache-path. When a window finishes its chunk files are deleted to free disk.

Checkpoints (LFS-safe policy):
  * Local: every 500 steps to --output-dir; only the newest 4 are kept.
  * --hf-repo: every 5,000 steps. Older checkpoints are removed with
    HfApi.delete_file so only the newest 4 remain on the main branch.
    Git history is not rewritten and nothing is force-pushed.
  * --milestone-repo: every 10,000 steps. Never deleted.

Recipe: LR 5e-5 (500-step warmup), grad clip 1.0, caption dropout 0.10, EMA 0.999.

The VAE (RAYVAE_v3) is loaded with strict=True and verified. Latents are
precomputed, so the loss does not use the VAE.
"""

import argparse
import copy
import gc
import logging
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, create_repo, hf_hub_download
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint

from .cfg import (
    NULL_PROMPT,
    apply_caption_dropout,
    ema_drift,
    encode_null_prompt,
    fit_null_to_length,
    load_or_build_null_prompt,
    update_ema,
)
from .dit_v4 import MODEL_DEPTH, MODEL_DIM, MODEL_HEADS, MODEL_PATCH, RAYDiT_v4
from .text_encoder import RAYTextEncoder
from .vae_v3 import RAYVAE_v3

# Quiet the 8-bit matmul chatter from bitsandbytes / transformers.
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
for _logger_name in ("bitsandbytes", "transformers"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

CACHE_DEFAULT_REPO = "Rishix500/monet-qwen-cache-200k-v4"
HF_DEFAULT_REPO = "Rishix500/ray-image-dit-v4-100m-512"
MILESTONE_DEFAULT_REPO = "Rishix500/ray-image-dit-v4-100m-512-milestones"
VAE_DEFAULT_REPO = "Rishix500/ray-image-vae-v3-512"
VAE_DEFAULT_CHECKPOINT = "ray_vae_v3_step_0030000.pt"
LOCAL_CHECKPOINT_DIR = "/kaggle/tmp/checkpoints"

IMAGE_SIZE = 512
LATENT_CHANNELS = 16
LATENT_SIZE = 64
TEXT_DIM = 2560
# Must match the cache (built with 128 text tokens); the null prompt uses the same length.
TEXT_MAX_TOKENS = 128
TOTAL_CACHE_CHUNKS = 20
CHUNK_SIZE = 10_000
WINDOW_STEPS = 50_000

CKPT_PREFIX = "ray_dit_v4_step_"          # full checkpoint: DiT + EMA + optimizer + step
EMA_CKPT_PREFIX = "ray_dit_v4_ema_step_"  # EMA-only weights under key "dit" (inference)
CKPT_PREFIXES = (CKPT_PREFIX, EMA_CKPT_PREFIX)
REPLAY_FILENAME = "replay_buffer.pt"
CHECKPOINT_EVERY = 500
LOCAL_KEEP = 4
HF_UPLOAD_EVERY = 5_000
HF_KEEP = 4
MILESTONE_EVERY = 10_000
LOG_EVERY = 100
WARMUP_STEPS = 500
TRAIN_LR = 5e-5
GRAD_CLIP = 1.0
DISK_REFUSE_SAVE_PCT = 90.0
STAGE_NAME = "dit_v4_100m_512"
LATENT_SPACE = "raw"

# Set in main() from the CLI (read by the checkpoint provenance block).
CAPTION_DROP = 0.10
EMA_DECAY = 0.999
VAE_CHECKPOINT = None
RUN_ID = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_dit() -> RAYDiT_v4:
    return RAYDiT_v4(latent_channels=LATENT_CHANNELS, latent_size=LATENT_SIZE, dim=MODEL_DIM,
                     depth=MODEL_DEPTH, heads=MODEL_HEADS, patch=MODEL_PATCH, cond_dim=TEXT_DIM,
                     cross_gate=True)


def checkpointed_forward(model, z_t, t, text, text_mask):
    b, c, h, w = z_t.shape
    if z_t.ndim != 4 or (c, h, w) != (model.latent_channels, model.latent_size, model.latent_size):
        raise ValueError(f"unexpected latent shape: {tuple(z_t.shape)}")
    x = model.in_proj(z_t)
    ph, pw = x.shape[-2:]
    x = x.flatten(2).transpose(1, 2)
    text = text.to(dtype=x.dtype)
    if text_mask is None:
        pooled = text.mean(dim=1)
    else:
        valid = (~text_mask).to(dtype=text.dtype).unsqueeze(-1)
        pooled = (text * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    cond = model.time(t).to(dtype=x.dtype) + model.text_proj(pooled)
    text_ctx = model.text_ctx_proj(text)

    def run_block(block_, x_, cond_, text_):
        return block_(x_, cond_, text_, text_mask=text_mask)

    for block in model.blocks:
        x = checkpoint(run_block, block, x, cond, text_ctx, use_reentrant=False)
    x = model.out(model.norm(x))
    x = x.view(b, ph, pw, model.patch, model.patch, model.latent_channels)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(b, model.latent_channels, h, w)


def download_vae_checkpoint(vae_repo: str, vae_checkpoint: str) -> Path:
    try:
        path = hf_hub_download(repo_id=vae_repo, filename=vae_checkpoint, repo_type="model",
                               token=os.getenv("HF_TOKEN"))
    except Exception as exc:
        raise RuntimeError(f"Could not download {vae_checkpoint} from {vae_repo}") from exc
    return Path(path)


def load_vae(device: torch.device, vae_repo: str, vae_checkpoint: str) -> RAYVAE_v3:
    checkpoint_path = download_vae_checkpoint(vae_repo, vae_checkpoint)
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vae = RAYVAE_v3(latent_channels=LATENT_CHANNELS, base_channels=64)
    # strict=True: a silent weight mismatch must fail here.
    vae.load_state_dict(checkpoint_data.get("vae", checkpoint_data), strict=True)
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"[VAE] strict-loaded {vae_repo}/{checkpoint_path.name}")
    return vae


def _encode_null_with_qwen() -> dict:
    encoder = RAYTextEncoder(model_name=RAYTextEncoder.MODEL_ID, max_tokens=TEXT_MAX_TOKENS,
                             load_in_8bit=True).eval()
    try:
        return encode_null_prompt(encoder, TEXT_MAX_TOKENS)
    finally:
        del encoder
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Cache windows and replay buffer
# --------------------------------------------------------------------------

def chunk_filename(index: int) -> str:
    start = index * CHUNK_SIZE
    end = start + CHUNK_SIZE
    return f"monet_qwen_cache_{start // 1000}k_{end // 1000}k.pt"


def make_windows(cache_window: int):
    indices = list(range(TOTAL_CACHE_CHUNKS))
    return [indices[i:i + cache_window] for i in range(0, len(indices), cache_window)]


def fetch_chunk(cache_dir: Path, cache_repo: str, index: int) -> Path:
    name = chunk_filename(index)
    local = cache_dir / name
    if local.is_file():
        return local
    print(f"[cache] downloading {name} from {cache_repo}")
    path = hf_hub_download(repo_id=cache_repo, filename=name, repo_type="dataset",
                           local_dir=str(cache_dir), token=os.getenv("HF_TOKEN"))
    return Path(path)


def load_chunk_payload(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError) as exc:
        print(f"[cache] WARNING: mmap load failed for {path.name} ({type(exc).__name__}: {exc}); "
              "falling back to a full in-memory load")
        payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in ("latents", "text_emb", "mask") if key not in payload]
    if missing:
        raise ValueError(f"cache missing keys in {path}: {missing}")
    latents, text_emb, mask = payload["latents"], payload["text_emb"], payload["mask"]
    if latents.ndim != 4 or tuple(latents.shape[1:]) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
        raise ValueError(f"invalid cache latent shape in {path}: {tuple(latents.shape)}")
    if text_emb.ndim != 3 or text_emb.shape[0] != latents.shape[0] or text_emb.shape[2] != TEXT_DIM:
        raise ValueError(f"invalid cache text_emb shape in {path}: {tuple(text_emb.shape)}")
    if mask.shape != text_emb.shape[:2]:
        raise ValueError(f"invalid cache mask shape in {path}: {tuple(mask.shape)}")
    if latents.shape[0] == 0:
        raise ValueError(f"cache is empty: {path}")
    return payload


def gather_from_payloads(payloads, offsets_t: torch.Tensor, global_idx: torch.Tensor, text_len: int):
    n = int(global_idx.numel())
    lat = torch.empty((n, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=torch.float16)
    txt = torch.empty((n, text_len, TEXT_DIM), dtype=torch.float16)
    msk = torch.empty((n, text_len), dtype=torch.bool)
    if n == 0:
        return lat, txt, msk
    chunk_ids = torch.bucketize(global_idx, offsets_t[1:-1], right=True)
    for cid in torch.unique(chunk_ids, sorted=True).tolist():
        pos = torch.nonzero(chunk_ids == cid, as_tuple=False).squeeze(1)
        local = global_idx[pos] - offsets_t[cid]
        payload = payloads[cid]
        lat[pos] = payload["latents"][local].to(torch.float16)
        txt[pos] = payload["text_emb"][local].to(torch.float16)
        msk[pos] = payload["mask"][local].bool()
    return lat, txt, msk


class Window:
    """The chunks of one training window, loaded for sampling."""

    def __init__(self, index: int, chunk_indices, cache_dir: Path, cache_repo: str):
        self.index = index
        self.chunk_indices = list(chunk_indices)
        self.cache_dir = cache_dir
        self.cache_repo = cache_repo
        self.paths = []
        self.payloads = []
        self.offsets_t = None
        self.count = 0
        self.text_len = None

    def load(self) -> None:
        for idx in self.chunk_indices:
            path = fetch_chunk(self.cache_dir, self.cache_repo, idx)
            self.payloads.append(load_chunk_payload(path))
            self.paths.append(path)
        text_lens = {int(p["text_emb"].shape[1]) for p in self.payloads}
        if len(text_lens) != 1:
            raise ValueError(f"window {self.index} chunks disagree on text length: {sorted(text_lens)}")
        self.text_len = text_lens.pop()
        offsets = [0]
        for payload in self.payloads:
            offsets.append(offsets[-1] + int(payload["latents"].shape[0]))
        self.offsets_t = torch.tensor(offsets, dtype=torch.long)
        self.count = offsets[-1]
        print(f"[window {self.index}] loaded chunks {self.chunk_indices} samples={self.count:,} "
              f"text_len={self.text_len}")

    def gather(self, global_idx: torch.Tensor):
        return gather_from_payloads(self.payloads, self.offsets_t, global_idx, self.text_len)

    def release(self) -> None:
        """Drop the memory maps, then delete this window's chunk files."""
        self.payloads = []
        gc.collect()
        for path in self.paths:
            try:
                path.unlink()
                print(f"[cache] deleted {path.name}")
            except FileNotFoundError:
                pass
        self.paths = []


class ReplayBuffer:
    """Samples from earlier windows, kept in CPU memory.

    At each window end, roughly half of the capacity is refreshed with new samples
    from the window that just finished; the rest is a random subset of the old
    buffer, so older windows stay represented.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.lat = None
        self.txt = None
        self.msk = None

    def __len__(self) -> int:
        return 0 if self.lat is None else int(self.lat.shape[0])

    def absorb(self, window: Window) -> None:
        if self.capacity <= 0 or window.count == 0:
            return
        if len(self) == 0:
            take_new = min(self.capacity, window.count)
            keep_old = 0
        else:
            take_new = min(self.capacity // 2, window.count)
            keep_old = min(self.capacity - take_new, len(self))
        new_idx = torch.randperm(window.count)[:take_new]
        new_lat, new_txt, new_msk = window.gather(new_idx)
        if keep_old > 0:
            old_sel = torch.randperm(len(self))[:keep_old]
            new_lat = torch.cat([self.lat[old_sel], new_lat])
            new_txt = torch.cat([self.txt[old_sel], new_txt])
            new_msk = torch.cat([self.msk[old_sel], new_msk])
        self.lat, self.txt, self.msk = new_lat, new_txt, new_msk
        print(f"[replay] size={len(self)} (capacity {self.capacity}) after window {window.index}")

    def sample(self, n: int):
        idx = torch.randint(0, len(self), (n,))
        return self.lat[idx], self.txt[idx], self.msk[idx]

    def save(self, path: Path) -> None:
        if len(self) == 0:
            return
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"lat": self.lat, "txt": self.txt, "msk": self.msk, "capacity": self.capacity}, tmp)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.lat, self.txt, self.msk = data["lat"], data["txt"], data["msk"]
        print(f"[replay] loaded {path.name}: size={len(self)}")


def sample_training_batch(window: Window, replay: ReplayBuffer, batch_size: int, device: torch.device):
    """Each sample comes from the replay buffer with probability len(replay) / (len(replay) + window)."""
    pool = len(replay)
    p_replay = pool / (pool + window.count) if pool else 0.0
    from_replay = torch.rand(batch_size) < p_replay
    n_replay = int(from_replay.sum())
    n_window = batch_size - n_replay
    lat = torch.empty((batch_size, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=torch.float16)
    txt = torch.empty((batch_size, window.text_len, TEXT_DIM), dtype=torch.float16)
    msk = torch.empty((batch_size, window.text_len), dtype=torch.bool)
    if n_window:
        w_lat, w_txt, w_msk = window.gather(torch.randint(0, window.count, (n_window,)))
        lat[~from_replay] = w_lat
        txt[~from_replay] = w_txt
        msk[~from_replay] = w_msk
    if n_replay:
        if replay.txt.shape[1] != window.text_len:
            raise ValueError("replay text length differs from the window text length")
        r_lat, r_txt, r_msk = replay.sample(n_replay)
        lat[from_replay] = r_lat
        txt[from_replay] = r_txt
        msk[from_replay] = r_msk
    return (lat.to(device=device, dtype=torch.float32, non_blocking=True),
            txt.to(device=device, dtype=torch.float16, non_blocking=True),
            msk.to(device=device, dtype=torch.bool, non_blocking=True))


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------

def set_learning_rate(optimizer, base_lr: float, step: int) -> float:
    lr = base_lr * step / WARMUP_STEPS if step <= WARMUP_STEPS else base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_step(dit, ema, optimizer, batch, null_emb_full, null_mask_full, caption_drop: float,
               ema_decay: float, device: torch.device):
    """One optimiser step. Returns (loss or None if skipped, number of dropped captions)."""
    clean_latents, text_hidden, text_mask = batch
    null_emb_fit, null_mask_fit = fit_null_to_length(null_emb_full, null_mask_full, text_hidden.shape[1])
    text_hidden, text_mask, dropped = apply_caption_dropout(
        text_hidden, text_mask, null_emb_fit, null_mask_fit, caption_drop,
    )
    noise = torch.randn_like(clean_latents)
    t = torch.rand(clean_latents.shape[0], device=device)
    t_view = t[:, None, None, None]
    z_t = (1.0 - t_view) * clean_latents + t_view * noise
    target_velocity = noise - clean_latents
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        prediction = checkpointed_forward(dit, z_t, t, text_hidden, text_mask)
        loss = F.mse_loss(prediction.float(), target_velocity.float())
    if not torch.isfinite(loss).item():
        optimizer.zero_grad(set_to_none=True)
        return None, int(dropped)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm).item():
        optimizer.zero_grad(set_to_none=True)
        return None, int(dropped)
    optimizer.step()
    update_ema(ema, dit, ema_decay)
    return float(loss.item()), int(dropped)


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

def log_disk_usage(path: Path, prefix: str = "[disk]") -> float:
    usage = shutil.disk_usage(path)
    percent = (usage.used / usage.total) * 100.0 if usage.total else 100.0
    print(f"{prefix} used={usage.used / (1024 ** 3):.2f}GB free={usage.free / (1024 ** 3):.2f}GB "
          f"total={usage.total / (1024 ** 3):.2f}GB ({percent:.1f}%)")
    return percent


def _generation_config() -> dict:
    """RB-11: the architecture values a resume must match exactly."""
    return {
        "image_size": IMAGE_SIZE,
        "latent_channels": LATENT_CHANNELS,
        "latent_size": LATENT_SIZE,
        "dim": MODEL_DIM,
        "depth": MODEL_DEPTH,
        "heads": MODEL_HEADS,
        "patch": MODEL_PATCH,
        "text_dim": TEXT_DIM,
    }


def check_resume_config(checkpoint_data, expected=None):
    """RB-11: raise on an architecture mismatch, warn on missing keys."""
    expected = dict(_generation_config() if expected is None else expected)
    config = checkpoint_data.get("config")
    if not isinstance(config, dict):
        print("[resume] WARNING: checkpoint has no 'config' dict; skipping the consistency check")
        return []
    mismatches = []
    missing = []
    for key, want in expected.items():
        if key not in config:
            missing.append(key)
        elif config[key] != want:
            mismatches.append(f"{key}: checkpoint={config[key]!r} current={want!r}")
    if mismatches:
        raise RuntimeError("[resume] refusing to resume: checkpoint/training config mismatch -> "
                           + "; ".join(mismatches))
    if missing:
        print(f"[resume] WARNING: checkpoint config is missing {missing}; assuming they match")
    if config.get("latent_space") != LATENT_SPACE:
        print(f"[resume] WARNING: checkpoint latent_space={config.get('latent_space')!r} "
              f"but this trainer optimises {LATENT_SPACE!r}")
    return missing


def checkpoint_has_corruption(model) -> bool:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                print(f"[checkpoint guard] non-finite weight in {name}; refusing to save")
                return True
    return False


def _step_of(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[1])


def build_config(args) -> dict:
    config = dict(_generation_config())
    config.update({
        "flow_matching": "linear_gaussian",
        "cross_gate": True,
        "text_max_tokens": TEXT_MAX_TOKENS,
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


def build_full_payload(dit, optimizer, ema, step: int, seed: int, window_index: int, config: dict) -> dict:
    """Full checkpoint: DiT weights, EMA weights, optimizer state, step number."""
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
    """EMA-only file for inference. The weights are under the key "dit", as generate.py expects."""
    return {
        "stage": STAGE_NAME,
        "step": step,
        "config": config,
        "dit": ema.state_dict(),
    }


def prune_local_checkpoints(out_dir: Path, keep: int) -> None:
    """Keep the newest `keep` files of EACH kind (full and EMA-only)."""
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
    """Upload both files for a step to the main branch, then keep only the newest HF_KEEP of each kind.

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


def load_resume(path: Path, dit, optimizer, ema, device: torch.device) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    data = torch.load(path, map_location=device, weights_only=False)
    check_resume_config(data)  # RB-11
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train RAY-IMAGE DiT v4 (100M @ 512) on the rolling-window cache.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=str, required=True,
                        help="Local directory for cache chunks. Missing chunks are downloaded here.")
    parser.add_argument("--cache-repo", type=str, default=CACHE_DEFAULT_REPO,
                        help="HF dataset repo that holds monet_qwen_cache_*.pt chunks.")
    parser.add_argument("--cache-window", type=int, default=2,
                        help="Number of 10k-sample chunks loaded per training window.")
    parser.add_argument("--replay-buffer", type=int, default=5000,
                        help="Max samples kept from earlier windows and mixed into training batches.")
    parser.add_argument("--vae-repo", type=str, default=VAE_DEFAULT_REPO)
    parser.add_argument("--vae-checkpoint", type=str, default=VAE_DEFAULT_CHECKPOINT)
    parser.add_argument("--hf-repo", type=str, default=HF_DEFAULT_REPO)
    parser.add_argument("--milestone-repo", type=str, default=MILESTONE_DEFAULT_REPO)
    parser.add_argument("--resume", type=str, default=None,
                        help="Local checkpoint (.pt) to resume from. Loaded with strict=True.")
    parser.add_argument("--output-dir", type=str, default=LOCAL_CHECKPOINT_DIR)
    parser.add_argument("--no-hf-upload", action="store_true")
    parser.add_argument("--caption-drop", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
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
    windows = make_windows(args.cache_window)
    needed = math.ceil(args.steps / WINDOW_STEPS)
    if needed > len(windows):
        raise ValueError(f"--steps {args.steps} needs {needed} windows of {WINDOW_STEPS} steps, "
                         f"but --cache-window {args.cache_window} gives only {len(windows)}")
    if not torch.cuda.is_available():
        raise RuntimeError("DiT v4 training requires CUDA")

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

    dit = build_dit().to(device)
    optimizer = AdamW(dit.parameters(), lr=0.0, betas=(0.9, 0.99), weight_decay=0.01)
    ema = copy.deepcopy(dit)
    ema.requires_grad_(False)
    ema.eval()
    step = 0
    if args.resume:
        step = load_resume(Path(args.resume), dit, optimizer, ema, device)
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

    print(f"[config] batch={args.batch_size} lr={args.lr:.2e} warmup={WARMUP_STEPS} target_steps={args.steps} "
          f"window_steps={WINDOW_STEPS} cache_window={args.cache_window} replay_buffer={args.replay_buffer}")
    print(f"[cfg] caption_drop={args.caption_drop:.3f} null_tokens={null_emb_full.shape[0]} ema_decay={args.ema_decay}")
    print(f"[DiT v4] trainable parameters: {sum(p.numel() for p in dit.parameters()):,}")
    log_disk_usage(out_dir)

    window = None
    run_loss = 0.0
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
            loss_value, dropped = train_step(dit, ema, optimizer, batch, null_emb_full, null_mask_full,
                                             args.caption_drop, args.ema_decay, device)
            seen_total += args.batch_size
            dropped_total += dropped
            if loss_value is not None:
                run_loss += loss_value
                run_count += 1

            if step % LOG_EVERY == 0:
                mean_loss = run_loss / run_count if run_count else float("nan")
                drift = ema_drift(ema, dit)
                dropped_frac = dropped_total / max(seen_total, 1)
                print(f"step={step} loss={mean_loss:.4f} ema_drift={drift:.4f} dropped={dropped_frac:.3f}")
                run_loss = 0.0
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

        # Window finished: refresh the replay buffer from it, save it, then free its chunk files.
        replay.absorb(window)
        replay.save(replay_path)
        window.release()
        window = None
        log_disk_usage(out_dir)

    print(f"[done] reached step={step}")
    del vae


if __name__ == "__main__":
    main()
