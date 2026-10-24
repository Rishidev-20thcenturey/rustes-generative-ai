"""DiT v4 (100M @ 512) flow-matching training on streamed MONET or a precomputed cache.

Copied from train_dit_v3.py. Uses RAYDiT_v4 (16x64x64 latents, patch=4, 256 tokens)
and RAYVAE_v3 (512px). Training recipe is unchanged: LR 5e-5, grad clip 1.0,
caption dropout 0.10, EMA 0.999.

With --cache-path, frozen VAE/Qwen inference is bypassed and cached latent/text
embeddings are sampled directly from a 64x64-latent MONET cache. A 32x32 (256px)
cache is rejected by the loader.
"""

import argparse
import copy
import io
import os
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

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

MONET_REPO = "jasperai/monet"
# The VAE checkpoint is a CLI input (--vae-repo / --vae-checkpoint). No v3 VAE
# checkpoint is bundled, so there is no default filename.
# New repos for v4 so the shipped 86M v3 repos are never written or cleaned up.
HF_DEFAULT_REPO = "Rishix500/ray-image-dit-v4-100m-512"
MILESTONE_REPO = "Rishix500/ray-image-dit-v4-100m-512-milestones"
MILESTONE_INTERVAL = 10000
IMAGE_SIZE = 512
LATENT_CHANNELS = 16
LATENT_SIZE = 64
CKPT_PREFIX = "ray_dit_v4_step_"
EMA_CHECKPOINT_PREFIX = "ray_dit_v4_ema_step_"
STAGE_NAME = "dit_v4_100m_512"
VAE_CHECKPOINT = None  # set in main() from --vae-checkpoint
TEXT_DIM = 2560
TEXT_MAX_TOKENS = 256
CHECKPOINT_EVERY = 500
LOG_EVERY = 100
WARMUP_STEPS = 500
HF_CHECKPOINT_WINDOW = 4
LOCAL_CHECKPOINT_WINDOW = 2
LOCAL_CHECKPOINT_DIR = "/kaggle/tmp/checkpoints"
DISK_REFUSE_SAVE_PCT = 90.0
TRAIN_LR = 5e-5
GRAD_CLIP = 1.0
CACHE_CHUNK_RE = re.compile(r"^monet_qwen_cache_(\d+)k_(\d+)k\.pt$")

# RB-03: the latent space this trainer actually optimises in. The v4 trainer
# feeds raw VAE latents (no channel whitening), so checkpoints declare "raw".
LATENT_SPACE = "raw"

# RB-11: keys that must match exactly when resuming; anything else warns only.
GENERATION_CONFIG_KEYS = (
    "image_size",
    "latent_channels",
    "latent_size",
    "dim",
    "depth",
    "heads",
    "patch",
    "text_dim",
)

# RB-03: populated once per process in main() so every checkpoint records which
# run produced it.
RUN_ID = None

# Issue #7 (RB-10): CFG settings, populated in main() and recorded in checkpoints.
CAPTION_DROP = 0.10
EMA_DECAY = 0.999


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_thumbnail(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pick_caption(sample) -> str:
    for key in (
        "caption_gemini-2.5-flash-lite",
        "caption_internvl-3-8b",
        "caption_sharegpt4v-7b",
        "caption_florence-2-large",
        "caption_original",
    ):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def next_batch(iterator, batch_size: int):
    images, captions = [], []
    while len(images) < batch_size:
        sample = next(iterator)
        image = sample.get("thumbnail")
        if image is None:
            continue
        try:
            images.append(preprocess_thumbnail(image))
            captions.append(_pick_caption(sample))
        except (TypeError, ValueError):
            continue
    return torch.stack(images), captions


def _load_cache_payload(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError) as exc:
        # RB-12: surface the fallback instead of silently loading a >2 GB file
        # fully into RAM.
        print(
            f"[cache] WARNING: mmap load failed for {path} ({type(exc).__name__}: {exc}); "
            "falling back to a full in-memory load"
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)

    required = ("latents", "text_emb", "mask")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"cache missing keys in {path}: {missing}")
    latents = payload["latents"]
    text_emb = payload["text_emb"]
    mask = payload["mask"]
    if latents.ndim != 4 or tuple(latents.shape[1:]) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
        raise ValueError(f"invalid cache latent shape in {path}: {tuple(latents.shape)}")
    if text_emb.ndim != 3 or text_emb.shape[0] != latents.shape[0] or text_emb.shape[2] != TEXT_DIM:
        raise ValueError(f"invalid cache text_emb shape in {path}: {tuple(text_emb.shape)}")
    if mask.shape != text_emb.shape[:2]:
        raise ValueError(f"invalid cache mask shape in {path}: {tuple(mask.shape)}")
    if latents.shape[0] == 0:
        raise ValueError(f"cache is empty: {path}")
    return payload


def _chunk_sort_key(path: Path):
    match = CACHE_CHUNK_RE.match(path.name)
    if not match:
        raise ValueError(f"invalid cache chunk filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_cache(path: Path):
    """Load one cache file or a directory of ordered MONET cache chunks."""
    path = Path(path)
    if path.is_file():
        payload = _load_cache_payload(path)
        latents = payload["latents"]
        payload["text_len"] = int(payload["text_emb"].shape[1])
        print(f"[cache] loaded {path} samples={latents.shape[0]:,} latent={tuple(latents.shape[1:])} text={tuple(payload['text_emb'].shape[1:])}")
        return payload
    if not path.is_dir():
        raise FileNotFoundError(f"cache not found: {path}")

    chunk_paths = sorted(path.glob("monet_qwen_cache_*.pt"), key=_chunk_sort_key)
    if not chunk_paths:
        raise FileNotFoundError(f"no cache chunks found in directory: {path}")

    chunks = []
    previous_end = None
    total_samples = 0
    for chunk_path in chunk_paths:
        start_name, end_name = _chunk_sort_key(chunk_path)
        payload = _load_cache_payload(chunk_path)
        source_start = int(payload.get("source_start", start_name * 1000))
        source_end = int(payload.get("source_end", end_name * 1000))
        if source_start != start_name * 1000 or source_end != end_name * 1000:
            raise ValueError(f"cache chunk metadata mismatch: {chunk_path.name} filename={start_name * 1000}:{end_name * 1000} payload={source_start}:{source_end}")
        if source_end <= source_start:
            raise ValueError(f"invalid source range in {chunk_path}: {source_start}:{source_end}")
        if previous_end is not None and source_start < previous_end:
            raise ValueError(f"duplicate/overlapping cache source range: previous_end={previous_end}, current={source_start}:{source_end}")
        if previous_end is not None and source_start != previous_end:
            raise ValueError(f"cache chunks are not contiguous: previous_end={previous_end}, current_start={source_start}")
        total_samples += int(payload["latents"].shape[0])
        previous_end = source_end
        chunks.append(payload)

    text_lengths = {int(payload["text_emb"].shape[1]) for payload in chunks}
    if len(text_lengths) != 1:
        raise ValueError(
            f"cache chunks disagree on text sequence length: {sorted(text_lengths)}; "
            "rebuild the cache so every chunk shares one length"
        )
    text_len = text_lengths.pop()

    offsets = [0]
    for payload in chunks:
        offsets.append(offsets[-1] + int(payload["latents"].shape[0]))
    cache = {"chunks": chunks, "offsets": torch.tensor(offsets, dtype=torch.long), "count": total_samples,
             "text_len": text_len}
    print(f"[cache] loaded directory {path} chunks={len(chunks)} total_samples={total_samples:,}")
    print(f"[cache] source ranges verified: {chunks[0].get('source_start', 0)}:{chunks[-1].get('source_end', total_samples)}; no duplicate/overlapping ranges")
    return cache


def next_cached_batch(cache, batch_size: int, device: torch.device):
    if "chunks" not in cache:
        count = cache["latents"].shape[0]
        indices = torch.randint(0, count, (batch_size,))
        clean_latents = cache["latents"][indices].to(device=device, dtype=torch.float32, non_blocking=True)
        text_hidden = cache["text_emb"][indices].to(device=device, dtype=torch.float16, non_blocking=True)
        text_mask = cache["mask"][indices].to(device=device, dtype=torch.bool, non_blocking=True)
        return clean_latents, text_hidden, text_mask

    global_indices = torch.randint(0, int(cache["count"]), (batch_size,))
    offsets = cache["offsets"]
    chunk_ids = torch.bucketize(global_indices, offsets[1:-1], right=True)
    # Sequence length comes from the cache payload, never from a hardcoded value.
    text_len = int(cache["text_len"])
    clean_latents = torch.empty((batch_size, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=torch.float32)
    text_hidden = torch.empty((batch_size, text_len, TEXT_DIM), dtype=torch.float16)
    text_mask = torch.empty((batch_size, text_len), dtype=torch.bool)
    for chunk_id in torch.unique(chunk_ids, sorted=True).tolist():
        positions = torch.nonzero(chunk_ids == chunk_id, as_tuple=False).squeeze(1)
        local_indices = global_indices[positions] - offsets[chunk_id]
        chunk = cache["chunks"][chunk_id]
        clean_latents[positions] = chunk["latents"][local_indices].float()
        text_hidden[positions] = chunk["text_emb"][local_indices].half()
        text_mask[positions] = chunk["mask"][local_indices].bool()
    return clean_latents.to(device=device, non_blocking=True), text_hidden.to(device=device, non_blocking=True), text_mask.to(device=device, non_blocking=True)


def download_vae_checkpoint(vae_repo: str, vae_checkpoint: str) -> Path:
    token = os.getenv("HF_TOKEN")
    try:
        path = hf_hub_download(repo_id=vae_repo, filename=vae_checkpoint, repo_type="model", token=token)
    except Exception as exc:
        raise RuntimeError(f"Could not download {vae_checkpoint} from {vae_repo}") from exc
    return Path(path)


def load_vae(device: torch.device, vae_repo: str, vae_checkpoint: str) -> RAYVAE_v3:
    checkpoint_path = download_vae_checkpoint(vae_repo, vae_checkpoint)
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vae = RAYVAE_v3(latent_channels=LATENT_CHANNELS, base_channels=64)
    # strict=True: a silent weight mismatch would train against the wrong VAE.
    vae.load_state_dict(checkpoint_data.get("vae", checkpoint_data), strict=True)
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"[VAE] loaded {checkpoint_path.name}")
    return vae


def build_dit() -> RAYDiT_v4:
    return RAYDiT_v4(latent_channels=LATENT_CHANNELS, latent_size=LATENT_SIZE, dim=MODEL_DIM, depth=MODEL_DEPTH, heads=MODEL_HEADS, patch=MODEL_PATCH, cond_dim=TEXT_DIM, cross_gate=True)


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


def _checkpoint_sort_key(filename: str):
    try:
        return int(Path(filename).stem.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return -1


def cleanup_hf_checkpoints(api: HfApi, repo_id: str) -> None:
    # Issue #7: resume checkpoints and EMA weights are windowed separately so an
    # EMA file is never evicted by the resume-checkpoint window (or vice versa).
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        # Only ray_dit_v4_* names are listed or deleted, so the v3 repo is never touched.
        for prefix in (CKPT_PREFIX, EMA_CHECKPOINT_PREFIX):
            checkpoints = sorted({p for p in files if Path(p).name.startswith(prefix) and Path(p).suffix == ".pt"}, key=_checkpoint_sort_key)
            for path in checkpoints[:-HF_CHECKPOINT_WINDOW]:
                try:
                    api.delete_file(path_in_repo=path, repo_id=repo_id, repo_type="model", commit_message=f"chore: remove old DiT v3 checkpoint {Path(path).name}")
                    print(f"[HF cleanup] deleted {path}")
                except Exception as exc:
                    print(f"[HF cleanup] failed to delete {path}: {exc}")
    except Exception as exc:
        print(f"[HF cleanup] listing failed for {repo_id}: {exc}")


def log_disk_usage(path: Path, prefix: str = "[disk]") -> float:
    usage = shutil.disk_usage(path)
    percent = (usage.used / usage.total) * 100.0 if usage.total else 100.0
    print(f"{prefix} used={usage.used / (1024 ** 3):.2f}GB free={usage.free / (1024 ** 3):.2f}GB total={usage.total / (1024 ** 3):.2f}GB ({percent:.1f}%)")
    return percent


def _local_checkpoint_paths(output_dir: Path):
    return sorted((p for p in output_dir.glob(f"{CKPT_PREFIX}*.pt") if p.is_file()), key=lambda p: _checkpoint_sort_key(p.name))


def cleanup_local_checkpoints(output_dir: Path, keep: int = LOCAL_CHECKPOINT_WINDOW) -> None:
    checkpoints = _local_checkpoint_paths(output_dir)
    for path in checkpoints[:-keep] if keep > 0 else checkpoints:
        try:
            path.unlink()
            print(f"[local cleanup] deleted {path}")
        except OSError as exc:
            print(f"[local cleanup] failed to delete {path}: {exc}")


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
    """RB-11: raise on an architecture mismatch, warn on missing keys.

    Returns the list of keys that were absent from the checkpoint config.
    """
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
        raise RuntimeError(
            "[resume] refusing to resume: checkpoint/training config mismatch -> "
            + "; ".join(mismatches)
        )
    if missing:
        print(f"[resume] WARNING: checkpoint config is missing {missing}; assuming they match")
    if config.get("latent_space") is None:
        print("[resume] WARNING: checkpoint does not declare latent_space (pre-RB-03 checkpoint)")
    elif config.get("latent_space") != LATENT_SPACE:
        print(
            f"[resume] WARNING: checkpoint latent_space={config.get('latent_space')!r} "
            f"but this trainer optimises {LATENT_SPACE!r}"
        )
    return missing


def _checkpoint_payload(model, optimizer, step, seed, ema=None):
    payload = {
        "stage": STAGE_NAME,
        "step": step,
        "seed": seed,
        "config": {
            "image_size": IMAGE_SIZE,
            "latent_channels": LATENT_CHANNELS,
            "latent_size": LATENT_SIZE,
            "dim": MODEL_DIM,
            "depth": MODEL_DEPTH,
            "heads": MODEL_HEADS,
            "patch": MODEL_PATCH,
            "text_dim": TEXT_DIM,
            "text_max_tokens": TEXT_MAX_TOKENS,
            "cross_gate": True,
            "flow_matching": "linear_gaussian",
            # RB-03: provenance so a later run (or generate.py) can tell exactly
            # what this checkpoint was trained against.
            "latent_space": LATENT_SPACE,
            "text_cap": TEXT_MAX_TOKENS,
            "vae_checkpoint": VAE_CHECKPOINT,
            "torch": torch.__version__,
            "run": RUN_ID,
            # Issue #7: CFG provenance (new keys only).
            "caption_drop": CAPTION_DROP,
            "ema_decay": EMA_DECAY,
            "null_prompt": NULL_PROMPT,
        },
        "dit": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    return payload


def upload_checkpoint(local_path: Path, repo_id: str) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[HF] HF_TOKEN not set; keeping checkpoint local only")
        return
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=local_path.name, repo_id=repo_id, repo_type="model")
    print(f"[HF] uploaded {local_path.name} -> {repo_id}")
    cleanup_hf_checkpoints(api, repo_id)


def _ema_checkpoint_payload(ema_model, step: int, config: dict) -> dict:
    """EMA weights under key 'dit' (v4 stage tag; same layout as the v3 EMA file)."""
    return {"stage": f"{STAGE_NAME}_ema", "step": step, "config": dict(config), "dit": ema_model.state_dict()}


def save_ema_checkpoint(path: Path, ema_model, step: int, config: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_ema_checkpoint_payload(ema_model, step, config), path)
    return path


def _ema_file_config() -> dict:
    """Config recorded in EMA weight files (architecture + CFG provenance)."""
    config = dict(_generation_config())
    config.update({
        "text_max_tokens": TEXT_MAX_TOKENS,
        "cross_gate": True,
        "latent_space": LATENT_SPACE,
        "vae_checkpoint": VAE_CHECKPOINT,
        "run": RUN_ID,
        "caption_drop": CAPTION_DROP,
        "ema_decay": EMA_DECAY,
        "null_prompt": NULL_PROMPT,
    })
    return config


def upload_checkpoint_direct(model, optimizer, step: int, seed: int, repo_id: str, ema=None) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for direct checkpoint upload when disk is near full")
    payload = _checkpoint_payload(model, optimizer, step, seed, ema=ema)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    filename = f"{CKPT_PREFIX}{step:07d}.pt"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=buffer, path_in_repo=filename, repo_id=repo_id, repo_type="model", commit_message=f"upload DiT v4 checkpoint {filename}")
    print(f"[HF] uploaded {filename} directly from memory (local save skipped)")
    buffer.close()
    if ema is not None:
        ema_filename = f"{EMA_CHECKPOINT_PREFIX}{step:07d}.pt"
        ema_buffer = io.BytesIO()
        torch.save(_ema_checkpoint_payload(ema, step, _ema_file_config()), ema_buffer)
        ema_buffer.seek(0)
        api.upload_file(path_or_fileobj=ema_buffer, path_in_repo=ema_filename, repo_id=repo_id, repo_type="model", commit_message=f"upload DiT v4 EMA weights {ema_filename}")
        print(f"[HF] uploaded {ema_filename} directly from memory (local save skipped)")
        ema_buffer.close()
    cleanup_hf_checkpoints(api, repo_id)


def checkpoint_has_corruption(model: RAYDiT_v4) -> bool:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                print(f"[checkpoint guard] non-finite weight in {name}; refusing to save")
                return True
    return False


def save_checkpoint(path, model, optimizer, step, seed, ema=None) -> bool:
    if checkpoint_has_corruption(model):
        print(f"[checkpoint guard] SKIP step={step}: corrupted model weights")
        return False
    torch.save(_checkpoint_payload(model, optimizer, step, seed, ema=ema), path)
    return True


def load_resume(path, model, optimizer, device, expected_config=None) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    checkpoint_data = torch.load(path, map_location=device, weights_only=False)
    check_resume_config(checkpoint_data, expected=expected_config)  # RB-11
    model.load_state_dict(checkpoint_data["dit"], strict=True)
    if "optimizer" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer"])
    step = int(checkpoint_data.get("step", 0))
    print(f"[resume] loaded {path} at step {step}")
    return step


def set_learning_rate(optimizer, base_lr, step) -> float:
    lr = base_lr * step / WARMUP_STEPS if step <= WARMUP_STEPS else base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def cleanup_local_ema(output_dir: Path, keep: int = LOCAL_CHECKPOINT_WINDOW) -> None:
    """Issue #7: same window policy as cleanup_local_checkpoints, for EMA files."""
    paths = sorted((p for p in output_dir.glob(f"{EMA_CHECKPOINT_PREFIX}*.pt") if p.is_file()), key=lambda p: _checkpoint_sort_key(p.name))
    for path in paths[:-keep] if keep > 0 else paths:
        try:
            path.unlink()
            print(f"[local cleanup] deleted {path}")
        except OSError as exc:
            print(f"[local cleanup] failed to delete {path}: {exc}")


def save_or_upload_checkpoint(output_dir: Path, dit, optimizer, step: int, seed: int, hf_repo: str, no_hf_upload: bool, ema=None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_disk_pct = log_disk_usage(output_dir, prefix=f"[disk] checkpoint step={step}")
    checkpoint_path = output_dir / f"{CKPT_PREFIX}{step:07d}.pt"
    ema_path = output_dir / f"{EMA_CHECKPOINT_PREFIX}{step:07d}.pt"

    if local_disk_pct >= DISK_REFUSE_SAVE_PCT:
        print(f"[checkpoint] local disk usage {local_disk_pct:.1f}% >= {DISK_REFUSE_SAVE_PCT:.1f}%; refusing local save")
        if no_hf_upload:
            print("[checkpoint] HF upload disabled; checkpoint skipped")
            return
        if checkpoint_has_corruption(dit):
            print(f"[checkpoint guard] SKIP step={step}: corrupted model weights")
            return
        upload_checkpoint_direct(dit, optimizer, step, seed, hf_repo, ema=ema)
        return

    try:
        if save_checkpoint(checkpoint_path, dit, optimizer, step, seed, ema=ema):
            print(f"[checkpoint] {checkpoint_path}")
            if ema is not None:
                save_ema_checkpoint(ema_path, ema, step, _ema_file_config())
                print(f"[checkpoint] {ema_path}")
            if not no_hf_upload:
                upload_checkpoint(checkpoint_path, hf_repo)
                if ema is not None:
                    upload_checkpoint(ema_path, hf_repo)
            cleanup_local_checkpoints(output_dir, keep=LOCAL_CHECKPOINT_WINDOW)
            if ema is not None:
                cleanup_local_ema(output_dir, keep=LOCAL_CHECKPOINT_WINDOW)
            if step > 0 and step % MILESTONE_INTERVAL == 0:
                print(f"\n💎 [milestone] Step {step} reached. Archiving to {MILESTONE_REPO}...")
                token = os.getenv("HF_TOKEN")
                if not token:
                    raise RuntimeError("HF_TOKEN is required for milestone archive upload")
                api = HfApi(token=token)
                api.create_repo(repo_id=MILESTONE_REPO, repo_type="model", private=True, exist_ok=True)
                api.upload_file(
                    path_or_fileobj=str(checkpoint_path),
                    path_in_repo=f"{CKPT_PREFIX}{step:07d}.pt",
                    repo_id=MILESTONE_REPO,
                    repo_type="model",
                )
                print(f"✅ [milestone] Archived step {step}")
            log_disk_usage(output_dir, prefix=f"[disk] after checkpoint step={step}")
    except (RuntimeError, OSError) as exc:
        print(f"[checkpoint] local save failed at step={step}: {exc}")
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except OSError:
                pass
        if ema_path.exists():
            try:
                ema_path.unlink()
            except OSError:
                pass
        if no_hf_upload:
            raise
        if checkpoint_has_corruption(dit):
            print(f"[checkpoint guard] SKIP step={step}: corrupted model weights")
            return
        upload_checkpoint_direct(dit, optimizer, step, seed, hf_repo, ema=ema)
        cleanup_local_checkpoints(output_dir, keep=LOCAL_CHECKPOINT_WINDOW)
        if ema is not None:
            cleanup_local_ema(output_dir, keep=LOCAL_CHECKPOINT_WINDOW)


def _encode_null_with_qwen() -> dict:
    """Issue #7: build the null prompt with a temporary Qwen encoder (cache mode)."""
    encoder = RAYTextEncoder(model_name=RAYTextEncoder.MODEL_ID, max_tokens=TEXT_MAX_TOKENS, load_in_8bit=True).eval()
    try:
        return encode_null_prompt(encoder, TEXT_MAX_TOKENS)
    finally:
        del encoder
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Train RAY-IMAGE DiT v4 (100M @ 512) with flow matching.")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=LOCAL_CHECKPOINT_DIR)
    parser.add_argument("--hf-repo", type=str, default=HF_DEFAULT_REPO)
    parser.add_argument("--no-hf-upload", action="store_true")
    parser.add_argument("--vae-repo", type=str, default=None,
                        help="HF repo holding the RAYVAE_v3 checkpoint (required without --cache-path).")
    parser.add_argument("--vae-checkpoint", type=str, default=None,
                        help="Filename of the RAYVAE_v3 checkpoint inside --vae-repo (required without --cache-path).")
    parser.add_argument("--caption-drop", type=float, default=0.10,
                        help="Issue #7: per-sample probability of replacing the caption with the null prompt.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="Issue #7: EMA decay; ema = decay * ema + (1 - decay) * model after each step.")
    args = parser.parse_args()
    if args.steps < 0 or args.batch_size <= 0 or args.lr <= 0 or args.start_step < 0:
        raise ValueError("invalid training arguments")
    if not 0.0 <= args.caption_drop <= 1.0:
        raise ValueError("--caption-drop must be in [0, 1]")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    global CAPTION_DROP, EMA_DECAY, VAE_CHECKPOINT
    CAPTION_DROP = args.caption_drop
    EMA_DECAY = args.ema_decay
    if args.cache_path is None and not (args.vae_repo and args.vae_checkpoint):
        raise ValueError("live training needs --vae-repo and --vae-checkpoint (no v3 VAE checkpoint is bundled)")
    VAE_CHECKPOINT = args.vae_checkpoint
    if args.resume and args.start_step:
        raise ValueError("Use either --resume or --start-step, not both")
    if not torch.cuda.is_available():
        raise RuntimeError("DiT v4 training requires CUDA")
    global RUN_ID
    RUN_ID = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-seed{args.seed}"
    seed_everything(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = None
    dataset = None
    iterator = None
    vae = None
    text_encoder = None
    if args.cache_path:
        cache = load_cache(Path(args.cache_path))
        print("[data] using precomputed cache; live VAE/Qwen inference disabled")
    else:
        dataset = load_dataset(MONET_REPO, name="parquet", split="train", streaming=True)
        iterator = iter(dataset)
        vae = load_vae(device, args.vae_repo, args.vae_checkpoint)
        text_encoder = RAYTextEncoder(model_name=RAYTextEncoder.MODEL_ID, max_tokens=TEXT_MAX_TOKENS, load_in_8bit=True).eval()
        print("[data] using live streamed MONET + frozen VAE + Qwen")
    dit = build_dit().to(device)
    optimizer = AdamW(dit.parameters(), lr=0.0, betas=(0.9, 0.99), weight_decay=0.01)
    last_step = args.start_step
    if args.resume:
        last_step = load_resume(Path(args.resume), dit, optimizer, device)
    if args.steps <= last_step:
        print(f"[done] target steps={args.steps} already reached at step={last_step}")
        return
    # Issue #7: null placeholder embedding. Fetched from HF on later sessions;
    # rebuilt once (with the loaded encoder, or a temporary Qwen in cache mode)
    # and uploaded to a private repo only if the fetch fails.
    if text_encoder is not None:
        null_build_fn = lambda: encode_null_prompt(text_encoder, TEXT_MAX_TOKENS)
    else:
        null_build_fn = _encode_null_with_qwen
    null_payload = load_or_build_null_prompt(
        args.hf_repo, text_dim=TEXT_DIM, build_fn=null_build_fn, upload=not args.no_hf_upload,
    )
    null_emb_full = null_payload["emb"].to(device=device, dtype=torch.float16)
    null_mask_full = null_payload["mask"].to(device=device, dtype=torch.bool)
    # Issue #7: EMA of the DiT weights. Resume does not restore EMA state (the
    # resume logic is unchanged), so EMA starts from the current weights.
    ema = copy.deepcopy(dit)
    ema.requires_grad_(False)
    ema.eval()
    print(f"[cfg] caption_drop={args.caption_drop:.3f} null_prompt={NULL_PROMPT!r} null_tokens={null_emb_full.shape[0]} ema_decay={args.ema_decay}")
    if last_step > 0:
        print("[ema] initialised from current weights at start/resume step (EMA state is not restored from checkpoint)")
    print(f"[DiT v4] trainable parameters: {sum(p.numel() for p in dit.parameters()):,}")
    print(f"[config] batch={args.batch_size} lr={args.lr:.2e} warmup={WARMUP_STEPS} target_steps={args.steps}")
    print("[precision] DiT forward fp16; cached Qwen embeddings are fp16")
    print("[stability] grad_clip=1.0; NaN guards=enabled; finite-checkpoints=enabled")
    print(f"[disk] checkpoint_dir={output_dir} local_window={LOCAL_CHECKPOINT_WINDOW} refuse_save_at={DISK_REFUSE_SAVE_PCT:.1f}%")
    nan_skip_count = 0
    running_loss = 0.0
    running_count = 0
    window_dropped = 0
    window_total = 0
    pbar = tqdm(total=args.steps - last_step, desc="RAY-DiT v4 100M-512")
    for step in range(last_step + 1, args.steps + 1):
        current_lr = set_learning_rate(optimizer, args.lr, step)
        if cache is not None:
            clean_latents, text_hidden, text_mask = next_cached_batch(cache, args.batch_size, device)
        else:
            try:
                images, captions = next_batch(iterator, args.batch_size)
            except StopIteration:
                iterator = iter(dataset)
                images, captions = next_batch(iterator, args.batch_size)
            images = images.to(device, non_blocking=True)
            with torch.inference_mode():
                _, clean_latents, _, _ = vae(images)
                text_hidden, text_mask = text_encoder.encode_texts(captions, max_length=TEXT_MAX_TOKENS)
            clean_latents = clean_latents.to(device=device, dtype=torch.float32)
            text_hidden = text_hidden.to(device=device, dtype=torch.float16)
            text_mask = text_mask.to(device=device, dtype=torch.bool)
        # Issue #7: caption dropout -> null placeholder, per sample.
        null_emb_fit, null_mask_fit = fit_null_to_length(null_emb_full, null_mask_full, text_hidden.shape[1])
        text_hidden, text_mask, dropped_now = apply_caption_dropout(
            text_hidden, text_mask, null_emb_fit, null_mask_fit, args.caption_drop,
        )
        window_dropped += dropped_now
        window_total += text_hidden.shape[0]
        noise = torch.randn_like(clean_latents)
        t = torch.rand(clean_latents.shape[0], device=device)
        t_view = t[:, None, None, None]
        z_t = (1.0 - t_view) * clean_latents + t_view * noise
        target_velocity = noise - clean_latents
        # RB-13: MSE of a trivial all-zeros predictor = mean(target^2). Logged on
        # every log line so the training loss can be read against a fixed baseline.
        zero_pred_mse = float((target_velocity.float() ** 2).mean())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            prediction = checkpointed_forward(dit, z_t, t, text_hidden, text_mask)
            loss = F.mse_loss(prediction.float(), target_velocity.float())
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            pbar.set_postfix(loss="nan", lr=f"{current_lr:.1e}")
            if step % LOG_EVERY == 0:
                vram_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                dropped_frac = window_dropped / max(window_total, 1)
                window_dropped = 0
                window_total = 0
                print(f"step={step} loss=nan grad_norm=nan zero_pred_mse={zero_pred_mse:.6f} NaN_skip_count={nan_skip_count} vram_used={vram_gb:.2f}GB dropped={dropped_frac:.3f} ema_drift={ema_drift(ema, dit):.6e}")
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), GRAD_CLIP)
        if not torch.isfinite(grad_norm).item():
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            continue
        optimizer.step()
        update_ema(ema, dit, args.ema_decay)  # Issue #7: after every optimizer.step()
        running_loss += float(loss.item())
        running_count += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.1e}")
        if step % LOG_EVERY == 0:
            mean_loss = running_loss / max(running_count, 1)
            vram_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            dropped_frac = window_dropped / max(window_total, 1)
            window_dropped = 0
            window_total = 0
            print(f"step={step} loss={loss.item():.6f} loss_avg={mean_loss:.6f} grad_norm={float(grad_norm):.4f} zero_pred_mse={zero_pred_mse:.6f} NaN_skip_count={nan_skip_count} vram_used={vram_gb:.2f}GB dropped={dropped_frac:.3f} ema_drift={ema_drift(ema, dit):.6e}")
            running_loss = 0.0
            running_count = 0
        if step % CHECKPOINT_EVERY == 0:
            save_or_upload_checkpoint(output_dir, dit, optimizer, step, args.seed, args.hf_repo, args.no_hf_upload, ema=ema)
    pbar.close()
    if args.steps % CHECKPOINT_EVERY != 0:
        save_or_upload_checkpoint(output_dir, dit, optimizer, args.steps, args.seed, args.hf_repo, args.no_hf_upload, ema=ema)


if __name__ == "__main__":
    main()
