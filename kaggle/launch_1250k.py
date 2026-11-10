"""Kaggle launcher for the 1.25M-step DiT v5 baseline run.

Baseline means the existing code with no SAGE and no sim loss (their flags stay off).

Run it from the bootstrap cell in kaggle/README.md. It stops before any GPU work unless:
  * both cache repos hold exactly the 25 expected chunks each,
  * the checked-out code contains the v5 regularizer flags,
  * a full v5 checkpoint is found on Hugging Face (newest one wins).

Rerunning it after a session timeout resumes from the newest uploaded checkpoint.
"""

import os
import re
import subprocess
import sys

REPO = "/kaggle/working/ray-image-v1"
MODEL_REPO = "Rishix500/ray-image-dit-v5-moe"
CACHE_REPO_1 = "Rishix500/monet-qwen-cache-200k-v4"    # chunks 0-24: starts 0k..240k
CACHE_REPO_2 = "Rishix500/mixed-qwen-cache-250k-v5"    # chunks 25-49: starts 250k..490k
CKPT_DIR = "/kaggle/tmp/checkpoints"
LOG_PATH = "/kaggle/tmp/train_1250k.log"
FALLBACK_STEP = 175000
TARGET_STEPS = 1250000

CACHE_NAME = re.compile(r"^monet_qwen_cache_(\d+)k_(\d+)k\.pt$")
CKPT_NAME = re.compile(r"^ray_dit_v5_step_(\d{7})\.pt$")
EXPECTED_STARTS_1 = {k * 10 for k in range(0, 25)}        # 0k..240k
EXPECTED_STARTS_2 = {k * 10 for k in range(25, 50)}       # 250k..490k
REQUIRED_FLAGS = ("use_sage", "use_sim_loss")

KEEP_LINE = re.compile(
    r"step=|\[aux\]|\[reg\]|milestone|checkpoint|\[HF\]|warm start|LR|resume|"
    r"Error|Traceback|NaN|dropped="
)


def run(cmd, **kwargs):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def check_cache_repo(files, expected_starts, repo_id):
    """Return the chunk starts found. Raise unless they match the expected set exactly."""
    starts = []
    for name in files:
        m = CACHE_NAME.match(name)
        if m:
            starts.append(int(m.group(1)))
    found = set(starts)
    if len(starts) != len(found):
        raise RuntimeError(f"{repo_id}: duplicate chunk files found")
    if len(found) != 25:
        raise RuntimeError(f"{repo_id}: found {len(found)} chunk files, expected 25. Not safe to launch.")
    missing = sorted(expected_starts - found)
    extra = sorted(found - expected_starts)
    if missing or extra:
        raise RuntimeError(f"{repo_id}: missing starts {missing} / unexpected starts {extra}")
    return sorted(found)


def pick_checkpoint(files):
    """Return (step, filename) of the newest full v5 checkpoint, or None if there is none."""
    best = None
    for name in files:
        m = CKPT_NAME.match(name)
        if m:
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, name)
    return best


def verify_code(repo):
    path = os.path.join(repo, "ray_image", "train_dit_v5.py")
    if not os.path.isfile(path):
        raise RuntimeError(f"{path} missing. Wrong checkout.")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    missing = [flag for flag in REQUIRED_FLAGS if flag not in text]
    if missing:
        raise RuntimeError(f"checked-out code lacks {missing}. Wrong commit.")


def main():
    # 1. Dependencies. Same kit as before; failures stop the run.
    subprocess.run(["pip", "uninstall", "-y", "-q", "torchvision", "datasets", "bitsandbytes", "scikit-learn"],
                   check=False)
    run(["pip", "install", "-q", "datasets==3.6.0", "transformers==4.51.3",
         "huggingface-hub>=0.30.0,<1.0", "torchvision", "lpips", "bitsandbytes", "accelerate"])

    from huggingface_hub import HfApi, hf_hub_download
    from kaggle_secrets import UserSecretsClient

    # 2. Token. Kaggle secret, not a value in this file.
    try:
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise RuntimeError("No HF_TOKEN secret found in Kaggle") from exc
    os.environ["HF_TOKEN"] = token
    api = HfApi(token=token)

    # 3. Code version.
    verify_code(REPO)
    print("[verify] code has v5 regularizer flags", flush=True)

    # 4. Data: both repos must hold exactly the expected chunks.
    files_1 = api.list_repo_files(repo_id=CACHE_REPO_1, repo_type="dataset")
    files_2 = api.list_repo_files(repo_id=CACHE_REPO_2, repo_type="dataset")
    starts_1 = check_cache_repo(files_1, EXPECTED_STARTS_1, CACHE_REPO_1)
    starts_2 = check_cache_repo(files_2, EXPECTED_STARTS_2, CACHE_REPO_2)
    print(f"[verify] repo 1: {len(starts_1)} chunks, repo 2: {len(starts_2)} chunks", flush=True)

    # 5. Checkpoint: newest full v5 checkpoint on Hugging Face.
    model_files = api.list_repo_files(repo_id=MODEL_REPO, repo_type="model")
    best = pick_checkpoint(model_files)
    if best is None:
        filename = f"ray_dit_v5_step_{FALLBACK_STEP:07d}.pt"
        if filename not in model_files:
            raise RuntimeError(f"no v5 checkpoint found in {MODEL_REPO}")
        step = FALLBACK_STEP
    else:
        step, filename = best
    if step >= TARGET_STEPS:
        print(f"[resume] newest checkpoint is step {step}, already at target. Nothing to do.", flush=True)
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[resume] downloading {filename} (step {step})", flush=True)
    ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=filename, local_dir=CKPT_DIR, token=token)
    print("[resume] checkpoint ready", flush=True)

    # 6. Train. Baseline: no --use-sage, no --use-sim-loss.
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "ray_image.train_dit_v5",
        "--resume", ckpt_path,
        "--steps", str(TARGET_STEPS),
        "--batch-size", "4",
        "--cache-window", "2",
        "--replay-buffer", "5000",
        "--cache-path", "/kaggle/tmp/cache",
        "--vae-repo", "Rishix500/ray-image-vae-v3-512",
        "--vae-checkpoint", "ray_vae_v3_step_0030000.pt",
        "--hf-repo", MODEL_REPO,
        "--cache-repos", f"{CACHE_REPO_1},{CACHE_REPO_2}",
        "--chunks-per-repo", "25",
        "--total-chunks", "50",
        "--caption-drop", "0.10",
        "--ema-decay", "0.999",
    ]
    print("\n[train] launching 1.25M baseline run", flush=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if KEEP_LINE.search(line):
                print(line, end="", flush=True)
        code = proc.wait()
    print(f"[train] exited with code {code}. Full log: {LOG_PATH}", flush=True)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
