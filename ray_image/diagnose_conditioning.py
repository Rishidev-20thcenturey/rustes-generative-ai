"""Check whether the text-conditioning path of the v5 DiT is alive.

Three tests, run in one process on a GPU:

1. Embedding similarity: Qwen3-4B embeddings of "a red apple" vs "a cyberpunk city".
   Raw pooled cosine > 0.95 suggests the encoder is not separating prompts.
   Raw pooled cosine < 0.80 suggests the encoder is fine, so look at the MoE.
   Hidden states share a large common direction, so raw cosine can be high even for
   unrelated text. The script also reports a centred cosine and the raw cosines of 20
   unrelated reference prompts, so the raw threshold can be read correctly.

2. CFG sweep: the same seed and prompts sampled at CFG 1.0, 3.0, 5.0, and 7.5 with the EMA
   checkpoint. The script reports (a) how far each image moves from its CFG 1.0 version,
   and (b) how different the two prompts are at each CFG. If neither changes at CFG 7.5,
   the text path has no effect. It also prints the learned cross-attention gate per block.

3. Null embedding: the stored aux/null_prompt_emb.pt is validated. Its shape and mask are
   compared with a live Qwen encoding of the same NULL_PROMPT.

Usage (GPU, from the repo root):
  python -m ray_image.diagnose_conditioning \\
      --ema-ckpt /path/ray_dit_v5_ema_step_0175000.pt \\
      --null-repo <org>/<repo-holding-aux/null_prompt_emb.pt> \\
      --out-dir /kaggle/working/conditioning_check
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from .cfg import NULL_PROMPT, NULL_PROMPT_FILENAME, fit_null_to_length, validate_null_payload
from .dit_v4 import LATENT_CHANNELS, LATENT_SIZE, MODEL_DEPTH, MODEL_DIM, MODEL_HEADS
from .dit_v5_moe import MOE_CAPACITY_FACTOR, MODEL_PATCH, RAYDiT_v5
from .flow import euler_sample
from .text_encoder import RAYTextEncoder
from .train_dit_v4 import VAE_DEFAULT_CHECKPOINT, VAE_DEFAULT_REPO, load_vae

TEXT_DIM = 2560
TEXT_TOKENS = 128
PROMPT_A = "a red apple"
PROMPT_B = "a cyberpunk city"
REFERENCE_PROMPTS = [
    "a portrait of an old fisherman with a grey beard",
    "a bowl of steaming ramen on a wooden table",
    "a snowy mountain landscape at sunrise",
    "a vintage red sports car parked on a coastal road",
    "a delicate pink lotus flower floating on water",
    "a cozy wooden cabin nestled in a pine forest",
    "a majestic white tiger walking through tall grass",
    "a bustling medieval marketplace full of merchants",
    "an abstract painting of swirling blue and gold shapes",
    "a sleeping cat curled up on a red cushion",
    "a stormy ocean with large waves under dark clouds",
    "a futuristic spaceship docked at a orbital station",
    "a field of sunflowers under a clear blue sky",
    "a close-up of a monarch butterfly on a leaf",
    "a quiet street in Paris on a rainy evening",
    "a plate of sushi arranged on black slate",
    "a medieval castle on a hill above a village",
    "a child flying a kite on a windy beach",
    "a glass of red wine next to a loaf of bread",
    "a dense jungle with a waterfall and mist",
]
CFG_LEVELS = [1.0, 3.0, 5.0, 7.5]
CROSS_ATTN_MOVED_THRESHOLD = 0.02  # mean |pixel difference| in [0,1] units; a choice made here, not from the spec


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def pooled(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid (non-padding) tokens. mask is True for padding."""
    valid = (~mask).float()[..., None]
    return (hidden * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0)


def encode_all(prompts, null_prompt):
    """Encode prompts and the live null prompt with Qwen. Returns CPU tensors and masks."""
    encoder = RAYTextEncoder(model_name=RAYTextEncoder.MODEL_ID, max_tokens=TEXT_TOKENS, load_in_8bit=True).eval()
    hidden, mask = encoder.encode_texts(list(prompts) + [null_prompt], max_length=TEXT_TOKENS, pad_to_max=True)
    del encoder
    torch.cuda.empty_cache()
    hidden = hidden.detach().float().cpu()
    mask = mask.detach().cpu().bool()
    return hidden[:-1], mask[:-1], hidden[-1], mask[-1]


def test_embedding_similarity(hidden, mask, log):
    pa, pb = pooled(hidden[0], mask[0]), pooled(hidden[1], mask[1])
    raw = cosine(pa, pb)

    ref_hidden = hidden[2:2 + len(REFERENCE_PROMPTS)]
    ref_mask = mask[2:2 + len(REFERENCE_PROMPTS)]
    ref = torch.stack([pooled(ref_hidden[i], ref_mask[i]) for i in range(len(REFERENCE_PROMPTS))])
    ref_pairs = [cosine(ref[i], ref[j]) for i in range(len(ref)) for j in range(i + 1, len(ref))]
    centre = ref.mean(dim=0)
    centred = cosine(pa - centre, pb - centre)

    log(f"raw pooled cosine('{PROMPT_A}', '{PROMPT_B}') = {raw:.4f}")
    log(f"centred cosine (mean of 20 reference prompts removed) = {centred:.4f}")
    log(f"raw cosine between unrelated reference prompts: min {min(ref_pairs):.4f} "
        f"median {float(np.median(ref_pairs)):.4f} max {max(ref_pairs):.4f}")
    if raw > 0.95:
        verdict = "FAIL: raw cosine > 0.95, Qwen embeddings nearly identical"
    elif raw < 0.80:
        verdict = "PASS: raw cosine < 0.80, Qwen separates prompts; look at the MoE/cross-attention"
    else:
        verdict = "AMBIGUOUS: raw cosine between 0.80 and 0.95"
    if float(np.median(ref_pairs)) > 0.95:
        verdict += (" (note: unrelated prompts also score > 0.95 raw, so the raw test cannot separate them;"
                    " the centred value is the better signal)")
    log(f"verdict: {verdict}")
    return {"raw_cosine": raw, "centred_cosine": centred,
            "reference_pairs_median": float(np.median(ref_pairs)), "verdict": verdict,
            "passed": raw < 0.95}


def load_null_payload(null_repo: str) -> dict:
    path = Path(hf_hub_download(repo_id=null_repo, filename=NULL_PROMPT_FILENAME, repo_type="model",
                                token=os.getenv("HF_TOKEN")))
    return torch.load(path, map_location="cpu", weights_only=False)


def test_null_embedding(payload, live_null_hidden, live_null_mask, log):
    valid = validate_null_payload(payload, text_dim=TEXT_DIM)
    emb = payload["emb"].float()
    mask = payload["mask"].bool()
    log(f"stored emb shape {tuple(emb.shape)} dtype {payload['emb'].dtype}; mask {tuple(mask.shape)}; "
        f"valid tokens {valid}; prompt {payload['prompt']!r}")
    log(f"expected shape for the DiT text input: [1, {TEXT_TOKENS}, {TEXT_DIM}]; stored emb is 2-D "
        f"[T, D], so it is unsqueezed to [1, T, D] before use")
    shape_ok = emb.shape == (TEXT_TOKENS, TEXT_DIM) and mask.shape == (TEXT_TOKENS,)
    fit_emb, fit_mask = fit_null_to_length(emb, mask, TEXT_TOKENS)
    if not shape_ok:
        log(f"warning: stored null is {tuple(emb.shape)}, not [{TEXT_TOKENS}, {TEXT_DIM}]; "
            f"it is padded or trimmed to {TEXT_TOKENS} tokens in training and sampling")
    live = live_null_hidden
    same_valid = bool(torch.equal(fit_mask[:valid], live_null_mask[:valid]))
    diff = float((fit_emb[:valid] - live[:valid]).abs().max())
    cos = cosine(fit_emb[:valid], live[:valid])
    log(f"live Qwen encoding of {NULL_PROMPT!r}: valid tokens {int((~live_null_mask).sum())}; "
        f"mask on valid tokens matches: {same_valid}; max |diff| {diff:.4f}; cosine {cos:.4f}")
    passed = shape_ok and same_valid and cos > 0.99
    verdict = "PASS" if passed else "FAIL"
    log(f"verdict: {verdict} (shape_ok={shape_ok}, mask_match={same_valid}, valid-token cosine {cos:.4f})")
    return {"stored_shape": list(emb.shape), "mask_shape": list(mask.shape), "valid_tokens": valid,
            "shape_ok": shape_ok, "mask_match": same_valid, "live_cosine": cos, "live_max_diff": diff,
            "passed": passed}


def load_dit(ema_ckpt: Path, device, log):
    payload = torch.load(ema_ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    model = RAYDiT_v5(
        latent_channels=LATENT_CHANNELS, latent_size=LATENT_SIZE,
        dim=MODEL_DIM, depth=MODEL_DEPTH, heads=MODEL_HEADS, patch=MODEL_PATCH,
        cond_dim=TEXT_DIM, cross_gate=True,
        num_experts=cfg.get("num_experts", 8), top_k=cfg.get("top_k", 2),
        expert_hidden_dim=cfg.get("expert_hidden_dim", 1728),
        capacity_factor=cfg.get("capacity_factor", MOE_CAPACITY_FACTOR),
    )
    model.load_state_dict(payload["dit"], strict=True)
    model.to(device).eval()
    gates = [float(block.cross_attn.cross_gate.detach().float().cpu()) for block in model.blocks
             if block.cross_attn.cross_gate is not None]
    log(f"loaded EMA step {payload.get('step')} ({len(model.blocks)} blocks); cross-attn gates "
        f"min {min(gates):.4f} mean {sum(gates) / len(gates):.4f} max {max(gates):.4f}")
    return model, gates


def test_cfg_sweep(model, vae, hidden, mask, null_emb, null_mask, out_dir: Path, steps: int, seed: int,
                   device, log):
    prompts = [PROMPT_A, PROMPT_B]
    rows = {}
    for cfg_scale in CFG_LEVELS:
        for idx, prompt in enumerate(prompts):
            torch.manual_seed(seed)
            text = hidden[idx][None].to(device)
            text_mask = mask[idx][None].to(device)
            z = euler_sample(model, text, shape=(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), steps=steps,
                             device=device, text_mask=text_mask, cfg_scale=cfg_scale,
                             null_text=null_emb[None].to(device), null_mask=null_mask[None].to(device))
            with torch.no_grad():
                image = vae.decode(z).float().cpu()
            rows[(cfg_scale, prompt)] = (z.float().cpu(), image)

    def pixels(img):
        return ((img.clamp(-1, 1) + 1) / 2 if img.min() < 0 else img.clamp(0, 1))[0]

    base = {p: pixels(rows[(CFG_LEVELS[0], p)][1]) for p in prompts}
    summary = {}
    for cfg_scale in CFG_LEVELS:
        moved = {p: float((pixels(rows[(cfg_scale, p)][1]) - base[p]).abs().mean()) for p in prompts}
        prompt_gap = float((pixels(rows[(cfg_scale, PROMPT_A)][1])
                            - pixels(rows[(cfg_scale, PROMPT_B)][1])).abs().mean())
        summary[cfg_scale] = {"moved_from_cfg1": moved, "prompt_gap": prompt_gap}
        log(f"CFG {cfg_scale}: |image - CFG1| mean  A={moved[PROMPT_A]:.4f} B={moved[PROMPT_B]:.4f}; "
            f"|A - B| prompt gap={prompt_gap:.4f}")

    top = max(summary[CFG_LEVELS[-1]]["moved_from_cfg1"].values())
    prompt_gap_top = summary[CFG_LEVELS[-1]]["prompt_gap"]
    passed = top >= CROSS_ATTN_MOVED_THRESHOLD and prompt_gap_top >= CROSS_ATTN_MOVED_THRESHOLD
    verdict = "PASS" if passed else "FAIL"
    log(f"verdict: {verdict} (at CFG {CFG_LEVELS[-1]}: max movement {top:.4f}, prompt gap "
        f"{prompt_gap_top:.4f}; threshold {CROSS_ATTN_MOVED_THRESHOLD})")

    grid_rows = []
    for cfg_scale in CFG_LEVELS:
        row = [pixels(rows[(cfg_scale, p)][1]) for p in prompts]
        grid_rows.append(torch.cat([img.permute(1, 2, 0) for img in row], dim=1))
    grid = torch.cat(grid_rows, dim=0).mul(255).round().clamp(0, 255).byte().numpy()
    out_path = out_dir / "cfg_sweep_apple_vs_city.png"
    Image.fromarray(grid).save(out_path)
    log(f"saved grid (rows = CFG {CFG_LEVELS}, columns = '{PROMPT_A}' | '{PROMPT_B}'): {out_path}")
    return {"summary": {str(k): v for k, v in summary.items()}, "threshold": CROSS_ATTN_MOVED_THRESHOLD,
            "passed": passed, "grid": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that the v5 DiT text-conditioning path is alive.")
    parser.add_argument("--ema-ckpt", type=Path, required=True, help="local path to ray_dit_v5_ema_step_*.pt")
    parser.add_argument("--null-repo", type=str, required=True,
                        help="HF repo holding aux/null_prompt_emb.pt (the v4 repo, per the plan)")
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/conditioning_check"))
    parser.add_argument("--vae-repo", type=str, default=VAE_DEFAULT_REPO)
    parser.add_argument("--vae-checkpoint", type=str, default=VAE_DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=20, help="Euler steps for the CFG sweep")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("diagnose_conditioning needs CUDA (Qwen and the DiT run on the GPU)")
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    log("== Test 1 + 3 setup: Qwen encodings ==")
    prompts = [PROMPT_A, PROMPT_B] + REFERENCE_PROMPTS
    hidden, mask, null_hidden, null_mask = encode_all(prompts, NULL_PROMPT)

    results = {}
    log("\n== Test 1: embedding similarity ==")
    results["embedding_similarity"] = test_embedding_similarity(hidden, mask, log)

    log("\n== Test 3: null embedding ==")
    null_payload = None
    try:
        null_payload = load_null_payload(args.null_repo)
        results["null_embedding"] = test_null_embedding(null_payload, null_hidden, null_mask, log)
    except Exception as exc:  # report and continue with the other tests
        log(f"verdict: FAIL ({type(exc).__name__}: {exc})")
        results["null_embedding"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    log("\n== Test 2: CFG sweep ==")
    try:
        if null_payload is None:
            raise RuntimeError("null prompt could not be loaded (see Test 3)")
        model, gates = load_dit(args.ema_ckpt, device, log)
        vae = load_vae(device, args.vae_repo, args.vae_checkpoint)
        null_fit_emb, null_fit_mask = fit_null_to_length(null_payload["emb"].float(), null_payload["mask"].bool(),
                                                         TEXT_TOKENS)
        cfg_result = test_cfg_sweep(model, vae, hidden, mask, null_fit_emb, null_fit_mask,
                                    args.out_dir, args.steps, args.seed, device, log)
        cfg_result["cross_attn_gates"] = {"min": min(gates), "mean": sum(gates) / len(gates), "max": max(gates)}
        results["cfg_sweep"] = cfg_result
    except Exception as exc:
        log(f"verdict: FAIL ({type(exc).__name__}: {exc})")
        results["cfg_sweep"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    log("\n== Summary ==")
    for name, res in results.items():
        log(f"{name}: {'PASS' if res.get('passed') else 'FAIL'}")
    alive = all(res.get("passed") for res in results.values())
    log(f"TEXT PATH {'ALIVE: safe to launch the 1M run' if alive else 'NOT CONFIRMED: do not launch the 1M run'}")
    (args.out_dir / "conditioning_check.json").write_text(json.dumps(results, indent=2, default=str))
    (args.out_dir / "conditioning_check.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
