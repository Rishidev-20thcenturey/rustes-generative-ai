"""Build the Qwen + VAE latent cache for the 512px / 64x64-latent DiT pipeline from any HF image dataset.

Per sample: image -> RGB 512x512 (LANCZOS) -> RAYVAE_v3 encoder (16 x 64 x 64 latent), and the
caption -> Qwen3-4B hidden states capped at 128 tokens. The output chunks use the same tensor layout
and file names as the v4 cache (monet_qwen_cache_{start//1000}k_{end//1000}k.pt), so
train_dit_v4 and train_dit_v5 load them without changes.

Chunk N covers source rows N*chunk_size .. (N+1)*chunk_size-1 of --dataset-name / --image-split.
Rows that cannot be used (missing image or caption, bad image, encoder failure) are skipped and
counted, but still use up a source row, so chunk boundaries stay fixed.

Behaviour:
- SafeIterator: a failing source row is retried (network drops), then skipped if it keeps failing.
- Atomic writes: a chunk is written to *.pt.partial and renamed to *.pt only when complete.
- Resume: a chunk already in --hf-repo is skipped. A complete local chunk is uploaded without
  re-encoding. Only the source rows are skipped.
- Disk auto-purge: after a chunk is uploaded, its local .pt is deleted and the Hugging Face hub
  cache is cleared (including *.incomplete files). The VAE and Qwen weights are re-downloaded on the
  next start. Without an upload (--no-hf-upload) nothing is deleted.

Run as a module from the repo root:
  python -m ray_image.build_cache_v5 --dataset-name huggan/wikiart --image-split train \\
      --caption-column caption --start-chunk 0 --end-chunk 9 --hf-repo <org>/<repo>
"""

import argparse
import os
import re
import string
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, constants, create_repo, hf_hub_download
from PIL import Image

from .text_encoder import RAYTextEncoder
from .vae_v3 import IMAGE_SIZE, LATENT_SIZE, RAYVAE_v3

LATENT_CHANNELS = 16
VAE_REPO = "Rishix500/ray-image-vae-v3-512"
VAE_CHECKPOINT = "ray_vae_v3_step_0030000.pt"
TEXT_DIM = 2560
CACHE_TOKENS = 128
DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_OUTPUT_DIR = "/kaggle/tmp"
LOG_EVERY = 1_000
CACHE_FORMAT = "ray-image-v4-qwen-cache-v1"  # same tensor layout as the v4 cache
MAX_RETRIES = 3
RETRY_BACKOFF_S = 5.0


# Caption sanitization (applied to every caption before the Qwen encoder).
CAPTION_LONG_WORDS = 80        # longer than this: cut at the first sentence end
CAPTION_MAX_WORDS = 120        # hard cap on the caption length
CAPTION_MIN_WORDS = 4          # shorter than this: the row is dropped
_SENTENCE_END = re.compile(r"[.!](?=\s|$)")  # "." or "!" followed by whitespace or end (so 3.5 is not a split)


def sanitize_caption(raw_caption: str) -> str:
    """Clean a synthetic VLM caption before encoding.

    1. If the caption has more than 80 words, cut it after the first "." or "!" that ends a
       sentence. The subject is usually in the first sentence.
    2. Cap the result at 120 words.
    3. Drop captions with fewer than 4 words.

    Returns the cleaned caption, or "" if the row should be dropped.
    """
    if not isinstance(raw_caption, str):
        return ""
    text = " ".join(raw_caption.split())  # collapse newlines and repeated spaces
    if len(text.split()) > CAPTION_LONG_WORDS:
        match = _SENTENCE_END.search(text)
        if match is not None:
            text = text[:match.end()]
    words = text.split()
    if len(words) > CAPTION_MAX_WORDS:
        text = " ".join(words[:CAPTION_MAX_WORDS])
    if len(text.split()) < CAPTION_MIN_WORDS:
        return ""
    return text


class SourceExhausted(RuntimeError):
    """The source dataset ended before the chunk was complete."""


def chunk_filename(start: int, end: int) -> str:
    # Same name the v4/v5 trainers build (see train_dit_v4.chunk_filename).
    return f"monet_qwen_cache_{start // 1000}k_{end // 1000}k.pt"


class FailedRow:
    """Yielded in place of a source row that could not be read after all retries."""

    def __init__(self, position: int, error: str):
        self.position = position
        self.error = error


class SafeIterator:
    """Iterate a source dataset that may hit network drops or corrupt rows.

    open_at(position) must return an iterator that starts at row `position`. Any exception
    while fetching row p (the generator is dead after an error) causes a reopen at p. After
    max_retries failures on the same row, a FailedRow marker is returned for it and iteration
    continues at p+1. Every source row occupies exactly one position, so chunk boundaries
    stay exact.
    """

    def __init__(self, open_at, start: int = 0, max_retries: int = MAX_RETRIES,
                 backoff_s: float = RETRY_BACKOFF_S, sleep=time.sleep, log=print):
        self._open_at = open_at
        self.position = start
        self._it = None
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._log = log
        self.retries = 0
        self.skipped_rows = 0

    def skip_to(self, position: int) -> None:
        """Move to an absolute row position. Reopens lazily on the next read."""
        if position < self.position:
            raise ValueError(f"cannot rewind source from row {self.position} to {position}")
        if position != self.position:
            self.position = position
            self._it = None

    def __iter__(self):
        return self

    def __next__(self):
        attempts = 0
        while True:
            try:
                if self._it is None:
                    self._it = iter(self._open_at(self.position))
                row = next(self._it)
            except StopIteration:
                raise
            except Exception as exc:  # network drop, corrupt row, decode error
                self._it = None
                attempts += 1
                if attempts <= self._max_retries:
                    self.retries += 1
                    self._log(f"[safe] row {self.position}: {type(exc).__name__}: {exc}; "
                              f"retry {attempts}/{self._max_retries}")
                    self._sleep(self._backoff_s * attempts)
                    continue
                self.skipped_rows += 1
                self._log(f"[safe] row {self.position}: failed {attempts} times; skipping it")
                marker = FailedRow(self.position, f"{type(exc).__name__}: {exc}")
                self.position += 1
                return marker
            self.position += 1
            return row


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """RGB, 512x512 (LANCZOS), float in [0, 1], shape [3, 512, 512]."""
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_vae(device: torch.device) -> RAYVAE_v3:
    checkpoint_path = Path(hf_hub_download(
        repo_id=VAE_REPO,
        filename=VAE_CHECKPOINT,
        repo_type="model",
        token=os.getenv("HF_TOKEN"),
    ))
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vae = RAYVAE_v3(latent_channels=LATENT_CHANNELS, base_channels=64)
    # strict=True: a wrong or partial checkpoint must fail here, not produce a bad cache.
    vae.load_state_dict(checkpoint_data["vae"], strict=True)
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"[VAE] loaded {VAE_REPO}/{checkpoint_path.name}")
    return vae


def template_fields(template: str) -> set:
    """Field names used by a caption template. Positional fields ({}) are rejected by validate_args."""
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def render_caption(row: dict, template: str):
    """Fill a caption template from named row fields.

    Returns (caption, None) on success, or (None, reason) if a field is missing or empty.
    """
    values = {}
    for name in template_fields(template):
        value = row.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None, "template_missing_field"
        values[name] = value
    caption = template.format(**values).strip()
    if not caption:
        return None, "template_missing_field"
    return caption, None


def build_chunk(source: SafeIterator, n_rows: int, vae, text_encoder, device: torch.device,
                batch_size: int, image_column: str, caption_column: str, dataset_name: str,
                start: int, end: int, caption_template=None) -> dict:
    """Encode the next n_rows source rows into a chunk payload. Raises SourceExhausted if the source ends early."""
    latents = torch.empty((n_rows, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=torch.float16)
    text_embs = torch.empty((n_rows, CACHE_TOKENS, TEXT_DIM), dtype=torch.float16)
    masks = torch.zeros((n_rows, CACHE_TOKENS), dtype=torch.bool)
    cached = 0
    seen = 0
    skip_reasons = Counter()
    retries_before = source.retries
    skipped_before = source.skipped_rows
    source_row_start = source.position

    while seen < n_rows:
        images = []
        captions = []
        while len(images) < batch_size and seen < n_rows:
            try:
                sample = next(source)
            except StopIteration:
                raise SourceExhausted(
                    f"source ended after {seen} of {n_rows} rows at source position {source.position}"
                )
            seen += 1
            if isinstance(sample, FailedRow):
                skip_reasons["source_error"] += 1
                continue
            image = sample.get(image_column)
            if image is None:
                skip_reasons["no_image"] += 1
                continue
            if caption_template is None:
                caption = sample.get(caption_column)
                if not isinstance(caption, str) or not caption.strip():
                    skip_reasons["empty_caption"] += 1
                    continue
            else:
                caption, reason = render_caption(sample, caption_template)
                if caption is None:
                    skip_reasons[reason] += 1
                    continue
            caption = sanitize_caption(caption)
            if not caption:
                skip_reasons["caption_dropped"] += 1
                continue
            try:
                images.append(preprocess_image(image))
                captions.append(caption.strip())
            except (TypeError, ValueError, OSError):
                skip_reasons["preprocess_error"] += 1

        if not images:
            continue

        image_batch = torch.stack(images).to(device, non_blocking=True)
        with torch.inference_mode():
            try:
                _, z, _, _ = vae(image_batch)
                # Pad every row to CACHE_TOKENS so each cached row has one fixed shape.
                text_hidden, mask = text_encoder.encode_texts(
                    captions, max_length=CACHE_TOKENS, pad_to_max=True
                )
            except Exception as exc:
                skip_reasons["encoder_failure"] += len(images)
                print(f"[skip] encoder batch failure: {exc}")
                continue

        z = z.detach().float().cpu().to(torch.float16)
        text_hidden = text_hidden.detach().cpu().to(torch.float16)
        mask = mask.detach().cpu().bool()

        for i in range(z.shape[0]):
            if not (torch.isfinite(z[i]).all().item() and torch.isfinite(text_hidden[i]).all().item()):
                skip_reasons["non_finite"] += 1
                continue
            if not (tuple(z[i].shape) == (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
                    and tuple(text_hidden[i].shape) == (CACHE_TOKENS, TEXT_DIM)
                    and tuple(mask[i].shape) == (CACHE_TOKENS,)):
                skip_reasons["shape_mismatch"] += 1
                continue
            latents[cached] = z[i]
            text_embs[cached] = text_hidden[i]
            masks[cached] = mask[i]
            cached += 1

        if seen % LOG_EVERY == 0 or seen >= n_rows:
            print(f"[progress] chunk={start//1000}k-{end//1000}k source_row={source.position} "
                  f"seen={seen}/{n_rows} cached={cached} skipped={sum(skip_reasons.values())} "
                  f"{dict(skip_reasons)}")

    if cached == 0:
        raise RuntimeError(f"No valid samples produced for chunk {start}:{end}")
    if cached < n_rows:
        # Trim so the file holds exactly the cached rows (clone releases the spare storage).
        latents = latents[:cached].clone()
        text_embs = text_embs[:cached].clone()
        masks = masks[:cached].clone()

    return {
        "format": CACHE_FORMAT,
        "source": dataset_name,
        "source_start": start,
        "source_end": end,
        "source_row_start": source_row_start,
        "source_row_end": source_row_start + seen,
        "samples_seen": seen,
        "samples_cached": cached,
        "samples_skipped": sum(skip_reasons.values()),
        "skip_reasons": dict(skip_reasons),
        "source_retries": source.retries - retries_before,
        "source_skipped_rows": source.skipped_rows - skipped_before,
        "latent_dtype": "float16",
        "text_dtype": "float16",
        "latents": latents,
        "text_emb": text_embs,
        "mask": masks,
    }


def save_chunk_atomic(payload: dict, path: Path) -> None:
    """Write to *.pt.partial and rename, so a *.pt file only exists once it was fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)
    print(f"[saved] {path} ({path.stat().st_size / (1024 ** 3):.2f} GB) cached={payload['samples_cached']} "
          f"skipped={payload['samples_skipped']} skip_reasons={payload['skip_reasons']}")


def chunk_is_complete(path: Path, chunk_size: int) -> bool:
    """True if path is a readable chunk with consistent shapes and at most chunk_size rows."""
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        print(f"[resume] {path.name} is unreadable ({type(exc).__name__}: {exc}); it will be rebuilt")
        return False
    try:
        latents, text_emb, mask = payload["latents"], payload["text_emb"], payload["mask"]
        n = latents.shape[0]
        ok = (0 < n <= chunk_size
              and tuple(latents.shape[1:]) == (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
              and tuple(text_emb.shape) == (n, CACHE_TOKENS, TEXT_DIM)
              and tuple(mask.shape) == (n, CACHE_TOKENS)
              and payload.get("samples_cached") == n)
    except KeyError:
        ok = False
    if not ok:
        print(f"[resume] {path.name} has the wrong layout; it will be rebuilt")
    return ok


def purge_hf_hub_cache(log=print) -> int:
    """Delete the Hugging Face hub cache (HF_HUB_CACHE, default ~/.cache/huggingface/hub).

    This removes the .incomplete download files as well. Returns the bytes freed.
    """
    root = Path(constants.HF_HUB_CACHE).expanduser()
    if not root.exists():
        return 0
    if root.name != "hub" or root.is_symlink():
        log(f"[purge] refusing to purge unexpected path {root}")
        return 0
    freed = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink())
    shutil.rmtree(root)
    log(f"[purge] cleared {root} ({freed / (1024 ** 3):.2f} GB, including .incomplete files)")
    return freed


def hf_repo_files(api: HfApi, repo_id: str) -> set:
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception as exc:
        print(f"[HF] could not list {repo_id}: {exc}")
        return set()


def upload_chunk(path: Path, api: HfApi, repo_id: str) -> None:
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"feat(cache-v5): upload cache chunk {path.name}",
    )
    print(f"[HF] uploaded {path.name} -> dataset/{repo_id}")


def validate_args(args) -> None:
    if args.caption_template is not None:
        try:
            names = [name for _, name, _, _ in string.Formatter().parse(args.caption_template) if name is not None]
        except ValueError as exc:
            raise ValueError(f"--caption-template is not a valid format string: {exc}")
        if not names or any(name == "" for name in names):
            raise ValueError("--caption-template must use named fields, e.g. 'a painting by {artist}, {style}'")
    if args.source_offset < 0:
        raise ValueError("--source-offset must be >= 0")
    if args.start_chunk * args.chunk_size < args.source_offset:
        raise ValueError(
            f"--start-chunk {args.start_chunk} starts at global row {args.start_chunk * args.chunk_size}, "
            f"which is before --source-offset {args.source_offset}; start at chunk "
            f"{-(-args.source_offset // args.chunk_size)} or later"
        )
    if not args.dataset_name.strip():
        raise ValueError("--dataset-name is required")
    if args.start_chunk < 0 or args.end_chunk <= args.start_chunk:
        raise ValueError("--start-chunk must be >= 0 and --end-chunk must be > --start-chunk")
    if args.chunk_size <= 0 or args.chunk_size % 1000 != 0:
        raise ValueError("--chunk-size must be a positive multiple of 1000 (chunk file names use thousands)")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not args.no_hf_upload and not args.hf_repo:
        raise ValueError("--hf-repo is required unless --no-hf-upload is set")
    if args.chunk_size != DEFAULT_CHUNK_SIZE:
        print(f"[warn] --chunk-size {args.chunk_size} differs from the trainer's {DEFAULT_CHUNK_SIZE}; "
              "the trainer's chunk ids will not line up with these files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Qwen + VAE latent cache from any HF image dataset.")
    parser.add_argument("--dataset-name", type=str, required=True, help="e.g. huggan/wikiart")
    parser.add_argument("--dataset-config", type=str, default=None, help="dataset config name, if it has one")
    parser.add_argument("--image-split", type=str, default="train")
    parser.add_argument("--image-column", type=str, default="image")
    parser.add_argument("--caption-template", type=str, default=None,
                        help="Build the caption from row fields, e.g. 'a painting by {artist}, {style}'. "
                             "Overrides --caption-column. Rows with a missing field are skipped.")
    parser.add_argument("--caption-column", type=str, default="caption",
                        help="column with the caption text. Rows without one are skipped.")
    parser.add_argument("--start-chunk", type=int, required=True)
    parser.add_argument("--end-chunk", type=int, required=True, help="exclusive")
    parser.add_argument("--source-offset", type=int, default=0,
                        help="Global row where this dataset begins. Chunk c reads dataset row "
                             "c*chunk_size - source_offset. Lets several datasets fill one chunk space.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--hf-repo", type=str, default=None, help="output dataset repo, e.g. Rishix500/mixed-qwen-cache-250k-v5")
    parser.add_argument("--no-hf-upload", action="store_true", help="build locally only; nothing is uploaded or deleted")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="cuda (default) or cpu for tests")
    args = parser.parse_args()
    validate_args(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cache build needs CUDA (use --device cpu only for tests)")

    hf_token = os.getenv("HF_TOKEN")
    if not args.no_hf_upload:
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for cache uploads. Use --no-hf-upload for a local build.")
        # private=True applies at creation only; existing repo visibility is never modified.
        create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True, private=True, token=hf_token)

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=hf_token)
    existing_hf = set() if args.no_hf_upload else hf_repo_files(api, args.hf_repo)

    # Load the models once for the whole run, not per chunk.
    vae = load_vae(device)
    text_encoder = RAYTextEncoder(model_name=RAYTextEncoder.MODEL_ID, max_tokens=CACHE_TOKENS,
                                  load_in_8bit=True).eval()

    def open_at(position: int):
        return load_dataset(args.dataset_name, args.dataset_config, split=args.image_split,
                            streaming=True).skip(position)

    offset = args.source_offset
    source = SafeIterator(open_at, start=args.start_chunk * args.chunk_size - offset)
    print(f"[window] dataset={args.dataset_name}:{args.image_split} chunks {args.start_chunk}.."
          f"{args.end_chunk - 1} (global rows {args.start_chunk * args.chunk_size:,}.."
          f"{args.end_chunk * args.chunk_size:,}; dataset rows "
          f"{args.start_chunk * args.chunk_size - offset:,}..{args.end_chunk * args.chunk_size - offset:,}; "
          f"source-offset {offset:,})")

    completed = 0
    for index in range(args.start_chunk, args.end_chunk):
        start = index * args.chunk_size
        end = start + args.chunk_size
        filename = chunk_filename(start, end)
        local_path = output_dir / filename
        source.skip_to(start - offset)

        if filename in existing_hf and not args.no_hf_upload:
            print(f"[resume] HF already has {filename}; skipping chunk {index}")
            if local_path.exists():
                local_path.unlink()
            source.skip_to(end - offset)
            completed += 1
            continue

        if chunk_is_complete(local_path, args.chunk_size):
            print(f"[resume] complete local chunk {local_path.name}; uploading without re-encoding")
        else:
            print(f"[build] chunk {index}: source rows {start}:{end}")
            try:
                payload = build_chunk(source, args.chunk_size, vae, text_encoder, device, args.batch_size,
                                      args.image_column, args.caption_column, args.dataset_name, start, end,
                                      caption_template=args.caption_template)
            except SourceExhausted as exc:
                print(f"[stop] chunk {index} not written: {exc}. Earlier chunks are complete.")
                raise SystemExit(1)
            except Exception:
                print(f"[error] chunk {index} failed; rerun with the same arguments to retry it")
                raise
            save_chunk_atomic(payload, local_path)
            del payload

        if not args.no_hf_upload:
            upload_chunk(local_path, api, args.hf_repo)
            existing_hf.add(filename)
            local_path.unlink()
            print(f"[purge] deleted local {filename}")
            purge_hf_hub_cache()
        completed += 1

    print(f"[done] completed {completed}/{args.end_chunk - args.start_chunk} chunks; "
          f"source retries={source.retries} skipped source rows={source.skipped_rows}")


if __name__ == "__main__":
    main()
