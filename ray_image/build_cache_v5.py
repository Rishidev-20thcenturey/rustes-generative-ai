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
- Disk auto-purge: after a chunk is uploaded, its local .pt is deleted, all dataset caches
  (datasets--*) are removed, and *.incomplete files are deleted. The models--* folders (Qwen, VAE)
  are kept, so they are not re-downloaded. Without an upload (--no-hf-upload) nothing is deleted
  in datasets mode.
- Parquet mode (--source-format parquet --parquet-chunk-glob 'repo/path/{chunk}-0/*.parquet'): each
  chunk reads only its own folder. The folder must hold exactly --chunk-size rows. The dataset
  cache is purged after every chunk attempt, so the disk holds at most one chunk's files.

Run as a module from the repo root:
  python -m ray_image.build_cache_v5 --dataset-name huggan/wikiart --image-split train \\
      --caption-column caption --start-chunk 0 --end-chunk 9 --hf-repo <org>/<repo>
"""

import argparse
import os
import re
import string
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, constants, create_repo, hf_hub_download
from PIL import Image

from .parquet_source import (ChunkAbandoned, FailedRow, ParquetRowSource, RandomSampleSource, chunk_files,
                             list_repo_files, random_chunk_files)
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
CAPTION_MAX_WORDS = 90         # hard cap on the caption length (~117 tokens at 1.3 tokens/word)
CAPTION_MIN_WORDS = 4          # shorter than this: the row is dropped
_SENTENCE_END = re.compile(r"[.!](?=\s|$)")  # "." or "!" followed by whitespace or end (so 3.5 is not a split)


def sanitize_caption(raw_caption: str) -> str:
    """Clean a synthetic VLM caption before encoding.

    1. If the caption has more than 80 words, cut it after the first "." or "!" that ends a
       sentence. The subject is usually in the first sentence.
    2. Cap the result at 90 words.
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


class Heartbeat:
    """Background progress thread, so a hang shows up in the log within minutes.

    Prints a [heartbeat] line every interval_s with the current source row and seconds since the
    last row was read. After stall_s with no row, prints one [stall] line (once per stall). It
    does not kill anything: a stall is a signal to look at the log and restart if it persists.
    """

    def __init__(self, interval_s: float = 60.0, stall_s: float = 300.0, log=None,
                 clock=time.monotonic):
        self._interval = interval_s
        self._stall = stall_s
        self._log = log or (lambda message: print(message, flush=True))
        self._clock = clock
        self._lock = threading.Lock()
        self._last = clock()
        self._position = None
        self._stalled = False
        self._stop = threading.Event()
        self._thread = None
        self.stall_count = 0

    def touch(self, position: int) -> None:
        """Call once per source row read."""
        with self._lock:
            self._last = self._clock()
            self._position = position
            self._stalled = False

    def check(self) -> None:
        """One heartbeat tick. Called by the thread; public so tests can drive it."""
        with self._lock:
            idle = self._clock() - self._last
            position = self._position
            stalled = self._stalled
            if idle >= self._stall and not stalled:
                self._stalled = True
                self.stall_count += 1
        if idle >= self._stall and not stalled:
            self._log(f"[stall] no source row for {idle:.0f}s (last row read: {position}). "
                      f"The reader may be stuck; check the log and restart if this does not clear.")
        else:
            self._log(f"[heartbeat] last source row read: {position}; idle {idle:.0f}s")

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.check()

    def start(self) -> "Heartbeat":
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


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


def build_chunk(source, n_rows: int, vae, text_encoder, device: torch.device,
                batch_size: int, image_column: str, caption_column: str, dataset_name: str,
                start: int, end: int, caption_template=None, heartbeat=None) -> dict:
    """Encode the next n_rows source rows into a chunk payload. Raises SourceExhausted if the source ends early."""
    latents = torch.empty((n_rows, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=torch.float16)
    text_embs = torch.empty((n_rows, CACHE_TOKENS, TEXT_DIM), dtype=torch.float16)
    masks = torch.zeros((n_rows, CACHE_TOKENS), dtype=torch.bool)
    cached = 0
    seen = 0
    next_log = LOG_EVERY  # log every LOG_EVERY source rows, whatever batch boundary they land on
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
                if heartbeat is not None:
                    heartbeat.touch(source.position)
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

        if seen >= next_log or seen >= n_rows:
            print(f"[progress] chunk={start//1000}k-{end//1000}k source_row={source.position} "
                  f"seen={seen}/{n_rows} cached={cached} skipped={sum(skip_reasons.values())} "
                  f"{dict(skip_reasons)}", flush=True)
            next_log = (seen // LOG_EVERY + 1) * LOG_EVERY

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
    """Delete all dataset caches from the Hugging Face hub cache. Model caches are kept.

    Removes every <HF_HUB_CACHE>/datasets--* folder and every *.incomplete file under the hub
    cache. models--* folders (Qwen, VAE) are never touched, so they are not re-downloaded.
    Returns the bytes freed.
    """
    root = Path(constants.HF_HUB_CACHE).expanduser()
    if not root.exists():
        return 0
    if root.name != "hub" or root.is_symlink():
        log(f"[purge] refusing to purge unexpected path {root}")
        return 0
    freed = 0
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.is_symlink() and child.name.startswith("datasets--"):
            size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file() and not f.is_symlink())
            shutil.rmtree(child)
            freed += size
            log(f"[purge] deleted {child.name} ({size / (1024 ** 3):.2f} GB)")
    for part in root.rglob("*.incomplete"):
        if part.is_file() and not part.is_symlink():
            freed += part.stat().st_size
            part.unlink()
            log(f"[purge] deleted partial download {part.name}")
    log(f"[purge] kept model caches (models--*); freed {freed / (1024 ** 3):.2f} GB")
    return freed


def purge_stale_partials(output_dir: Path, log=print) -> int:
    """Delete *.pt.partial left by a build that was killed during torch.save. Returns bytes freed.

    Only called between chunks, when this process has no open partial file of its own.
    """
    freed = 0
    for part in output_dir.glob("*.pt.partial"):
        if part.is_file() and not part.is_symlink():
            freed += part.stat().st_size
            part.unlink()
            log(f"[purge] deleted stale partial chunk {part.name}")
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
    if args.random_sample and args.source_format != "parquet":
        raise ValueError("--random-sample requires --source-format parquet")
    if args.source_format == "parquet":
        if args.random_sample:
            if not args.random_sample_glob.strip():
                raise ValueError("--random-sample-glob must not be empty")
            if args.random_sample_files <= 0:
                raise ValueError("--random-sample-files must be >= 1")
            if args.max_skipped < 0:
                raise ValueError("--max-skipped must be >= 0")
        else:
            if "{chunk}" not in args.parquet_chunk_glob:
                raise ValueError("--parquet-chunk-glob must contain {chunk}, e.g. 'v1.2.0/cc12m/{chunk}-0/*.parquet'")
            try:
                args.parquet_chunk_glob.format(chunk=0)
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(f"--parquet-chunk-glob may only use the {{chunk}} field: {exc}")
        if args.source_offset != 0:
            raise ValueError("--source-offset must be 0 with --source-format parquet: each chunk reads its own folder")
    if args.probe_rows < 0:
        raise ValueError("--probe-rows must be >= 0")
    if args.heartbeat_s <= 0 or args.stall_s <= 0:
        raise ValueError("--heartbeat-s and --stall-s must be > 0")
    if not args.no_hf_upload and not args.hf_repo:
        raise ValueError("--hf-repo is required unless --no-hf-upload is set")
    if args.chunk_size != DEFAULT_CHUNK_SIZE:
        print(f"[warn] --chunk-size {args.chunk_size} differs from the trainer's {DEFAULT_CHUNK_SIZE}; "
              "the trainer's chunk ids will not line up with these files")


def source_columns(args) -> list:
    """Columns the parquet reader needs: the image plus the caption field(s) only."""
    if args.caption_template is None:
        caption_fields = {args.caption_column}
    else:
        caption_fields = template_fields(args.caption_template)
    return sorted({args.image_column} | caption_fields)


def make_datasets_source(args, start_position: int, log=print):
    """Row source for --source-format datasets: a SafeIterator over the streaming dataset."""

    def open_at(position: int):
        return load_dataset(args.dataset_name, args.dataset_config, split=args.image_split,
                            streaming=True).skip(position)

    return SafeIterator(open_at, start=start_position, log=log)


def make_chunk_source(args, index: int, all_files: list, log=print):
    """Row source for one chunk in parquet mode: exactly the files of that chunk's folder.

    Reads only the footers of those files (the files are downloaded to the local HF cache, which is
    purged after the chunk). The folder must hold exactly chunk_size rows, so a wrong folder-to-chunk
    mapping stops the build before any encoding.
    """
    pattern = args.parquet_chunk_glob.format(chunk=index)
    files = chunk_files(all_files, pattern)
    print(f"[parquet] chunk {index}: {len(files)} file(s) match {pattern!r}: "
          f"{files[0]} .. {files[-1]}", flush=True)
    t0 = time.monotonic()
    source = ParquetRowSource(args.dataset_name, files, source_columns(args), args.image_column,
                              start=0, log=log, token=os.getenv("HF_TOKEN"))
    print(f"[parquet] chunk {index}: {source.total_rows:,} rows ({time.monotonic() - t0:.1f}s)", flush=True)
    if source.total_rows != args.chunk_size:
        raise RuntimeError(
            f"chunk {index}: {pattern!r} has {source.total_rows:,} rows, expected {args.chunk_size:,}; "
            f"the folder-to-chunk mapping is wrong for this dataset")
    return source


def make_validator(args):
    """Row check used by random-sample mode. Mirrors the checks build_chunk does before encoding."""

    def validate(row):
        if row.get(args.image_column) is None:
            return False, "no_image"
        if args.caption_template is None:
            caption = row.get(args.caption_column)
            if not isinstance(caption, str) or not caption.strip():
                return False, "empty_caption"
        else:
            caption, reason = render_caption(row, args.caption_template)
            if caption is None:
                return False, reason
        if not sanitize_caption(caption):
            return False, "caption_dropped"
        return True, None

    return validate


def make_random_source(args, index: int, all_files: list, log=print):
    """Random-sample source for one chunk: valid rows drawn from this chunk's own slice of files."""
    files = random_chunk_files(all_files, args.random_sample_glob, args.seed, index, args.random_sample_files)
    print(f"[random] chunk {index}: {len(files)} file(s), {files[0]} .. {files[-1]}", flush=True)
    source = RandomSampleSource(args.dataset_name, files, source_columns(args), args.image_column,
                                validate=make_validator(args), target=args.chunk_size,
                                max_skipped=args.max_skipped, seed=args.seed * 1_000_003 + index,
                                log=log, token=os.getenv("HF_TOKEN"))
    print(f"[random] chunk {index}: {source.total_rows:,} candidate rows; need {args.chunk_size:,} valid; "
          f"abandon after {args.max_skipped:,} rejected", flush=True)
    return source


def make_parquet_source(args, index: int, all_files: list, log=print):
    """Parquet source for one chunk: random-sample mode or the fixed folder-per-chunk mode."""
    if args.random_sample:
        return make_random_source(args, index, all_files, log)
    return make_chunk_source(args, index, all_files, log)


def run_probe(args, source, heartbeat) -> bool:
    """Read --probe-rows rows from the source on this machine only (no models, no upload).

    Measures the source on its own: row reads, image decode, caption render. Use it to confirm
    the reader gets past the rows that stalled the full build.
    """
    counts = Counter()
    errors = []
    t0 = time.monotonic()
    begin = source.position
    first_error_at = {}
    for _ in range(args.probe_rows):
        try:
            sample = next(source)
        except StopIteration:
            counts["end_of_source"] += 1
            break
        except ChunkAbandoned as exc:
            counts["abandoned"] += 1
            errors.append(str(exc))
            break
        heartbeat.touch(source.position)
        if isinstance(sample, FailedRow):
            counts["source_error"] += 1
            if len(errors) < 10:
                errors.append(f"row {sample.position}: {sample.error}")
            continue
        image = sample.get(args.image_column)
        if image is None:
            counts["no_image"] += 1
            continue
        if args.caption_template is None:
            caption = sample.get(args.caption_column)
            reason = None if isinstance(caption, str) and caption.strip() else "empty_caption"
        else:
            caption, reason = render_caption(sample, args.caption_template)
        if reason is not None:
            counts[reason] += 1
            continue
        if not sanitize_caption(caption):
            counts["caption_dropped"] += 1
            continue
        try:
            preprocess_image(image)
        except (TypeError, ValueError, OSError) as exc:
            counts["preprocess_error"] += 1
            first_error_at.setdefault("preprocess_error", f"row {source.position - 1}: {exc}")
            continue
        counts["ok"] += 1
    elapsed = time.monotonic() - t0
    rows = source.position - begin
    print(f"[probe] read rows {begin:,}..{source.position:,} ({rows:,} rows in {elapsed:.1f}s, "
          f"{rows / max(elapsed, 1e-9):.1f} rows/s)")
    print(f"[probe] outcome {dict(counts)}; source retries={source.retries} "
          f"source failed rows={source.skipped_rows}")
    for line in errors:
        print(f"[probe] {line}")
    for key, value in first_error_at.items():
        print(f"[probe] first {key}: {value}")
    # Pass only if every row was readable and at least one row is usable.
    problems = []
    if counts["source_error"]:
        problems.append(f"{counts['source_error']} unreadable source row(s)")
    if counts["end_of_source"]:
        problems.append("source ended before --probe-rows")
    if counts["abandoned"]:
        problems.append("chunk would be abandoned (too many rejected rows)")
    if counts["ok"] == 0:
        problems.append("no usable rows")
    if problems:
        print(f"[probe] failed: {'; '.join(problems)}")
        return False
    print("[probe] passed")
    return True


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
    parser.add_argument("--source-format", choices=["datasets", "parquet"], default="datasets",
                        help="datasets: HF datasets streaming (default). parquet: read the Parquet files "
                             "directly, one row group at a time (see parquet_source.py).")
    parser.add_argument("--parquet-chunk-glob", type=str, default="",
                        help="with --source-format parquet: repo file pattern for ONE chunk, with {chunk} "
                             "replaced by the chunk index, e.g. 'v1.2.0/cc12m/{chunk}-0/*.parquet'. "
                             "Each chunk reads only its own files; the folder must hold exactly --chunk-size rows.")
    parser.add_argument("--random-sample", action="store_true",
                        help="parquet mode: draw each chunk's rows at random from its own slice of files "
                             "(--random-sample-glob, --random-sample-files). Ignores --parquet-chunk-glob.")
    parser.add_argument("--random-sample-glob", type=str, default="v1.2.0/cc12m/*.parquet",
                        help="with --random-sample: repo files to draw from")
    parser.add_argument("--random-sample-files", type=int, default=8,
                        help="with --random-sample: files per chunk (disjoint across chunks; ~0.5 GB each for Monet)")
    parser.add_argument("--max-skipped", type=int, default=2000,
                        help="with --random-sample: abandon a chunk after more than this many rejected rows")
    parser.add_argument("--probe-rows", type=int, default=0,
                        help="read this many source rows on the CPU (no models, no upload) and report, then exit")
    parser.add_argument("--heartbeat-s", type=float, default=60.0, help="progress line interval, seconds")
    parser.add_argument("--stall-s", type=float, default=300.0,
                        help="print [stall] after this many seconds with no source row")
    args = parser.parse_args()
    if args.probe_rows > 0:
        args.no_hf_upload = True  # the probe never uploads (parquet mode only clears the dataset download cache)
    validate_args(args)

    if args.probe_rows > 0:
        heartbeat = Heartbeat(args.heartbeat_s, args.stall_s).start()
        try:
            if args.source_format == "parquet":
                all_files = list_repo_files(args.dataset_name, token=os.getenv("HF_TOKEN"))
                source = make_parquet_source(args, args.start_chunk, all_files)
            else:
                source = make_datasets_source(args, args.start_chunk * args.chunk_size - args.source_offset)
            passed = run_probe(args, source, heartbeat)
        finally:
            heartbeat.stop()
            if args.source_format == "parquet":
                purge_hf_hub_cache()
        raise SystemExit(0 if passed else 2)

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

    offset = args.source_offset
    parquet_mode = args.source_format == "parquet"
    all_files = []
    source = None
    if parquet_mode:
        all_files = list_repo_files(args.dataset_name, token=hf_token)
    else:
        source = make_datasets_source(args, args.start_chunk * args.chunk_size - offset)
    heartbeat = Heartbeat(args.heartbeat_s, args.stall_s).start()
    print(f"[window] dataset={args.dataset_name}:{args.image_split} chunks {args.start_chunk}.."
          f"{args.end_chunk - 1} (global rows {args.start_chunk * args.chunk_size:,}.."
          f"{args.end_chunk * args.chunk_size:,}; dataset rows "
          f"{args.start_chunk * args.chunk_size - offset:,}..{args.end_chunk * args.chunk_size - offset:,}; "
          f"source-offset {offset:,})")

    completed = 0
    abandoned = []
    for index in range(args.start_chunk, args.end_chunk):
        start = index * args.chunk_size
        end = start + args.chunk_size
        filename = chunk_filename(start, end)
        local_path = output_dir / filename
        if not parquet_mode:
            source.skip_to(start - offset)

        if filename in existing_hf and not args.no_hf_upload:
            print(f"[resume] HF already has {filename}; skipping chunk {index}")
            if local_path.exists():
                local_path.unlink()
            if not parquet_mode:
                source.skip_to(end - offset)
            completed += 1
            continue

        if chunk_is_complete(local_path, args.chunk_size):
            print(f"[resume] complete local chunk {local_path.name}; uploading without re-encoding")
        else:
            print(f"[build] chunk {index}: source rows {start}:{end}")
            try:
                if parquet_mode:
                    source = make_parquet_source(args, index, all_files)
                payload = build_chunk(source, args.chunk_size, vae, text_encoder, device, args.batch_size,
                                      args.image_column, args.caption_column, args.dataset_name, start, end,
                                      caption_template=args.caption_template, heartbeat=heartbeat)
            except SourceExhausted as exc:
                print(f"[stop] chunk {index} not written: {exc}. Earlier chunks are complete.")
                raise SystemExit(1)
            except ChunkAbandoned as exc:
                print(f"[warn] chunk {index} ABANDONED, no file written: {exc}", flush=True)
                abandoned.append(index)
                continue
            except Exception:
                print(f"[error] chunk {index} failed; rerun with the same arguments to retry it")
                raise
            finally:
                if parquet_mode:
                    # Chunk files are only needed while this chunk is read; free the disk now.
                    source = None
                    purge_hf_hub_cache()
            save_chunk_atomic(payload, local_path)
            del payload

        if not args.no_hf_upload:
            upload_chunk(local_path, api, args.hf_repo)
            existing_hf.add(filename)
            local_path.unlink()
            print(f"[purge] deleted local {filename}")
            purge_hf_hub_cache()
            purge_stale_partials(output_dir)
        completed += 1

    heartbeat.stop()
    print(f"[done] completed {completed}/{args.end_chunk - args.start_chunk} chunks; "
          f"source retries={source.retries} skipped source rows={source.skipped_rows} "
          f"stalls={heartbeat.stall_count}")
    if abandoned:
        print(f"[warn] abandoned chunks (no file written): {abandoned}. A rerun with the same settings "
              f"abandons them again (same files, same row order). Changing --seed reshuffles every chunk "
              f"and can duplicate rows already built; do not do that mid-run.")


if __name__ == "__main__":
    main()
