---
license: apache-2.0
language:
- en
tags:
- vae
- cached-latents
- embedding
- qwen
- ray-image
size_categories:
- 100K<n<1M
---

# Mixed Qwen Cache 250K v5

Cached RAY-IMAGE VAE latents and Qwen3-4B text embeddings for chunks 25 to 49 of the RAY-IMAGE
training cache. It is the second half of the 500K-sample cache. The first half is
[monet-qwen-cache-200k-v4](https://huggingface.co/datasets/Rishix500/monet-qwen-cache-200k-v4).

This repo holds **25 chunks, 250,000 samples** (chunk 25 to chunk 49, samples 250,000 to 499,999).

**TODO before publishing:** the file names all start with `monet_qwen_cache_`, even though the
later chunks are not all Monet paintings. Confirm the source of each range below and decide whether
to rename the files. Renaming would break the training config, so do not rename without updating it.

| Samples | Source (TODO: confirm) |
|---|---|
| 250,000 to 319,999 (chunks 25 to 31) | TODO: MMArt |
| 320,000 to 499,999 (chunks 32 to 49) | TODO: Anime |

## What is in this repo

Each chunk is a `.pt` file with 10,000 samples, named `monet_qwen_cache_{start}k_{end}k.pt`.
For example, `monet_qwen_cache_250k_260k.pt` is chunk 25 and `monet_qwen_cache_490k_500k.pt` is chunk 49.
Each file is about 7.9 GB (about 197 GB in total).

## File contents

Same format as the v4 cache:

| Key | Shape (per chunk) | Description |
|---|---|---|
| `latents` | `[10000, 16, 64, 64]` | RAY-IMAGE VAE v3 latents |
| `text_emb` | `[10000, 128, 2560]` | Qwen3-4B text embeddings |
| `mask` | `[10000, 128]` | Valid-token mask for the text embeddings |

## How to load a chunk

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Rishix500/mixed-qwen-cache-250k-v5",
    filename="monet_qwen_cache_250k_260k.pt",
    repo_type="dataset",
)
chunk = torch.load(path, map_location="cpu")
```

## How the cache was built

- Latents from the RAY-IMAGE VAE v3 (512 px).
- Text embeddings from Qwen3-4B.
- Built one chunk at a time and uploaded one chunk at a time.

## Status

**TODO:** the repo is still filling. Chunk 49 (`monet_qwen_cache_490k_500k.pt`) had not finished uploading at
the last check. Update this section once all 25 files are present.

## License

**TODO:** the license depends on each source. Check the terms of each source dataset before
redistributing. Do not publish this card with the license line unless the sources allow it.
