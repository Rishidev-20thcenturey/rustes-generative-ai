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

# MONET Qwen Cache 200K v4

Cached RAY-IMAGE VAE latents and Qwen3-4B text embeddings for the public-domain Monet
paintings in [jasperai/monet](https://huggingface.co/datasets/jasperai/monet).

The name says "200K", but this repo holds **250,000 samples in 25 chunks** (chunk 0 to chunk 24).
The name is kept for consistency with earlier versions.

## What is in this repo

| Chunk | Samples | File |
|---|---|---|
| 0 | 0 to 9,999 | `monet_qwen_cache_0k_10k.pt` |
| 1 | 10,000 to 19,999 | `monet_qwen_cache_10k_20k.pt` |
| ... | ... | ... |
| 24 | 240,000 to 249,999 | `monet_qwen_cache_240k_250k.pt` |

Each chunk has 10,000 samples. All 25 files are about 7.9 GB each (about 197 GB in total).

## File contents

Each `.pt` file is a dictionary with three tensors per chunk:

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
    repo_id="Rishix500/monet-qwen-cache-200k-v4",
    filename="monet_qwen_cache_0k_10k.pt",
    repo_type="dataset",
)
chunk = torch.load(path, map_location="cpu")
latents = chunk["latents"]   # [10000, 16, 64, 64]
text_emb = chunk["text_emb"] # [10000, 128, 2560]
mask = chunk["mask"]         # [10000, 128]
```

## How the cache was built

- Source: `jasperai/monet` (Monet paintings, public domain).
- Latents from the RAY-IMAGE VAE v3 (512 px).
- Text embeddings from Qwen3-4B.
- Chunks were built one at a time and uploaded one at a time.

## Known issues

- **TODO (confirm before publishing):** The earlier version of this card said chunk 14 was corrupted and skipped. The current file list has `monet_qwen_cache_140k_150k.pt`, so chunk 14 was rebuilt. Please check its sample count and contents before we say anything about it here.

## License

The artwork is in the public domain. The cache files are derived from it. Check the source
dataset's terms before redistributing.
