"""Hugging Face downloads over plain HTTP with requests.

Replaces huggingface_hub.hf_hub_download on Kaggle, where that function fails with
"cannot import name 'XetDownloadProgressReporter'". Uploads and file listing still use HfApi.

Only the parts the training code uses are supported: repo_id, filename, repo_type,
token, revision, local_dir, cache_dir. Other keyword arguments are accepted and ignored.
"""

import os

import requests

HF_BASE = "https://huggingface.co"
DEFAULT_CACHE = "/kaggle/tmp/hf_rest_cache"
CHUNK_BYTES = 8 * 1024 * 1024
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 120


def _prefix(repo_type):
    if repo_type in (None, "model"):
        return ""
    if repo_type == "dataset":
        return "datasets/"
    raise ValueError(f"unsupported repo_type: {repo_type!r}")


def _headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def resolve_url(repo_id, filename, repo_type="model", revision="main"):
    return f"{HF_BASE}/{_prefix(repo_type)}{repo_id}/resolve/{revision}/{filename}"


def list_repo_files(repo_id, repo_type="model", token=None, revision="main"):
    """Return every file path in the repo, following pagination."""
    kind = "datasets" if repo_type == "dataset" else "models"
    url = f"{HF_BASE}/api/{kind}/{repo_id}/tree/{revision}?recursive=true"
    files = []
    while url:
        resp = requests.get(url, headers=_headers(token), timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        resp.raise_for_status()
        files.extend(item["path"] for item in resp.json() if item.get("type") == "file")
        url = resp.links.get("next", {}).get("url")
    return files


def hf_hub_download(repo_id, filename, repo_type=None, token=None, revision="main",
                    local_dir=None, cache_dir=None, **_ignored):
    """Download one file and return its local path, like huggingface_hub.hf_hub_download.

    The file is written to <local_dir>/<filename> when local_dir is given. Otherwise it goes
    under DEFAULT_CACHE (or cache_dir). An existing non-empty file is reused. The download goes
    to a .part file first, so an interrupted run never leaves a half-written file under the real name.
    """
    rtype = repo_type or "model"
    if local_dir:
        dest = os.path.join(local_dir, filename)
    else:
        dest = os.path.join(cache_dir or DEFAULT_CACHE, rtype, repo_id, filename)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    url = resolve_url(repo_id, filename, rtype, revision)
    with requests.get(url, headers=_headers(token), stream=True, allow_redirects=True,
                      timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
        resp.raise_for_status()
        expected = resp.headers.get("Content-Length")
        written = 0
        with open(part, "wb") as out:
            for chunk in resp.iter_content(CHUNK_BYTES):
                if chunk:
                    out.write(chunk)
                    written += len(chunk)
    if expected is not None and written != int(expected):
        os.remove(part)
        raise IOError(f"download of {filename} incomplete: got {written} of {expected} bytes")
    os.replace(part, dest)
    return dest
