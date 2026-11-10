"""Hugging Face downloads over plain HTTP with requests.

Replaces huggingface_hub.hf_hub_download on Kaggle, where that function fails with
"cannot import name 'XetDownloadProgressReporter'". Uploads and file listing still use HfApi.

Only the parts the training code uses are supported: repo_id, filename, repo_type,
token, revision, local_dir, cache_dir. Other keyword arguments are accepted and ignored.

Downloads resume. A dropped connection keeps the bytes already written to <file>.part, and the
next attempt asks for the rest with an HTTP Range header. Short drops retry after RETRY_SLEEP.
Server errors (429, 5xx) back off for RATE_LIMIT_SLEEP. Only a run with no progress across
MAX_ATTEMPTS tries gives up.
"""

import os
import time

import requests

HF_BASE = "https://huggingface.co"
DEFAULT_CACHE = "/kaggle/tmp/hf_rest_cache"
CHUNK_BYTES = 1024 * 1024  # block size for reads; a drop loses at most one block of progress
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 60
RETRY_SLEEP = 0.1       # after a dropped connection
RATE_LIMIT_SLEEP = 2.0  # after 429 or 5xx
MAX_ATTEMPTS = 400      # consecutive attempts with no new bytes before giving up


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


def _total_size(resp, offset):
    """Full file size from a 200 or 206 response, or None if the server does not say."""
    content_range = resp.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        if total.isdigit():
            return int(total)
    length = resp.headers.get("Content-Length")
    if length is not None and length.isdigit():
        return offset + int(length)
    return None


def _download_resumable(url, headers, dest):
    """Write url to dest, resuming from dest + '.part' after drops. Returns when complete."""
    part = dest + ".part"
    total = None
    best_offset = -1
    stalled = 0
    while True:
        offset = os.path.getsize(part) if os.path.exists(part) else 0
        if total is not None and offset == total:
            break
        if offset > best_offset:
            best_offset = offset
            stalled = 0
        stalled += 1
        if stalled > MAX_ATTEMPTS:
            raise IOError(f"giving up on {url} after {MAX_ATTEMPTS} attempts with no progress "
                          f"at byte {offset}")

        req_headers = dict(headers)
        if offset:
            req_headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=req_headers, stream=True, allow_redirects=True,
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
                if resp.status_code == 416:
                    # Range past the end: the .part file does not match the server. Start over.
                    os.remove(part)
                    total = None
                    continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue
                resp.raise_for_status()
                if resp.status_code == 200 and offset:
                    # Server ignored Range, so the body is the whole file. Start over.
                    offset = 0
                total = _total_size(resp, offset)
                mode = "ab" if offset else "wb"
                with open(part, mode) as out:
                    for chunk in resp.iter_content(CHUNK_BYTES):
                        if chunk:
                            out.write(chunk)
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError):
            time.sleep(RETRY_SLEEP)
            continue
        if total is None:
            # No size from the server, so one complete response is all we can trust.
            break
    os.replace(part, dest)


def hf_hub_download(repo_id, filename, repo_type=None, token=None, revision="main",
                    local_dir=None, cache_dir=None, **_ignored):
    """Download one file and return its local path, like huggingface_hub.hf_hub_download.

    The file is written to <local_dir>/<filename> when local_dir is given. Otherwise it goes
    under DEFAULT_CACHE (or cache_dir). An existing non-empty file is reused.
    """
    rtype = repo_type or "model"
    if local_dir:
        dest = os.path.join(local_dir, filename)
    else:
        dest = os.path.join(cache_dir or DEFAULT_CACHE, rtype, repo_id, filename)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    url = resolve_url(repo_id, filename, rtype, revision)
    _download_resumable(url, _headers(token), dest)
    return dest
