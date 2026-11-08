"""Row source that reads Parquet files directly, one row group at a time.

Why this exists: `datasets` streaming reopens the dataset with `.skip(position)` after every
failure, so a bad row deep in a dataset costs a re-scan from row 0 each time, and a cast error
inside the `datasets` feature layer can stall the whole build. This source reads the Parquet
footers once, then reads row groups with pyarrow. Only the requested columns are read.

It has the same interface as build_cache_v5.SafeIterator (position, skip_to, retries,
skipped_rows, __next__ yielding a row dict or a FailedRow), so build_chunk can use either.

Failure policy:
- A row group that cannot be read is retried (download and read), then every row it holds is
  yielded as a FailedRow. Chunk boundaries stay exact because each row still takes one position.
- A row whose image bytes cannot be decoded is yielded as a dict with image=None, so build_chunk
  counts it as a skip and keeps going.

The source only ever sees the files it is given. build_cache_v5 gives it the files of one chunk
(for example every 000000..000007 file in folder 14-0), so the footer pass touches a few files, not
the whole dataset. Files are ordered by natural sort (2-0 before 10-0, 000002 before 000010).
"""

import bisect
import fnmatch
import io
import random
import re
import time
from collections import Counter

import pyarrow.parquet as pq
from PIL import Image

MAX_RETRIES = 3
RETRY_BACKOFF_S = 5.0


class FailedRow:
    """Yielded in place of a source row whose row group could not be read after all retries."""

    def __init__(self, position: int, error: str):
        self.position = position
        self.error = error


def list_repo_files(repo_id: str, token=None) -> list:
    """All file paths in a dataset repo (one API call, no downloads)."""
    from huggingface_hub import HfApi

    return HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="dataset")


def natural_key(path: str):
    """Sort key that orders digit runs by value: 'x/2-0' < 'x/10-0', '000002' < '000010'."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def chunk_files(all_files, pattern: str) -> list:
    """Repo paths matching the glob, in natural order. Raises if nothing matches."""
    matched = sorted((f for f in all_files if fnmatch.fnmatch(f, pattern)), key=natural_key)
    if not matched:
        raise RuntimeError(f"no files match {pattern!r}")
    return matched


def decode_image(value):
    """Turn a Parquet image cell into a PIL image. Accepts {'bytes': ...} structs or raw bytes."""
    if value is None:
        return None
    data = value.get("bytes") if isinstance(value, dict) else value
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        return None
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


class ParquetRowSource:
    """Iterate rows of a list of Parquet files as one continuous row space.

    fetch_file(repo_path) -> local path. The default downloads via huggingface_hub; tests pass
    a function that returns a local file.
    """

    def __init__(self, repo_id, files, columns, image_column, fetch_file=None, start: int = 0,
                 max_retries: int = MAX_RETRIES, backoff_s: float = RETRY_BACKOFF_S,
                 sleep=time.sleep, log=print, token=None, on_row=None):
        self._repo_id = repo_id
        self._files = list(files)
        self._columns = list(columns)
        self._image_column = image_column
        if image_column not in self._columns:
            raise ValueError("image_column must be one of columns")
        self._fetch = fetch_file or self._download
        self._token = token
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._log = log
        self._on_row = on_row
        self.retries = 0
        self.skipped_rows = 0
        self.position = start

        # Footer-only pass: row group sizes and their global starts. Files are downloaded
        # one at a time here (the cache is purged between chunks anyway).
        self._groups = []  # (file_index, row_group_index, global_start, num_rows)
        self._group_starts = []
        total = 0
        for file_index, repo_path in enumerate(self._files):
            local = self._fetch(repo_path)
            meta = pq.ParquetFile(local).metadata
            for rg in range(meta.num_row_groups):
                n = meta.row_group(rg).num_rows
                if n == 0:
                    continue
                self._groups.append((file_index, rg, total, n))
                self._group_starts.append(total)
                total += n
        self.total_rows = total
        self._loaded_group = None  # index into self._groups
        self._table = None
        self._table_error = None
        self._check_columns(self._files[0])

    def _check_columns(self, repo_path):
        local = self._fetch(repo_path)
        names = set(pq.ParquetFile(local).schema_arrow.names)
        missing = [c for c in self._columns if c not in names]
        if missing:
            raise ValueError(f"{repo_path} has no column(s) {missing}; available: {sorted(names)}")

    def _download(self, repo_path):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=self._repo_id, filename=repo_path, repo_type="dataset",
                               token=self._token)

    def skip_to(self, position: int) -> None:
        if position < self.position:
            raise ValueError(f"cannot rewind source from row {self.position} to {position}")
        self.position = position

    def __iter__(self):
        return self

    def _group_for(self, position: int) -> int:
        return bisect.bisect_right(self._group_starts, position) - 1

    def _load_group(self, group_index: int):
        """Read the row group into an Arrow table. Retries; returns (table, None) or (None, error)."""
        if self._loaded_group == group_index:
            return self._table, self._table_error
        file_index, rg, _, _ = self._groups[group_index]
        last_error = None
        for attempt in range(1, self._max_retries + 2):
            try:
                local = self._fetch(self._files[file_index])
                table = pq.ParquetFile(local).read_row_group(rg, columns=self._columns)
                self._loaded_group, self._table, self._table_error = group_index, table, None
                return table, None
            except Exception as exc:  # corrupt page, truncated download, network drop
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= self._max_retries:
                    self.retries += 1
                    self._log(f"[parquet] {self._files[file_index]} row group {rg}: {last_error}; "
                              f"retry {attempt}/{self._max_retries}")
                    self._sleep(self._backoff_s * attempt)
        self._log(f"[parquet] {self._files[file_index]} row group {rg} unreadable after "
                  f"{self._max_retries} retries; its rows are marked failed")
        self._loaded_group, self._table, self._table_error = group_index, None, last_error
        return None, last_error

    def __next__(self):
        if self.position >= self.total_rows:
            raise StopIteration
        position = self.position
        group_index = self._group_for(position)
        _, _, group_start, _ = self._groups[group_index]
        table, error = self._load_group(group_index)
        self.position += 1
        if self._on_row is not None:
            self._on_row(position)
        if table is None:
            self.skipped_rows += 1
            return FailedRow(position, error or "row group unreadable")
        row_in_group = position - group_start
        row = {name: table.column(name)[row_in_group].as_py() for name in self._columns}
        row[self._image_column] = self._decode_or_none(row[self._image_column], position)
        return row

    def _decode_or_none(self, value, position):
        try:
            return decode_image(value)
        except Exception as exc:
            self._log(f"[parquet] row {position}: image decode failed: {type(exc).__name__}: {exc}")
            return None


class ChunkAbandoned(RuntimeError):
    """Too many rows were rejected to build this chunk. Nothing is written for it."""


class RandomSampleSource:
    """Yield `target` valid rows drawn at random, without replacement, from `files`.

    Row groups are visited in random order, and rows inside a group in random order, so one
    chunk mixes all of its files. A candidate is rejected if its read failed or validate(row)
    returns (False, reason). Each rejection counts once. More than `max_skipped` rejections raises
    ChunkAbandoned. Running out of rows before `target` valid ones ends iteration with StopIteration.

    validate(row) -> (ok: bool, reason: str or None). It sees the decoded row; image is a PIL image or None.
    """

    def __init__(self, repo_id, files, columns, image_column, validate, target: int, max_skipped: int,
                 seed: int, fetch_file=None, log=print, token=None,
                 max_retries: int = MAX_RETRIES, backoff_s: float = RETRY_BACKOFF_S, sleep=time.sleep):
        self._inner = ParquetRowSource(repo_id, files, columns, image_column, fetch_file=fetch_file,
                                       start=0, max_retries=max_retries, backoff_s=backoff_s,
                                       sleep=sleep, log=log, token=token)
        self._validate = validate
        self._target = target
        self._max_skipped = max_skipped
        self._rng = random.Random(seed)
        self.total_rows = self._inner.total_rows
        self.accepted = 0
        self.skipped_rows = 0
        self.reject_reasons = Counter()
        self._candidates = self._iter_candidates()

    @property
    def retries(self) -> int:
        return self._inner.retries

    @property
    def position(self) -> int:
        """Candidate rows consumed so far (accepted plus rejected)."""
        return self.accepted + self.skipped_rows

    def __iter__(self):
        return self

    def _iter_candidates(self):
        groups = self._inner._groups
        order = list(range(len(groups)))
        self._rng.shuffle(order)
        for group_index in order:
            _, _, group_start, n_rows = groups[group_index]
            table, error = self._inner._load_group(group_index)
            rows = list(range(n_rows))
            self._rng.shuffle(rows)
            for r in rows:
                position = group_start + r
                if table is None:
                    yield FailedRow(position, error or "row group unreadable")
                    continue
                row = {name: table.column(name)[r].as_py() for name in self._inner._columns}
                row[self._inner._image_column] = self._inner._decode_or_none(
                    row[self._inner._image_column], position)
                yield row

    def _reject(self, reason: str) -> None:
        self.skipped_rows += 1
        self.reject_reasons[reason] += 1
        if self.skipped_rows > self._max_skipped:
            raise ChunkAbandoned(
                f"{self.skipped_rows} rejected rows (limit {self._max_skipped}) after {self.accepted} of "
                f"{self._target} accepted; reasons {dict(self.reject_reasons)}")

    def __next__(self):
        while True:
            if self.accepted >= self._target:
                raise StopIteration
            candidate = next(self._candidates)  # StopIteration here: the files ran out
            if isinstance(candidate, FailedRow):
                self._reject("read_error")
                continue
            ok, reason = self._validate(candidate)
            if not ok:
                self._reject(reason or "invalid")
                continue
            self.accepted += 1
            return candidate


def random_chunk_files(all_files, pattern: str, seed: int, index: int, files_per_chunk: int) -> list:
    """The files for chunk `index` in random-sample mode.

    The matching files are shuffled once with `seed`. Chunk i gets the slice [i*k, (i+1)*k), so no
    file is shared between chunks. Raises if the pool runs out.
    """
    pool = sorted((f for f in all_files if fnmatch.fnmatch(f, pattern)), key=natural_key)
    if not pool:
        raise RuntimeError(f"no files match {pattern!r}")
    random.Random(seed).shuffle(pool)
    start = index * files_per_chunk
    if start + files_per_chunk > len(pool):
        raise RuntimeError(
            f"file pool exhausted: chunk {index} needs files {start}..{start + files_per_chunk} but only "
            f"{len(pool)} files match {pattern!r} ({len(pool) // files_per_chunk} chunks fit)")
    return pool[start:start + files_per_chunk]

