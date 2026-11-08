"""CPU tests for the parquet row source, the heartbeat, and the probe. No network, no GPU.

Run from the repo root:
    python -m unittest ray_image.test_parquet_source -v

torch, datasets and huggingface_hub are stubbed when they are not installed, so the build script
can be imported on a machine without them. Only the stubs are fake; pyarrow and Pillow are real.
"""

import io
import importlib
import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _install_stubs_if_needed():
    for name in ("torch", "datasets", "huggingface_hub", "ray_image.text_encoder", "ray_image.vae_v3"):
        try:
            importlib.import_module(name)
        except Exception:
            pass
        else:
            continue
    if "torch" not in sys.modules:
        try:
            importlib.import_module("torch")
        except Exception:
            torch = types.ModuleType("torch")
            torch.float16 = "float16"
            torch.bool = "bool"
            torch.device = lambda x: x
            torch.Tensor = object
            torch.no_grad = torch.inference_mode = lambda: (lambda f: f)
            sys.modules["torch"] = torch
            nn = types.ModuleType("torch.nn")
            nn.Module = object
            sys.modules["torch.nn"] = nn
            torch.nn = nn
    if "datasets" not in sys.modules:
        try:
            importlib.import_module("datasets")
        except Exception:
            ds = types.ModuleType("datasets")
            ds.load_dataset = lambda *a, **k: None
            sys.modules["datasets"] = ds
    if "huggingface_hub" not in sys.modules:
        try:
            importlib.import_module("huggingface_hub")
        except Exception:
            hub = types.ModuleType("huggingface_hub")
            hub.HfApi = object
            hub.constants = types.SimpleNamespace(HF_HUB_CACHE="/nonexistent/hub")
            hub.create_repo = hub.hf_hub_download = lambda *a, **k: None
            sys.modules["huggingface_hub"] = hub
    for mod_name, attrs in (("ray_image.text_encoder", {"RAYTextEncoder": object}),
                            ("ray_image.vae_v3", {"IMAGE_SIZE": 512, "LATENT_SIZE": 64, "RAYVAE_v3": object})):
        if mod_name not in sys.modules:
            try:
                importlib.import_module(mod_name)
            except Exception:
                mod = types.ModuleType(mod_name)
                for key, value in attrs.items():
                    setattr(mod, key, value)
                sys.modules[mod_name] = mod


_install_stubs_if_needed()

from ray_image import parquet_source as ps  # noqa: E402
from ray_image.parquet_source import FailedRow, ParquetRowSource  # noqa: E402


def png_bytes(color, size=4):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def make_files(directory, layout):
    """layout: list of files; each file is a list of row groups; each row group is a list of
    (caption, image_bytes_or_None) rows. Returns the local paths in order."""
    paths = []
    counter = 0
    for f_index, groups in enumerate(layout):
        path = os.path.join(directory, f"data-{f_index:05d}.parquet")
        writer = None
        for group in groups:
            captions = [c for c, _ in group]
            images = [{"bytes": b, "path": None} if b is not None else None for _, b in group]
            table = pa.table({
                "image": pa.array(images, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                "caption": pa.array(captions, type=pa.string()),
                "row_id": pa.array(list(range(counter, counter + len(group))), type=pa.int64()),
            })
            counter += len(group)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
        writer.close()
        paths.append(path)
    return paths


class ParquetSourceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.logs = []
        self.sleeps = []

    def tearDown(self):
        self._tmp.cleanup()

    def log(self, *args, **kwargs):
        self.logs.append(" ".join(str(a) for a in args))

    def source(self, paths, start=0, **kwargs):
        names = [os.path.basename(p) for p in paths]
        by_name = dict(zip(names, paths))
        kw = dict(fetch_file=lambda name: by_name[name], start=start, sleep=self.sleeps.append,
                  log=self.log, backoff_s=0.0)
        kw.update(kwargs)
        return ParquetRowSource("local/test", names, ["image", "caption", "row_id"], "image", **kw)

    def simple_layout(self):
        # 3 files, row groups of 3, 4 and 2 rows; 2 + 3 + 1 groups.
        return [
            [[("a%d" % i, png_bytes((i, 0, 0))) for i in range(3)],
             [("b%d" % i, png_bytes((0, i, 0))) for i in range(4)]],
            [[("c%d" % i, png_bytes((0, 0, i))) for i in range(3)],
             [("d%d" % i, png_bytes((i, i, 0))) for i in range(2)],
             [("e0", png_bytes((9, 9, 9)))]],
            [[("f%d" % i, png_bytes((5, 5, i))) for i in range(4)]],
        ]

    def test_reads_every_row_in_global_order(self):
        paths = make_files(self.dir, self.simple_layout())
        src = self.source(paths)
        self.assertEqual(src.total_rows, 3 + 4 + 3 + 2 + 1 + 4)
        rows = [next(src) for _ in range(src.total_rows)]
        self.assertEqual([r["row_id"] for r in rows], list(range(src.total_rows)))
        self.assertEqual(rows[0]["caption"], "a0")
        self.assertEqual(rows[-1]["caption"], "f3")
        self.assertIsInstance(rows[0]["image"], Image.Image)
        self.assertEqual(src.position, src.total_rows)
        with self.assertRaises(StopIteration):
            next(src)

    def test_start_in_the_middle_of_a_row_group(self):
        paths = make_files(self.dir, self.simple_layout())
        src = self.source(paths, start=5)  # row 5 is the third row of the first file's second group
        first = next(src)
        self.assertEqual(first["row_id"], 5)
        self.assertEqual(first["caption"], "b2")  # groups: a0-a2 | b0-b3 (rows 3-6)

    def test_skip_to_moves_forward_only(self):
        paths = make_files(self.dir, self.simple_layout())
        src = self.source(paths)
        next(src)
        src.skip_to(10)
        self.assertEqual(next(src)["row_id"], 10)
        with self.assertRaises(ValueError):
            src.skip_to(3)

    def test_bad_image_bytes_become_none_and_do_not_stop_the_read(self):
        layout = [[[("good", png_bytes((1, 2, 3))), ("garbage", b"not a png"), ("empty", b""),
                    ("missing", None), ("good2", png_bytes((4, 5, 6)))]]]
        paths = make_files(self.dir, layout)
        src = self.source(paths)
        rows = [next(src) for _ in range(5)]
        self.assertIsInstance(rows[0]["image"], Image.Image)
        self.assertIsNone(rows[1]["image"])
        self.assertIsNone(rows[2]["image"])
        self.assertIsNone(rows[3]["image"])
        self.assertIsInstance(rows[4]["image"], Image.Image)
        self.assertEqual(src.skipped_rows, 0)

    def test_unreadable_row_group_yields_failed_rows_for_its_rows_only(self):
        paths = make_files(self.dir, self.simple_layout())
        real = pq.ParquetFile

        # Break only the second file's second row group (rows 3+4+3..3+4+3+2 = rows 10..11 in
        # the 'd' group of file 1). Patch by file path via a closure.
        target = paths[1]

        class Targeted(real):
            def __init__(self, path, *a, **k):
                super().__init__(path, *a, **k)
                self._is_target = os.path.abspath(str(path)) == os.path.abspath(target)

            def read_row_group(self, rg, *a, **k):
                if self._is_target and rg == 1:
                    raise OSError("simulated corrupt page")
                return super().read_row_group(rg, *a, **k)

        with mock.patch.object(ps.pq, "ParquetFile", Targeted):
            src = self.source(paths, max_retries=2)
            rows = [next(src) for _ in range(src.total_rows)]
        failed = [r for r in rows if isinstance(r, FailedRow)]
        self.assertEqual([r.position for r in failed], [10, 11])
        self.assertIn("simulated corrupt page", failed[0].error)
        good_ids = [r["row_id"] for r in rows if not isinstance(r, FailedRow)]
        self.assertEqual(good_ids, [i for i in range(src.total_rows) if i not in (10, 11)])
        self.assertEqual(src.skipped_rows, 2)
        self.assertEqual(len(self.sleeps), 2)  # two retries, backed off
        self.assertTrue(any("unreadable after 2 retries" in line for line in self.logs))

    def test_transient_read_error_is_retried_and_succeeds(self):
        paths = make_files(self.dir, self.simple_layout())
        real = pq.ParquetFile
        calls = {"n": 0}

        class Flaky(real):
            def read_row_group(self, rg, *a, **k):
                if rg == 0 and calls["n"] == 0:
                    calls["n"] += 1
                    raise OSError("connection reset")
                return super().read_row_group(rg, *a, **k)

        with mock.patch.object(ps.pq, "ParquetFile", Flaky):
            src = self.source(paths)
            rows = [next(src) for _ in range(src.total_rows)]
        self.assertFalse(any(isinstance(r, FailedRow) for r in rows))
        self.assertEqual(src.retries, 1)
        self.assertEqual(src.skipped_rows, 0)

    def test_missing_column_is_a_clear_error(self):
        paths = make_files(self.dir, self.simple_layout())
        with self.assertRaises(ValueError) as ctx:
            names = [os.path.basename(p) for p in paths]
            by_name = dict(zip(names, paths))
            ParquetRowSource("local/test", names, ["image", "text_missing"], "image",
                             fetch_file=lambda n: by_name[n], sleep=lambda s: None, log=self.log)
        self.assertIn("text_missing", str(ctx.exception))

    def test_start_past_the_end_is_empty(self):
        paths = make_files(self.dir, self.simple_layout())
        src = self.source(paths, start=999)
        with self.assertRaises(StopIteration):
            next(src)


class HeartbeatTests(unittest.TestCase):
    def setUp(self):
        from ray_image.build_cache_v5 import Heartbeat
        self.Heartbeat = Heartbeat
        self.now = [1000.0]
        self.lines = []

    def clock(self):
        return self.now[0]

    def make(self):
        return self.Heartbeat(60, 300, log=self.lines.append, clock=self.clock)

    def test_heartbeat_reports_idle_time_and_position(self):
        hb = self.make()
        hb.touch(42)
        self.now[0] += 61
        hb.check()
        self.assertIn("[heartbeat]", self.lines[-1])
        self.assertIn("42", self.lines[-1])
        self.assertIn("61s", self.lines[-1])

    def test_stall_is_printed_once_and_clears_when_a_row_arrives(self):
        hb = self.make()
        hb.touch(7)
        self.now[0] += 301
        hb.check()
        hb.check()
        stalls = [line for line in self.lines if line.startswith("[stall]")]
        self.assertEqual(len(stalls), 1)
        self.assertIn("last row read: 7", stalls[0])
        self.assertEqual(hb.stall_count, 1)
        hb.touch(8)
        self.now[0] += 301
        hb.check()
        self.assertEqual(hb.stall_count, 2)

    def test_thread_starts_and_stops(self):
        hb = self.Heartbeat(0.01, 300, log=self.lines.append).start()
        hb.stop()
        hb._thread.join(timeout=2)
        self.assertFalse(hb._thread.is_alive())


class ProbeTests(unittest.TestCase):
    def test_probe_reports_outcomes_without_models(self):
        from ray_image import build_cache_v5 as b

        with tempfile.TemporaryDirectory() as d:
            layout = [[[("a red square on a white background", png_bytes((1, 1, 1))),
                        ("", png_bytes((2, 2, 2))),
                        ("a caption whose image is missing", None),
                        ("a caption with a corrupt image", b"junk"),
                        ("a blue circle on a dark green field", png_bytes((3, 3, 3)))]]]
            paths = make_files(d, layout)
            names = [os.path.basename(p) for p in paths]
            by_name = dict(zip(names, paths))
            src = ParquetRowSource("local/test", names, ["image", "caption", "row_id"], "image",
                                   fetch_file=lambda n: by_name[n], sleep=lambda s: None,
                                   log=lambda *a, **k: None)
            args = Namespace(probe_rows=5, image_column="image", caption_column="caption",
                             caption_template=None, start_chunk=0, chunk_size=10_000)
            hb = b.Heartbeat(60, 300, log=lambda message: None)
            out = []
            with mock.patch("builtins.print", lambda *a, **k: out.append(" ".join(str(x) for x in a))):
                # Bind the preprocess step to PIL only (the real one needs torch).
                with mock.patch.object(b, "preprocess_image", lambda im: im.convert("RGB")):
                    b.run_probe(args, src, hb)
            text = "\n".join(out)
            self.assertIn("'ok': 2", text)
            self.assertIn("'empty_caption': 1", text)
            self.assertIn("'no_image': 2", text)  # missing image + corrupt bytes
            self.assertIn("rows 0..5", text)


class ProbeGateTests(unittest.TestCase):
    def _run(self, layout, probe_rows, break_group=None):
        from ray_image import build_cache_v5 as b
        with tempfile.TemporaryDirectory() as d:
            paths = make_files(d, layout)
            names = [os.path.basename(p) for p in paths]
            by_name = dict(zip(names, paths))
            real = pq.ParquetFile

            class Broken(real):
                def read_row_group(self, rg, *a, **k):
                    if break_group is not None and rg == break_group:
                        raise OSError("simulated corrupt page")
                    return super().read_row_group(rg, *a, **k)

            src = ParquetRowSource("local/test", names, ["image", "caption", "row_id"], "image",
                                   fetch_file=lambda n: by_name[n], sleep=lambda s: None,
                                   log=lambda *a, **k: None, max_retries=0)
            args = Namespace(probe_rows=probe_rows, image_column="image", caption_column="caption",
                             caption_template=None, start_chunk=0, chunk_size=10_000)
            hb = b.Heartbeat(60, 300, log=lambda message: None)
            with mock.patch.object(ps.pq, "ParquetFile", Broken), \
                    mock.patch.object(b, "preprocess_image", lambda im: im.convert("RGB")), \
                    mock.patch("builtins.print", lambda *a, **k: None):
                return b.run_probe(args, src, hb)

    good = [[[("a red square on a white background", png_bytes((1, 1, 1)))]]]

    def test_gate_passes_on_clean_rows(self):
        self.assertTrue(self._run(self.good, 1))

    def test_gate_fails_on_unreadable_row_group(self):
        layout = [[[("a red square on a white background", png_bytes((1, 1, 1)))],
                   [("a blue circle on a dark green field", png_bytes((2, 2, 2)))]]]
        self.assertFalse(self._run(layout, 2, break_group=1))

    def test_gate_fails_when_source_is_too_short(self):
        self.assertFalse(self._run(self.good, 5))


class SharedFailedRowTests(unittest.TestCase):
    def test_build_script_uses_the_parquet_source_failed_row(self):
        from ray_image import build_cache_v5 as b
        self.assertIs(b.FailedRow, ps.FailedRow)

class ChunkFolderTests(unittest.TestCase):
    def test_natural_order_and_glob(self):
        from ray_image.parquet_source import chunk_files, natural_key
        names = ["v/cc12m/10-0/000000.parquet", "v/cc12m/2-0/000000.parquet",
                 "v/cc12m/14-0/000010.parquet", "v/cc12m/14-0/000002.parquet",
                 "v/cc12m/14-0/000001.parquet", "v/cc12m/14-1/000000.parquet",
                 "v/cc12m/1-0/000000.parquet", "README.md"]
        self.assertEqual(
            chunk_files(names, "v/cc12m/14-0/*.parquet"),
            ["v/cc12m/14-0/000001.parquet", "v/cc12m/14-0/000002.parquet", "v/cc12m/14-0/000010.parquet"])
        self.assertEqual(sorted(names, key=natural_key)[:3],
                         ["README.md", "v/cc12m/1-0/000000.parquet", "v/cc12m/2-0/000000.parquet"])
        with self.assertRaises(RuntimeError):
            chunk_files(names, "v/cc12m/99-0/*.parquet")

    def _chunk_args(self, chunk_size):
        return Namespace(dataset_name="local/test", parquet_chunk_glob="folder/{chunk}-0/*.parquet",
                         image_column="image", caption_column="caption", caption_template=None,
                         chunk_size=chunk_size)

    def _folder(self, d, rows, chunk=0):
        """Write one folder 'folder/<chunk>-0/' with files of the given row counts."""
        paths, mapping = [], {}
        for f_index, n in enumerate(rows):
            name = f"folder/{chunk}-0/{f_index:06d}.parquet"
            path = os.path.join(d, f"{f_index}.parquet")
            table = pa.table({
                "image": pa.array([{"bytes": png_bytes((1, 1, 1)), "path": None}] * n,
                                  type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                "caption": pa.array(["a caption with enough words here"] * n, type=pa.string()),
            })
            pq.write_table(table, path)
            paths.append(name)
            mapping[name] = path
        return paths, mapping

    def test_chunk_source_checks_row_count(self):
        from ray_image import build_cache_v5 as b
        with tempfile.TemporaryDirectory() as d:
            names, mapping = self._folder(d, [3, 2], chunk=0)  # 5 rows in total
            args = self._chunk_args(chunk_size=5)
            with mock.patch.object(ps.ParquetRowSource, "_download", lambda self, p: mapping[p]):
                src = b.make_chunk_source(args, 0, names, log=lambda *a, **k: None)
                self.assertEqual(src.total_rows, 5)
                args.chunk_size = 6
                with self.assertRaises(RuntimeError) as ctx:
                    b.make_chunk_source(args, 0, names, log=lambda *a, **k: None)
                self.assertIn("folder-to-chunk mapping", str(ctx.exception))

    def _validate_args(self, **overrides):
        from ray_image import build_cache_v5 as b
        base = dict(caption_template=None, source_offset=0, start_chunk=14, end_chunk=15, chunk_size=10_000,
                    dataset_name="jasperai/monet", batch_size=8, source_format="parquet",
                    parquet_chunk_glob="v1.2.0/cc12m/{chunk}-0/*.parquet", probe_rows=0,
                    heartbeat_s=60.0, stall_s=300.0, no_hf_upload=False, hf_repo="x/y", image_split="train",
                    random_sample=False, random_sample_glob="v1.2.0/cc12m/*.parquet",
                    random_sample_files=8, max_skipped=2000)
        base.update(overrides)
        b.validate_args(Namespace(**base))

    def test_validate_parquet_arguments(self):
        self._validate_args()  # the intended launch shape is accepted
        with self.assertRaises(ValueError):
            self._validate_args(parquet_chunk_glob="v1.2.0/cc12m/**/*.parquet")  # no {chunk}: the old OOM case
        with self.assertRaises(ValueError):
            self._validate_args(source_offset=140000)
        with self.assertRaises(ValueError):
            self._validate_args(parquet_chunk_glob="v1.2.0/{shard}/*.parquet")


class RandomSampleTests(unittest.TestCase):
    """Random-sample mode: valid rows only, no repeats, abandon past the rejection limit."""

    GOOD = "a red square on a white background"
    BAD = "bad caption"  # rejected by the validator below

    def validate(self, row):
        if row.get("image") is None:
            return False, "no_image"
        if row["caption"].startswith("bad"):
            return False, "bad_caption"
        return True, None

    def _layout(self):
        # file0: group0 rows 0-4 (rows 3-4 bad), group1 rows 5-6 good; file1: rows 7-9 (row 9 bad)
        def row(i, bad):
            return (self.BAD if bad else self.GOOD, png_bytes((i, i, i)))
        return [
            [[row(0, 0), row(1, 0), row(2, 0), row(3, 1), row(4, 1)], [row(5, 0), row(6, 0)]],
            [[row(7, 0), row(8, 0), row(9, 1)]],
        ]

    def _source(self, paths, target, max_skipped, seed=0, fetch=None):
        from ray_image.parquet_source import RandomSampleSource
        names = [os.path.basename(p) for p in paths]
        by_name = dict(zip(names, paths))
        return RandomSampleSource("local/test", names, ["image", "caption", "row_id"], "image",
                                  validate=self.validate, target=target, max_skipped=max_skipped, seed=seed,
                                  fetch_file=fetch or (lambda n: by_name[n]), sleep=lambda s: None,
                                  log=lambda *a, **k: None)

    def test_yields_exactly_target_valid_unique_rows(self):
        with tempfile.TemporaryDirectory() as d:
            paths = make_files(d, self._layout())
            for seed in range(5):
                src = self._source(paths, target=5, max_skipped=100, seed=seed)
                rows = list(src)
                self.assertEqual(len(rows), 5)
                self.assertEqual(len({r["row_id"] for r in rows}), 5)
                self.assertTrue(all(r["caption"] == self.GOOD for r in rows))
                self.assertLessEqual(src.skipped_rows, 3)

    def test_running_out_of_rows_ends_iteration(self):
        with tempfile.TemporaryDirectory() as d:
            paths = make_files(d, self._layout())
            src = self._source(paths, target=50, max_skipped=100)
            rows = list(src)
            self.assertEqual(sorted(r["row_id"] for r in rows), [0, 1, 2, 5, 6, 7, 8])  # all good rows

    def test_abandons_past_the_rejection_limit(self):
        from ray_image.parquet_source import ChunkAbandoned
        layout = [[[(self.BAD, png_bytes((i, 0, 0))) for i in range(10)]]]
        with tempfile.TemporaryDirectory() as d:
            paths = make_files(d, layout)
            src = self._source(paths, target=1, max_skipped=2)
            with self.assertRaises(ChunkAbandoned):
                next(src)
            self.assertEqual(src.skipped_rows, 3)
            self.assertEqual(src.reject_reasons["bad_caption"], 3)

    def test_unreadable_row_group_is_rejected_not_yielded(self):
        layout = [[[(self.GOOD, png_bytes((i, 0, 0))) for i in range(3)],
                   [(self.GOOD, png_bytes((i, 1, 0))) for i in (3, 4)]],
                  [[(self.GOOD, png_bytes((i, 2, 0))) for i in (5, 6, 7)]]]
        with tempfile.TemporaryDirectory() as d:
            paths = make_files(d, layout)
            real = pq.ParquetFile
            target_path = os.path.abspath(paths[0])

            class Targeted(real):
                def __init__(self, path, *a, **k):
                    super().__init__(path, *a, **k)
                    self._bad = os.path.abspath(str(path)) == target_path

                def read_row_group(self, rg, *a, **k):
                    if self._bad and rg == 1:
                        raise OSError("simulated corrupt page")
                    return super().read_row_group(rg, *a, **k)

            with mock.patch.object(ps.pq, "ParquetFile", Targeted):
                src = self._source(paths, target=8, max_skipped=10)
                rows = list(src)
            self.assertNotIn(3, [r["row_id"] for r in rows])
            self.assertNotIn(4, [r["row_id"] for r in rows])
            self.assertEqual(src.reject_reasons["read_error"], 2)
            self.assertEqual(len(rows), 6)  # 8 rows total minus the 2 in the broken group

    def test_chunk_files_are_disjoint_and_deterministic(self):
        from ray_image.parquet_source import random_chunk_files
        all_files = [f"v1.2.0/cc12m/{g}-0/{f:06d}.parquet" for g in range(20) for f in range(8)] + \
            ["v1.2.0/other/0.parquet", "README.md"]
        pattern = "v1.2.0/cc12m/*.parquet"
        slices = [random_chunk_files(all_files, pattern, seed=7, index=i, files_per_chunk=8) for i in range(20)]
        flat = [f for s_ in slices for f in s_]
        self.assertEqual(len(flat), len(set(flat)))  # no file in two chunks
        self.assertTrue(all(f.startswith("v1.2.0/cc12m/") for f in flat))
        self.assertEqual(slices[3], random_chunk_files(all_files, pattern, seed=7, index=3, files_per_chunk=8))
        self.assertNotEqual(slices[0], random_chunk_files(all_files, pattern, seed=8, index=0, files_per_chunk=8))
        with self.assertRaises(RuntimeError):
            random_chunk_files(all_files, pattern, seed=7, index=20, files_per_chunk=8)

    def test_validator_reasons(self):
        from ray_image import build_cache_v5 as b
        args = Namespace(image_column="image", caption_column="caption", caption_template=None)
        validate = b.make_validator(args)
        self.assertEqual(validate({"image": None, "caption": "a long enough caption here"}), (False, "no_image"))
        self.assertEqual(validate({"image": object(), "caption": "  "}), (False, "empty_caption"))
        self.assertEqual(validate({"image": object(), "caption": "too short"}), (False, "caption_dropped"))
        self.assertEqual(validate({"image": object(), "caption": "a long enough caption here"}), (True, None))

    def test_random_mode_argument_checks(self):
        ChunkFolderTests._validate_args(self, random_sample=True, parquet_chunk_glob="", source_offset=0)
        with self.assertRaises(ValueError):
            ChunkFolderTests._validate_args(self, random_sample=True, random_sample_files=0)
        with self.assertRaises(ValueError):
            ChunkFolderTests._validate_args(self, random_sample=True, max_skipped=-1)
        with self.assertRaises(ValueError):
            ChunkFolderTests._validate_args(self, random_sample=True, source_format="datasets")

class SourceColumnsTests(unittest.TestCase):
    def test_columns_include_only_needed_fields(self):
        from ray_image import build_cache_v5 as b
        args = Namespace(image_column="image", caption_column="caption", caption_template=None)
        self.assertEqual(b.source_columns(args), ["caption", "image"])
        args = Namespace(image_column="img", caption_column="caption",
                         caption_template="a painting by {artist}, {style}")
        self.assertEqual(b.source_columns(args), ["artist", "img", "style"])


if __name__ == "__main__":
    unittest.main()
