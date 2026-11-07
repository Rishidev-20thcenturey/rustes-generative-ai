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
