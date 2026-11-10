"""Tests for hf_rest.py against a local HTTP server (no network, no Hugging Face)."""
import http.server
import json
import os
import tempfile
import threading
import unittest

import hf_rest

PAYLOAD = b"x" * 100_000
BIG = bytes(range(256)) * 4000          # 1,024,000 bytes, distinct content to check offsets
SEEN_AUTH = []
STATE = {"dropped": False, "range_requests": 0}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        SEEN_AUTH.append(self.headers.get("Authorization"))
        if self.path.startswith("/api/models/Rishix500/demo/tree/main"):
            body = json.dumps([
                {"type": "file", "path": "ray_dit_v5_step_0175000.pt"},
                {"type": "directory", "path": "old"},
                {"type": "file", "path": "README.md"},
            ]).encode()
        elif self.path == "/Rishix500/demo/resolve/main/model.pt":
            body = PAYLOAD
        elif self.path == "/datasets/Rishix500/data/resolve/main/chunk.pt":
            body = b"chunk-bytes"
        elif self.path == "/Rishix500/demo/resolve/main/big.pt":
            rng = self.headers.get("Range")
            if rng:
                STATE["range_requests"] += 1
                start = int(rng.split("=", 1)[1].rstrip("-"))
                part = BIG[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(BIG) - 1}/{len(BIG)}")
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                self.wfile.write(part)
                return
            if not STATE["dropped"]:
                # First full request: promise everything, send 300,000 bytes, then close.
                STATE["dropped"] = True
                self.send_response(200)
                self.send_header("Content-Length", str(len(BIG)))
                self.end_headers()
                self.wfile.write(BIG[:300_000])
                self.wfile.flush()
                self.close_connection = True
                return
            body = BIG
        elif self.path == "/Rishix500/demo/resolve/main/ignore.pt":
            body = BIG  # always a full 200, even when Range is sent
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class HfRestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        hf_rest.HF_BASE = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_download_writes_full_file_into_local_dir(self):
        with tempfile.TemporaryDirectory() as d:
            path = hf_rest.hf_hub_download(repo_id="Rishix500/demo", filename="model.pt",
                                           repo_type="model", token="hf_test", local_dir=d)
            self.assertEqual(path, os.path.join(d, "model.pt"))
            with open(path, "rb") as f:
                self.assertEqual(f.read(), PAYLOAD)
            self.assertFalse(os.path.exists(path + ".part"))

    def test_token_is_sent_as_bearer(self):
        SEEN_AUTH.clear()
        with tempfile.TemporaryDirectory() as d:
            hf_rest.hf_hub_download(repo_id="Rishix500/demo", filename="model.pt",
                                    repo_type="model", token="hf_test", local_dir=d)
        self.assertIn("Bearer hf_test", SEEN_AUTH)

    def test_existing_file_is_reused_without_request(self):
        with tempfile.TemporaryDirectory() as d:
            existing = os.path.join(d, "model.pt")
            with open(existing, "wb") as f:
                f.write(b"already here")
            SEEN_AUTH.clear()
            path = hf_rest.hf_hub_download(repo_id="Rishix500/demo", filename="model.pt",
                                           repo_type="model", local_dir=d)
            self.assertEqual(path, existing)
            self.assertEqual(SEEN_AUTH, [])

    def test_dataset_repo_uses_datasets_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = hf_rest.hf_hub_download(repo_id="Rishix500/data", filename="chunk.pt",
                                           repo_type="dataset", local_dir=d)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"chunk-bytes")

    def test_default_cache_path_when_no_local_dir(self):
        with tempfile.TemporaryDirectory() as d:
            path = hf_rest.hf_hub_download(repo_id="Rishix500/demo", filename="model.pt",
                                           repo_type="model", cache_dir=d)
            self.assertEqual(path, os.path.join(d, "model", "Rishix500/demo", "model.pt"))
            self.assertTrue(os.path.isfile(path))

    def test_missing_file_raises_and_leaves_no_partial(self):
        import requests
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(requests.HTTPError):
                hf_rest.hf_hub_download(repo_id="Rishix500/demo", filename="nope.pt",
                                        repo_type="model", local_dir=d)
            self.assertFalse(os.path.exists(os.path.join(d, "nope.pt")))
            self.assertFalse(os.path.exists(os.path.join(d, "nope.pt.part")))

    def test_dropped_connection_resumes_with_range(self):
        STATE["dropped"] = False
        STATE["range_requests"] = 0
        old_chunk = hf_rest.CHUNK_BYTES
        hf_rest.CHUNK_BYTES = 65536  # small blocks, so the 300,000-byte drop leaves saved bytes
        try:
            with tempfile.TemporaryDirectory() as d:
                path = hf_rest.hf_hub_download(repo_id="Rishix500/demo", filename="big.pt",
                                               repo_type="model", local_dir=d)
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), BIG)
                self.assertGreaterEqual(STATE["range_requests"], 1)
                self.assertFalse(os.path.exists(path + ".part"))
        finally:
            hf_rest.CHUNK_BYTES = old_chunk

    def test_server_ignoring_range_restarts_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            part = os.path.join(d, "ignore.pt.part")
            with open(part, "wb") as f:
                f.write(b"junk-from-an-old-attempt")
            path = hf_rest.hf_hub_download(repo_id="Rishix500/demo", filename="ignore.pt",
                                           repo_type="model", local_dir=d)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), BIG)

    def test_list_repo_files_returns_files_only(self):
        files = hf_rest.list_repo_files("Rishix500/demo", repo_type="model", token="hf_test")
        self.assertEqual(files, ["ray_dit_v5_step_0175000.pt", "README.md"])

    def test_unsupported_repo_type_rejected(self):
        with self.assertRaises(ValueError):
            hf_rest.resolve_url("a/b", "c", repo_type="space")


if __name__ == "__main__":
    unittest.main()
