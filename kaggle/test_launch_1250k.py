"""Offline tests for the validation logic in launch_1250k.py (no network, no Kaggle)."""
import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("launch_1250k", os.path.join(HERE, "launch_1250k.py"))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def cache_files(starts):
    return [f"monet_qwen_cache_{s}k_{s + 10}k.pt" for s in starts]


class CacheCheckTests(unittest.TestCase):
    def test_repo1_complete(self):
        files = cache_files(range(0, 250, 10)) + ["README.md", ".gitattributes"]
        self.assertEqual(len(launch.check_cache_repo(files, launch.EXPECTED_STARTS_1, "r1")), 25)

    def test_repo2_with_chunk_49_missing_is_rejected(self):
        files = cache_files(range(250, 490, 10))  # 250k..480k: 24 files
        with self.assertRaisesRegex(RuntimeError, "found 24"):
            launch.check_cache_repo(files, launch.EXPECTED_STARTS_2, "r2")

    def test_repo2_complete(self):
        files = cache_files(range(250, 500, 10))
        self.assertEqual(len(launch.check_cache_repo(files, launch.EXPECTED_STARTS_2, "r2")), 25)

    def test_right_count_wrong_chunks_is_rejected(self):
        files = cache_files(range(250, 250 + 250, 10))[:24] + ["monet_qwen_cache_0k_10k.pt"]
        with self.assertRaisesRegex(RuntimeError, "missing starts"):
            launch.check_cache_repo(files, launch.EXPECTED_STARTS_2, "r2")

    def test_non_chunk_files_are_ignored(self):
        files = cache_files(range(0, 250, 10)) + ["monet_qwen_cache_0k_10k.pt.tmp", "x.pt"]
        self.assertEqual(len(launch.check_cache_repo(files, launch.EXPECTED_STARTS_1, "r1")), 25)


class CheckpointTests(unittest.TestCase):
    def test_newest_full_checkpoint_wins(self):
        files = ["ray_dit_v5_step_0175000.pt", "ray_dit_v5_step_0190000.pt",
                 "ray_dit_v5_ema_step_0190000.pt", "ray_dit_v5_step_0185000.pt"]
        self.assertEqual(launch.pick_checkpoint(files), (190000, "ray_dit_v5_step_0190000.pt"))

    def test_ema_only_files_are_ignored(self):
        self.assertIsNone(launch.pick_checkpoint(["ray_dit_v5_ema_step_0190000.pt"]))


class CodeCheckTests(unittest.TestCase):
    def test_missing_flags_rejected(self):
        with tempfile.TemporaryDirectory() as repo:
            os.makedirs(os.path.join(repo, "ray_image"))
            with open(os.path.join(repo, "ray_image", "train_dit_v5.py"), "w") as f:
                f.write("print('old commit')\n")
            with self.assertRaisesRegex(RuntimeError, "lacks"):
                launch.verify_code(repo)

    def test_flags_present_accepted(self):
        with tempfile.TemporaryDirectory() as repo:
            os.makedirs(os.path.join(repo, "ray_image"))
            with open(os.path.join(repo, "ray_image", "train_dit_v5.py"), "w") as f:
                f.write("use_sage = 1\nuse_sim_loss = 1\n")
            launch.verify_code(repo)


if __name__ == "__main__":
    unittest.main()
