"""Run ray_image.train_dit_v5 with Hugging Face downloads done by hf_rest.py.

Patches hf_hub_download before the training modules are imported. Those modules import the name
at import time, so they pick up the patched version. Arguments are passed through unchanged:

    python kaggle/train_wrapper.py --resume ... --steps ...
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import huggingface_hub  # noqa: E402
import huggingface_hub.file_download as file_download  # noqa: E402

import hf_rest  # noqa: E402

huggingface_hub.hf_hub_download = hf_rest.hf_hub_download
file_download.hf_hub_download = hf_rest.hf_hub_download

sys.argv = ["ray_image.train_dit_v5"] + sys.argv[1:]
runpy.run_module("ray_image.train_dit_v5", run_name="__main__", alter_sys=True)
