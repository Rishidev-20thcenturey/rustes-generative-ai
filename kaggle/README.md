# Kaggle launcher: 1.25M baseline

`launch_1250k.py` checks both cache repos (25 chunks each, exact names), checks the code has the
v5 flags, finds the newest full checkpoint on Hugging Face, and then starts training. It stops
before any training if a check fails.

Paste this into a fresh Kaggle notebook cell (needs the `HF_TOKEN` secret):

```python
import subprocess, os
REPO = "/kaggle/working/ray-image-v1"
URL = "https://github.com/Rishidev-20thcenturey/rustes-generative-ai.git"
BRANCH = "arena/416eda21-rustes-generative-ai"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-b", BRANCH, URL, REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO, "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run(["python", "-u", f"{REPO}/kaggle/launch_1250k.py"], check=True)
```

Rerunning the same cell after a session ends resumes from the newest uploaded checkpoint.
Upload gap: checkpoints go to Hugging Face every 5,000 steps, so up to 4,999 steps can be lost.
