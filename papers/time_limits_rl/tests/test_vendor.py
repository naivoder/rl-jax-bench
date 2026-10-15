"""The vendored PPO-details PPO files must stay identical to their pinned SHA-256."""
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def test_vendored_files_unchanged():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from src.ppo8 import VENDOR_SHA256
    for f, h in VENDOR_SHA256.items():
        with open(os.path.join(REPO, "src", "ppo8", f), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == h, f
