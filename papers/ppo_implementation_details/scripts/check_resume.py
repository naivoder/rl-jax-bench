"""Resume check: an interrupted-then-resumed run must reproduce the uninterrupted one exactly (same per-update
metrics and episodes), because every key is a pure function of (seed, update, step, epoch). CPU, ~1 min."""
import json
import os
import shutil
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.train import CONFIG_DIR, load_config, train_one  # noqa: E402

cfg = load_config(os.path.join(CONFIG_DIR, "classic_control.yaml"),
                  ["num_steps=16", "total_timesteps=384", "chunk_updates=2", "seeds=[1,2]"])
root = os.path.join(REPO, "runs", "dev", "resume_check")
shutil.rmtree(root, ignore_errors=True)
train_one(cfg, "a_full", root, ckpt=False)
assert train_one(cfg, "b_resumed", root, ckpt=True, max_chunks=1) is None
assert os.path.isdir(os.path.join(root, "CartPole-v1", "b_resumed", "ckpt"))
train_one(cfg, "b_resumed", root, ckpt=True)
ok = True
for s in [1, 2]:
    a = json.load(open(os.path.join(root, "CartPole-v1", "a_full", f"seed_{s}", "metrics.json")))
    b = json.load(open(os.path.join(root, "CartPole-v1", "b_resumed", f"seed_{s}", "metrics.json")))
    same = a["per_update"] == b["per_update"] and a["episodes"] == b["episodes"]
    ok &= same
    print(f"[METRIC] resume seed={s} identical={same} updates={len(a['per_update']['loss'])}")
print("[INFO] RESUME CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
