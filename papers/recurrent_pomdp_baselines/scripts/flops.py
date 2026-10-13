"""FLOPs per gradient update for each full stage arm (compute-parity table, the README).

XLA's cost analysis counts a while-loop body ONCE, so the RNN scans are fully unrolled (scan_unroll >= L+1) at two
short context lengths (L = 5 and 10, T = L+1 steps) and the per-update FLOPs are extrapolated linearly in T (every
term of the update is either T-independent or linear in T: embedders/heads are applied to all T steps, the RNN
forward/backward unrolls are T steps).  CPU only (no GPU use).  Writes runs/full/flops.json.
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import info, load_config  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.train import Runner  # noqa: E402


def flops_one_update(cfg_name, over):
    cfg = load_config(cfg_name, over)
    R = Runner(cfg)
    ts = R.init_state(0, (1.0, 1.0))
    f = jax.jit(lambda t: R._one_update(t, False))   # full output, so nothing is dead-code eliminated
    ca = f.lower(ts).compile().cost_analysis()
    ca = ca[0] if isinstance(ca, (list, tuple)) else ca
    return float(ca["flops"])


def main():
    out = {}
    cells = [("pendulum_v.yaml", a, arch, c) for a in ("sac", "td3") for arch in ("separate", "shared")
             for c in ("lstm", "gru")]
    cells += [("semicircle.yaml", "td3", arch, "lstm") for arch in ("separate", "shared")]
    for cfg_name, algo, arch, cell in cells:
        pts = {}
        for L in (5, 10):
            over = [f"algo={algo}", f"arch={arch}", f"cell={cell}", f"sampled_seq_len={L}", f"scan_unroll={L + 1}"]
            pts[L + 1] = flops_one_update(cfg_name, over)
        slope = (pts[11] - pts[6]) / 5.0
        icpt = pts[6] - 6 * slope
        Ls = (5, 64, 200) if cfg_name.startswith("pendulum") else (64,)
        key = f"{cfg_name.split('.')[0]}/{algo}-{cell}/{arch}"
        out[key] = {"flops_T6": pts[6], "flops_T11": pts[11], "per_step": slope, "const": icpt,
                    **{f"gflops_per_update_L{L}": (icpt + slope * (L + 1)) / 1e9 for L in Ls}}
        info("flops", cell=key, **{k: round(v, 3) for k, v in out[key].items() if k.startswith("gflops")})
    path = os.path.join(common.REPO, "runs", "full", "flops.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
