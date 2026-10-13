"""Small stage: the paper's smallest setting at a ~2.5x / 10x smaller horizon, all arms, 4 seeds (pre-declared in
the README before launch).

  group P (Pendulum-V, 20k env steps = 5 random + 95 x [1 rollout, 200 updates], L = 200, oar, lstm):
     sac / td3 x {separate} and {shared official, shared noleak, shared detach}
  group S (Semi-Circle, 150k env steps = 400 random + 34 x [25 rollouts, 1000 updates], td3-lstm-64-or):
     separate, shared official
References: the authors' logs over the same evaluation window (Pendulum 16k-20k, Semi-Circle 120k-150k env steps)
and the official code's own CPU reference runs (Pendulum-V sac-lstm separate / shared, 20k steps, 2 seeds).
Usage: python scripts/small.py P   |   python scripts/small.py S
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import info, load_config  # noqa: E402
from src.train import train  # noqa: E402

OUT = os.path.join(common.REPO, "runs", "small")
SEEDS = [0, 1, 2, 3]
PREC = "matmul_precision=default"   # TF32 on the RTX 3090 for training runs (README dtype policy)

RUNS = {
    "P": [("pendulum_v.yaml", [f"algo={a}", "arch=separate", "num_iters=95", PREC], ["separate"], f"P_{a}_separate")
          for a in ("sac", "td3")]
         + [("pendulum_v.yaml", [f"algo={a}", "arch=shared", "num_iters=95", PREC], ["official", "noleak", "detach"],
             f"P_{a}_shared") for a in ("sac", "td3")],
    # "Psh": the two shared-encoder Pendulum runs only (run in a second process to overlap with group P's separate runs)
    "Psh": [("pendulum_v.yaml", [f"algo={a}", "arch=shared", "num_iters=95", PREC], ["official", "noleak", "detach"],
             f"P_{a}_shared") for a in ("td3", "sac")],
    "S": [("semicircle.yaml", ["arch=separate", "num_iters=34", "ckpt_every_iters=0", PREC], ["separate"], "S_td3_separate"),
          ("semicircle.yaml", ["arch=shared", "num_iters=34", "ckpt_every_iters=0", PREC], ["official"], "S_td3_shared")],
}


def main(group):
    for cfg_name, over, variants, name in RUNS[group]:
        cfg = load_config(cfg_name, over)
        common.set_precision(cfg["matmul_precision"])
        info("small run", name=name)
        train(cfg, SEEDS, variants, os.path.join(OUT, name), resume=True)


if __name__ == "__main__":
    main(sys.argv[1])
