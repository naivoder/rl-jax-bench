"""Full-fp32 stage (review resolution; pre-declared in the README before launch).

Pendulum-V sac-lstm-200-oar at the paper's full horizon (51k env steps), FRESH seeds 8-11 (never run in any earlier
phase), from scratch, matmul_precision=highest (true fp32, as on the paper's RTX 2080 Ti) in EVERY arm:
  sep  separate (R = 4)
  sh   shared {official, noleak, detach} (R = 12)
The two groups run as two processes (paired by seed: same key tree -> same init keys, warm-up data and eval keys;
NOT the same arithmetic, because the vmapped batch size R differs -- README 'Reproducibility').

Usage: python scripts/full_fp32.py sep | sh
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import info, load_config  # noqa: E402
from src.train import train  # noqa: E402

OUT = os.path.join(common.REPO, "runs", "full_fp32")
SEEDS = [8, 9, 10, 11]
PREC = "matmul_precision=highest"   # fp32 in both arms (review)
GROUPS = {"sep": ("separate", ["separate"], "P1b_sac_separate"),
          "sh": ("shared", ["official", "noleak", "detach"], "P1b_sac_shared")}


def main(group):
    arch, variants, name = GROUPS[group]
    cfg = load_config("pendulum_v.yaml", ["algo=sac", f"arch={arch}", PREC, "ckpt_every_iters=25"])
    assert int(cfg["num_iters"]) == int(cfg["paper_num_iters"]) == 250
    common.set_precision(cfg["matmul_precision"])
    info("full_fp32 run", name=name, seeds=SEEDS, variants=variants, precision=cfg["matmul_precision"])
    train(cfg, SEEDS, variants, os.path.join(OUT, name), resume=True)


if __name__ == "__main__":
    main(sys.argv[1])
