"""Recheck re-runs (pre-declared in the README before launch).

Same configs, seeds (0-3 = the small stage keys) and vmapped batch R = 4 as small stage; ONLY the training precision changes
to matmul_precision=highest (true fp32, as on the paper's RTX 2080 Ti, App. A.2). The pre-fix behaviour is the small stage
TF32 runs (matmul_precision=default), kept unchanged in runs/small/.
  S5sep  Semi-Circle td3-lstm-64-or separate, 150k env steps            ()
  S5sh   Semi-Circle td3-lstm-64-or shared-official, 150k env steps     (, paired P5 at fp32)
  P3sh   Pendulum-V td3-lstm-200-oar shared-official, 20k env steps     (; small stage ran R = 12 with noleak/detach)
Usage: python scripts/recheck_rerun.py S5sep | S5sh | P3sh
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import info, load_config  # noqa: E402
from src.train import train  # noqa: E402

OUT = os.path.join(common.REPO, "runs", "recheck", "rerun")
SEEDS = [0, 1, 2, 3]
PREC = "matmul_precision=highest"
RUNS = {
    "S5sep": ("semicircle.yaml", ["arch=separate", "num_iters=34", "ckpt_every_iters=0", PREC], ["separate"], "S_td3_separate_fp32"),
    "S5sh": ("semicircle.yaml", ["arch=shared", "num_iters=34", "ckpt_every_iters=0", PREC], ["official"], "S_td3_shared_fp32"),
    "P3sh": ("pendulum_v.yaml", ["algo=td3", "arch=shared", "num_iters=95", PREC], ["official"], "P_td3_shared_fp32"),
}


def main(group):
    cfg_name, over, variants, name = RUNS[group]
    cfg = load_config(cfg_name, over)
    common.set_precision(cfg["matmul_precision"])
    info("recheck rerun", name=name, seeds=SEEDS, variants=variants, precision=cfg["matmul_precision"])
    train(cfg, SEEDS, variants, os.path.join(OUT, name), resume=True)


if __name__ == "__main__":
    main(sys.argv[1])
