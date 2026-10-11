"""Write per-seed metrics of an MNIST run that was stopped at a checkpoint boundary (budget stop, see README).

    JAX_PLATFORMS=cpu uv run python -m scripts.partial_mnist <tag>

Restores the latest orbax checkpoint of runs/<tag> with the run's own config (runs/<tag>/config.yaml), truncates the
per-task history to the tasks completed, and writes runs/<tag>/seed_<n>/metrics.json + summary.json exactly as a
finished run would (src.train.write_mnist_metrics).  No training happens here.
"""
import os
import sys


import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src import data as D  # noqa: E402
from src import train as T  # noqa: E402


def main(tag):
    out = os.path.join(T.REPO, "runs", tag)
    cfg = yaml.safe_load(open(os.path.join(out, "config.yaml")))
    seeds = T.seeds_of(cfg)
    hps = T.arm_hps(cfg)
    static = T.static_of(cfg, 784, 10, "ce")
    n_tasks = int(cfg["n_tasks"])
    states, _, _ = T.init_states(static, hps, seeds, 784, 10)
    S, Aa = len(seeds), len(hps)
    keys_hist = ("acc", "loss", "n_rep", "wmag_mean", "acc_first_10k", "acc_last_10k", "dead_pct", "erank_scaled",
                 "wmag_before", "wmag_after")
    hist = {k: np.zeros((n_tasks, S, Aa), np.float32) for k in keys_hist}
    hist["dead_per_layer"] = np.zeros((n_tasks, S, Aa, len(static.widths)), np.float32)
    hist["erank_per_layer"] = np.zeros((n_tasks, S, Aa, len(static.widths)), np.float32)
    mgr = T.ckpt_mgr(os.path.join(out, "ckpt"))
    r = T.ckpt_restore(mgr, {"states": states, "task": np.asarray(0, np.int32), "hist": hist})
    t_done = int(r["task"])
    hist = {k: np.asarray(v)[:t_done] for k, v in r["hist"].items()}
    _, _, sums = D.load_mnist()
    T.write_mnist_metrics(dict(cfg, stopped_at_task=t_done), out, seeds, hps, hist, None, sums)
    print(f"[INFO] wrote metrics for {t_done} of {n_tasks} tasks -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
