"""E9 driver (paper §3.4 / Figure 7): Buffer-Q with/without PEB for every buffer size; 30 seeds x 2 PEB per process.

Writes runs/<tag>/e9_difficult_grid/B<size>/<peb|no_peb>/seed_<n>/metrics.json (binned online eps-greedy return).
Usage: uv run python scripts/small/run_e9.py --tag small [--set total_steps=200000]
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.train import check_device, git_sha_at_start, info, load_config, metric, train_buffer_q, write_json, cfg_hash  # noqa: E402

import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(REPO, "src", "configs", "e9_difficult_grid.yaml"))
    p.add_argument("--tag", default="dev")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    check_device()
    cfg = load_config(a.config, a.set)
    out = os.path.join(REPO, "runs", a.tag, cfg["name"])
    import hashlib
    from src import data as D
    tab = D.difficult_grid()
    env_ck = dict(transition_table_sha256=hashlib.sha256(np.ascontiguousarray(np.asarray(tab.NX, np.int32)).tobytes()).hexdigest())
    for B in cfg["buffer_sizes"]:
        names, res, timing = train_buffer_q(cfg, int(B))
        r = np.asarray(res.bin_ret_sum) / np.maximum(np.asarray(res.bin_cnt), 1)
        for i, (peb, s) in enumerate(names):
            arm = "peb" if peb else "no_peb"
            write_json(os.path.join(out, f"B{B}", arm, f"seed_{s}", "metrics.json"),
                       dict(exp=cfg["name"], B=B, arm=arm, seed=s, git=git_sha_at_start(), config_hash=cfg_hash(cfg), config=cfg,
                            env_checksum=env_ck, bin_steps=cfg["bin_steps"], bin_return_mean=r[i], bin_return_sum=res.bin_ret_sum[i],
                            bin_episodes=res.bin_cnt[i], bin_len_sum=res.bin_len_sum[i], timing=timing))
        nb = r.shape[1]
        last = max(1, nb // 10)
        for peb in cfg["peb"]:
            ii = [i for i, (pp, _) in enumerate(names) if pp == peb]
            fin = r[ii, -last:].mean(1)
            metric("E9", B=B, peb=peb, final_mean=float(fin.mean()), final_se=float(fin.std(ddof=1) / np.sqrt(len(ii))),
                   curve=",".join(f"{v:.1f}" for v in r[ii].mean(0)))
    info("E9 done", out=out)


if __name__ == "__main__":
    main()
