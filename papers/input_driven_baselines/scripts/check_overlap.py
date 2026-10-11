"""/§3: hash EVERY training input sequence a full-scale run can draw and intersect with the
test (and validation) sets; also check int32 time headroom (no overflow) under the worst policy
(every job on the slowest server).

    python scripts/check_overlap.py --seeds 0-9,100,101 --iters 2500
"""
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import key_tree, lb_input_kwargs, sample_lb_batch, seq_hashes, test_key, val_key  # noqa: E402
from src.envs import lb_durations, make_lb_spec  # noqa: E402
from src.train import load_config  # noqa: E402


def seeds_of(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/configs/load_balance.yaml")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--out", default="runs/full/overlap_check.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    kw = lb_input_kwargs(cfg)
    test = sample_lb_batch(test_key(), 100, **kw)
    val = sample_lb_batch(val_key(), 100, **kw)
    th, vh = set(seq_hashes(test)), set(seq_hashes(val))
    res = {"test_val_overlap": len(th & vh), "per_seed": {}}
    spec = make_lb_spec(cfg)
    max_t = 0
    for sd in seeds_of(a.seeds):
        keys = key_tree(sd)
        n_hit, n_seq = 0, 0
        for G in (16, 1):  # state/none draw 16 groups per iteration, maml 1 (different splits)
            gen = jax.jit(jax.vmap(lambda it: sample_lb_batch(jax.random.fold_in(keys.inputs_train, it), G, **kw)))
            for c0 in range(0, a.iters, 500):
                b = gen(jnp.arange(c0, min(c0 + 500, a.iters)))
                arr = np.asarray(b.arrivals).reshape(-1, spec.n_jobs)
                siz = np.asarray(b.sizes).reshape(-1, spec.n_jobs)
                from src.data import LBInputs
                hs = set(seq_hashes(LBInputs(arr, siz)))
                n_hit += len(hs & (th | vh))
                n_seq += arr.shape[0]
                # worst-case finish time: every job FIFO on the slowest server
                dur = (siz.astype(np.int64) * spec.rate_den) // min(spec.rate_num)
                fin = np.zeros(arr.shape[0], np.int64)
                for j in range(spec.n_jobs):
                    fin = np.maximum(fin, arr[:, j]) + dur[:, j]
                max_t = max(max_t, int(fin.max()), int((siz.astype(np.int64) * spec.rate_den).max()))
        fixed = sample_lb_batch(keys.inputs_fixedN, int(cfg["n_value_nets"]), **kw)
        n_hit += len(set(seq_hashes(fixed)) & (th | vh))
        res["per_seed"][sd] = {"train_sequences_hashed": n_seq + 10, "overlap_with_test_or_val": n_hit}
        print(f"[METRIC] overlap seed={sd} sequences={n_seq + 10} overlap={n_hit}", flush=True)
    res["max_int_time_or_size_x100_worst_policy"] = max_t
    res["int32_max"] = 2 ** 31 - 1
    res["int32_headroom_ok"] = bool(max_t < 2 ** 31 - 1)
    print(f"[METRIC] int32 worst-case max={max_t} ok={res['int32_headroom_ok']} test_val_overlap={res['test_val_overlap']}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
