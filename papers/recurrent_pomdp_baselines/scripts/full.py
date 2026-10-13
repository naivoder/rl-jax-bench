"""Full stage (pre-declared in the README before launch).

  P1  Pendulum-V sac-lstm-200-oar at the paper's full horizon (51k env steps = 5 random + 250 x [1 rollout, 200
      updates], Fig. 21 / authors' run config): the small stage runs (seeds 0-3, TF32) are CONTINUED from their 20k
      orbax checkpoint (iteration 95) to iteration 250.  separate (R=4) and shared {official, detach} (R=8; the
      small stage 'noleak' replications are dropped to save budget; its 20k result stays the reported number).
      The replay buffer is re-allocated for 255 episodes (zero-padded beyond the 100 stored ones); the stored data,
      counters, parameters, optimiser and key tree are restored unchanged.  The only difference from a fresh 51k run
      is the replay-sampling stream (jax.random.categorical draws over a buffer of 100 vs 255 slots), i.e. a
      different but identically distributed sample path - stated in the README.
  L5  Pendulum-V sac-lstm-5-oar (context length 5; authors' logs: separate -1089.9, shared -820.4), separate and
      shared official, seeds 0-3, 51k env steps from scratch, TF32 (paired with each other).

Usage: python scripts/full.py P1 | L5
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import info, load_config, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.train import Runner, VARIANTS, load_ckpt, save_ckpt, train  # noqa: E402

OUT = os.path.join(common.REPO, "runs", "full")
G2 = os.path.join(common.REPO, "runs", "small")
SEEDS = [0, 1, 2, 3]
PREC = "matmul_precision=default"   # TF32: the small stage runs being continued used it; ONE precision for both arms
BUF_KEYS = ("obs", "act", "rew", "term", "length")


def reps_of(seeds, variants):
    return [(s, v) for v in variants for s in seeds]


def continue_from_small(algo, arch, g2_name, g2_variants, keep_variants, name):
    """Copy the small stage iteration-95 checkpoint into a 250-iteration run directory (buffer padded, reps selected)."""
    out = os.path.join(OUT, name)
    ckpt = os.path.join(os.path.abspath(out), "ckpt")
    over = [f"algo={algo}", f"arch={arch}", PREC, "ckpt_every_iters=25"]
    cfg_new = load_config("pendulum_v.yaml", over)
    if os.path.exists(os.path.join(ckpt, "host.json")):
        info("continuation checkpoint already present", out=out)
        return cfg_new, out
    cfg_old = load_config("pendulum_v.yaml", over[:-1] + ["num_iters=95"])
    reps_old = reps_of(SEEDS, g2_variants)
    reps_new = reps_of(SEEDS, keep_variants)
    sel = np.array([reps_old.index(r) for r in reps_new])
    R_old, R_new = Runner(cfg_old), Runner(cfg_new)
    seed_old = jnp.array([s for s, _ in reps_old])
    coef_old = jnp.array([VARIANTS.get(v, (1.0, 1.0)) for _, v in reps_old], jnp.float32)
    ts_like = jax.jit(jax.vmap(R_old.init_state))(seed_old, coef_old)
    ts, host = load_ckpt(os.path.join(G2, g2_name, "ckpt"), ts_like)
    assert ts is not None and host["it"] == 95 and host["n_env"] == 20000, host.get("it")
    ts = jax.tree_util.tree_map(lambda x: x[sel], ts)
    n_old, n_new = R_old.n_ep_total, R_new.n_ep_total
    assert int(np.asarray(ts["buf"]["count"]).max()) == n_old == 100 and n_new == 255
    buf = dict(ts["buf"])
    for k in BUF_KEYS:
        x = buf[k]
        pad = [(0, 0), (0, n_new - n_old)] + [(0, 0)] * (x.ndim - 2)
        buf[k] = jnp.pad(x, pad)
    ts = {**ts, "buf": buf}
    # host log: keep only the selected replications in every per-replication list
    nR = len(reps_old)

    def pick(v):
        return [v[i] for i in sel] if isinstance(v, list) and len(v) == nR else v
    host = {**host,
            "evals": [{**e, "return": pick(e["return"])} for e in host["evals"]],
            "train": [{k: pick(v) for k, v in t.items()} for t in host["train"]],
            "continued_from": {"run": os.path.join("runs", "small", g2_name), "it": 95, "n_env": 20000,
                               "kept_reps": [list(r) for r in reps_new], "buffer_slots": [n_old, n_new]}}
    # sanity: the restored state matches a fresh 250-iteration state in every non-buffer leaf shape
    seed_new = jnp.array([s for s, _ in reps_new])
    coef_new = jnp.array([VARIANTS.get(v, (1.0, 1.0)) for _, v in reps_new], jnp.float32)
    fresh = jax.eval_shape(jax.vmap(R_new.init_state), seed_new, coef_new)
    jax.tree_util.tree_map(lambda a, b: None if a.shape == b.shape else (_ for _ in ()).throw(
        AssertionError(f"shape {a.shape} vs {b.shape}")), ts, fresh)
    np.testing.assert_array_equal(np.asarray(ts["coef"]["leak"]), np.asarray(coef_new[:, 0]))
    save_ckpt(ckpt, ts, host)
    write_json(os.path.join(out, "continuation.json"), host["continued_from"])
    info("continuation checkpoint written", out=out, reps=len(reps_new))
    return cfg_new, out


def main(group):
    if group == "P1":
        jobs = [("sac", "separate", "P_sac_separate", ["separate"], ["separate"], "P1_sac_separate"),
                ("sac", "shared", "P_sac_shared", ["official", "noleak", "detach"], ["official", "detach"],
                 "P1_sac_shared")]
        for algo, arch, g2, g2v, keep, name in jobs:
            cfg, out = continue_from_small(algo, arch, g2, g2v, keep, name)
            common.set_precision(cfg["matmul_precision"])
            info("full run", name=name)
            train(cfg, SEEDS, keep, out, resume=True)
    elif group == "L5":
        for arch, variants in (("separate", ["separate"]), ("shared", ["official"])):
            cfg = load_config("pendulum_v.yaml", ["algo=sac", f"arch={arch}", "sampled_seq_len=5", PREC,
                                                  "ckpt_every_iters=50"])
            common.set_precision(cfg["matmul_precision"])
            name = f"L5_sac_{arch}"
            info("full run", name=name)
            train(cfg, SEEDS, variants, os.path.join(OUT, name), resume=True)
    else:
        raise SystemExit(f"unknown group {group}")


if __name__ == "__main__":
    main(sys.argv[1])
