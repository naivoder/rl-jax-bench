"""Sensitivity of OUR OWN trajectory to a 1-ulp-scale perturbation of the initial weights (SCR seed 0, 200k steps).

Shows whether a trajectory mismatch vs the oracle is explained by chaotic amplification of float rounding
(Adam rescales near-zero gradients to O(lr) steps) rather than by an implementation difference.
    JAX_PLATFORMS=cpu uv run python -m scripts.oracle.chaos
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from scripts.oracle.diff import OUT, our_traj
from src import algorithm as A
from src import data as D
from src.model import ACT_ID, INIT_GAIN, init_mlp


def main():
    spec = D.scr_spec("flip_one")
    root = jax.random.PRNGKey(0)
    X, Y, _ = D.scr_stream(jax.random.fold_in(root, 0), jax.random.fold_in(root, 1), spec, 20)
    res = {}
    for name, act, opt in (("tanh_adam_0.01", "tanh", "adam"), ("tanh_sgd_0.01", "tanh", "sgd"),
                           ("relu_sgd_0.01", "relu", "sgd")):
        st = A.Static(n_in=20, widths=(5,), n_out=1, opt=opt)
        p = init_mlp(jax.random.PRNGKey(7), 20, (5,), 1, INIT_GAIN[act])
        hp = A.make_hp(lr=0.01, act=ACT_ID[act])
        e0, _ = our_traj(st, p, hp, X, Y)
        p1 = jax.tree.map(lambda a: a * (1 + 1e-7), p)
        e1, _ = our_traj(st, p1, hp, X, Y)
        d = np.abs(e0 - e1)
        div = np.where(d > 1e-3)[0]
        res[name] = {"first_step_abs_diff_gt_1e-3": int(div[0]) if len(div) else None,
                     "mse_unperturbed": float(e0.mean()), "mse_perturbed": float(e1.mean()),
                     "max_abs_step_err_diff": float(d.max())}
        print("[METRIC] chaos", name, res[name], flush=True)
    with open(os.path.join(OUT, "chaos.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
