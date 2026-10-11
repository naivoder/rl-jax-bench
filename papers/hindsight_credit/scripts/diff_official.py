"""Differential test of our State/Return HCA update directions against the official snippet
(github.com/hca-neurips2019/hca @ 3a564ad), run in its own isolated venv.

    uv run python -m scripts.diff_official [--n 100] [--out runs/exact_checks/diff_official.json]

For every env, gamma in {1.0, 0.9} and random tables, we sample ``n`` episodes with our JAX sampler, compute
our per-episode (dtheta, dV, dpsi / dphi) and the official (dlogits, dV, dlogits_h) on the identical inputs, and
report the max abs difference on every row whose source observation is a decision observation. (The official
loop also treats the terminal reward-only step as a source; those rows are irrelevant and excluded, A-TERMINAL.)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import src  # noqa: F401  (bootstrap + x64)
import jax
import jax.numpy as jnp
import numpy as np

from src import data as D
from src.algorithm import return_hca_grads, state_hca_grads
from src.model import Params, hindsight, policy
from rl_jax_bench.jaxlog import info, metric, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# authors' clone (repo/) and its numpy venv (venv/); see README "Checked against"
OFFICIAL = os.environ.get("OFFICIAL_ROOT", os.path.join(ROOT, "external", "official"))
ENVS = {
    "shortcut": lambda: D.shortcut(),
    "delayed_effect_n3_s2": lambda: D.delayed_effect(3, 2.0),
    "delayed_effect_n5_s0": lambda: D.delayed_effect(5, 0.0),
    "delayed_effect_n3_aliased": lambda: D.delayed_effect(3, 1.0, "shared"),
    "bandit_observed": lambda: D.ambiguous_bandit(0.1, 1.5, False),
    "bandit_hidden": lambda: D.ambiguous_bandit(0.2, 0.5, True),
    "thm7_counterexample": lambda: D.thm7_counterexample(),
}


def one_case(name, env, gamma, n, seed):
    k = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(k, 5)
    n_bins = 10
    centres = jnp.asarray(D.bin_centres(env, n_bins))
    A = D.N_ACTIONS
    p = Params(jax.random.normal(k1, (env.n_obs, A)), jax.random.normal(k2, (env.n_obs, env.n_obs, A)),
               jax.random.normal(k3, (env.n_obs, n_bins, A)), jax.random.normal(k4, (env.n_obs,)),
               jnp.zeros((env.n_obs, A)))
    ea = D.env_arrays(env)
    pi = policy(p.theta)
    keys = jax.random.split(k5, n)
    eps = jax.vmap(lambda kk: D.sample_episode(ea, env.s0, env.L, lambda o: pi[o], kk))(keys)
    gs = jax.vmap(lambda e: state_hca_grads(p, e, None, gamma, "official"))(eps)
    gr = jax.vmap(lambda e: return_hca_grads(p, e, gamma, centres))(eps)
    inp = {"pi": np.asarray(pi), "V": np.asarray(p.V), "gamma": gamma, "n_episodes": n,
           "h_state": np.transpose(np.asarray(hindsight(p.psi)), (2, 0, 1)),
           "h_return": np.transpose(np.asarray(hindsight(p.phi)), (2, 0, 1)),
           "return_bins": np.asarray(centres)}
    for i in range(n):
        T = int(np.asarray(eps.valid[i]).sum())
        inp[f"ep{i}_states"] = np.asarray(eps.obs[i, :T])
        inp[f"ep{i}_actions"] = np.asarray(eps.action[i, :T])
        inp[f"ep{i}_rewards"] = np.asarray(eps.reward[i, :T])
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        np.savez(fi, **inp)
        subprocess.run([os.path.join(OFFICIAL, "venv", "bin", "python"), "-I",
                        os.path.join(ROOT, "scripts", "official", "run_official.py"), fi, fo], check=True, capture_output=True)
        off = dict(np.load(fo))
    dec_obs = np.unique(env.obs[~env.terminal])
    err = {"state_dlogits": 0.0, "state_dV": 0.0, "state_dh": 0.0,
           "return_dlogits": 0.0, "return_dV": 0.0, "return_dh": 0.0}
    for i in range(n):
        for tag, g, hkey in (("state", gs, "psi"), ("return", gr, "phi")):
            dl = np.asarray(g.theta[i])[dec_obs]
            dv = np.asarray(g.V[i])[dec_obs]
            dh = np.transpose(np.asarray(getattr(g, hkey)[i]), (2, 0, 1))[:, dec_obs]
            err[f"{tag}_dlogits"] = max(err[f"{tag}_dlogits"], float(np.abs(dl - off[f"ep{i}_{tag}_dlogits"][dec_obs]).max()))
            err[f"{tag}_dV"] = max(err[f"{tag}_dV"], float(np.abs(dv - off[f"ep{i}_{tag}_dV"][dec_obs]).max()))
            err[f"{tag}_dh"] = max(err[f"{tag}_dh"], float(np.abs(dh - off[f"ep{i}_{tag}_dh"][:, dec_obs]).max()))
    scale = max(float(np.abs(np.asarray(gs.theta)).max()), float(np.abs(np.asarray(gr.theta)).max()))
    return err, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "exact_checks",
                                                    "diff_official.json"))
    args = ap.parse_args()
    commit = subprocess.check_output(["git", "-C", os.path.join(OFFICIAL, "repo"), "rev-parse", "HEAD"], text=True).strip()
    res, worst = {"official_commit": commit, "n_episodes": args.n, "cases": {}}, 0.0
    for si, (name, mk) in enumerate(ENVS.items()):
        for gamma in (1.0, 0.9):
            err, scale = one_case(name, mk(), gamma, args.n, 100 + si)
            res["cases"][f"{name}_g{gamma}"] = {"max_abs_err": err, "grad_scale": scale}
            worst = max(worst, *err.values())
            metric("diff_official", case=name, gamma=gamma, max_err=max(err.values()), grad_scale=scale)
    res["worst_max_abs_err"] = worst
    res["pass"] = worst <= 1e-10
    write_json(args.out, res)
    info("diff_official done", worst=worst, passed=res["pass"], out=args.out)


if __name__ == "__main__":
    main()
