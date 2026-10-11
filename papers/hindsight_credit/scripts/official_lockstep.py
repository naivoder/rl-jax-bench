"""End-to-end lockstep differential test against the official snippet.

    uv run python -m scripts.official_lockstep [--n_runs 20] [--seed 2] [--fixtures]

For every Monte-Carlo learning experiment (E1, E3 run as MC, E4, E5, E6, E7, E8) and both HCA arms we take the
small-scale run keys, materialise the exact per-step uniforms/normals our sampler consumes (data.sample_episode key
use), and feed them to scripts/official/ref_driver.py, which runs the UNMODIFIED
official StateHCA/ReturnHCA updates inside a separately written numpy env/sampler/caller (its own venv). Same
inputs -> the two learners must produce the same per-episode exact-value curves and final tables.

Not covered (not in the official code): the PG baseline, State-HCA bootstrapping (E3's 3-step return), the
paper-pseudocode r-hat variant. ``--fixtures`` also writes tests/fixtures/official_*.npz for test_official_parity.
"""
from __future__ import annotations

import argparse
import json
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
from src.train import load_config, make_env, make_runner, method_lrs, run_keys, settings
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# authors' clone (repo/) and its numpy venv (venv/); see README "Checked against"
OFFICIAL = os.environ.get("OFFICIAL_ROOT", os.path.join(ROOT, "external", "official"))
OFF_PY = os.path.join(OFFICIAL, "venv", "bin", "python")
FIX = os.path.join(ROOT, "tests", "fixtures")
EXPS = ("E1", "E3", "E4", "E5", "E6", "E7", "E8")
TOL = 1e-9


def episode_draws(keys, episodes: int, L: int):
    """The exact (u_env, z_noise, u_act) of data.sample_episode for ep_key = fold_in(run_key, e)."""
    def one(key, e):
        k = jax.random.fold_in(key, e)
        f = lambda i: jax.random.fold_in(k, i)  # noqa: E731
        return (jax.random.uniform(f(0), (L,), dtype=jnp.float64), jax.random.normal(f(1), (L,), dtype=jnp.float64),
                jax.random.uniform(f(2), (L,), dtype=jnp.float64))
    g = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))
    return [np.asarray(x) for x in g(keys, jnp.arange(episodes))]


def run_official_lockstep(env_name, env_kwargs, method, lrs, centres, draws):
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        np.savez(fi, env=env_name, env_kwargs=json.dumps(env_kwargs), method=method, lrs=np.asarray(lrs, float),
                 gamma=1.0, centres=np.asarray(centres), u_env=draws[0], z=draws[1], u_act=draws[2])
        subprocess.run([OFF_PY, "-I", os.path.join(ROOT, "scripts", "official", "ref_driver.py"), "lockstep", fi, fo], check=True)
        return dict(np.load(fo))


def compare(env, ours, off):
    """Max abs differences on the rows the learner actually uses (decision observations; the snippet also treats
    the terminal reward-only step as a source, A-TERMINAL, so terminal-observation rows are excluded)."""
    dec = np.unique(env.obs[~env.terminal])
    return {"curve": float(np.abs(np.asarray(ours["curve"]) - off["curve"]).max()),
            "returns": float(np.abs(np.asarray(ours["returns"]) - off["returns"]).max()),
            "theta": float(np.abs(np.asarray(ours["params"].theta)[:, dec] - off["theta"][:, dec]).max()),
            "V": float(np.abs(np.asarray(ours["params"].V)[:, dec] - off["V"][:, dec]).max())}


def lockstep_case(cfg, exp, si, kw, method, n_runs, seed, episodes=None):
    ec = cfg["experiments"][exp]
    env = make_env(ec, kw)
    E = episodes or ec["episodes"]
    keys = run_keys(seed, exp, si, n_runs)
    lrs = method_lrs(cfg, method)
    centres = D.bin_centres(env, ec["n_bins"], cfg["hca"]["bin_widen"])
    runner = make_runner(env, method, episodes=E, seg_len=None, n_bins=ec["n_bins"], bin_widen=cfg["hca"]["bin_widen"],
                         gamma=cfg["gamma"])
    ours = jax.block_until_ready(runner(keys, lrs))
    draws = episode_draws(keys, E, env.L)
    off = run_official_lockstep(ec["env"], kw, method, [float(lrs[0]), float(lrs[1]), float(lrs[2])], centres, draws)
    return env, keys, draws, ours, off


def gradient_fixture():
    """Per-episode gradient fixture: random tables, 8 episodes per (env, gamma), official outputs recorded."""
    from scripts.diff_official import ENVS
    rec = {}
    names = ("shortcut", "delayed_effect_n3_s2", "bandit_hidden", "bandit_observed")
    for si, name in enumerate(names):
        env = ENVS[name]()
        for gamma in (1.0, 0.9):
            k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(500 + si), 5)
            centres = np.asarray(D.bin_centres(env, 10))
            p = Params(jax.random.normal(k1, (env.n_obs, 2)), jax.random.normal(k2, (env.n_obs, env.n_obs, 2)),
                       jax.random.normal(k3, (env.n_obs, 10, 2)), jax.random.normal(k4, (env.n_obs,)),
                       jnp.zeros((env.n_obs, 2)))
            pi = policy(p.theta)
            eps = jax.vmap(lambda kk: D.sample_episode(D.env_arrays(env), env.s0, env.L, lambda o: pi[o], kk))(
                jax.random.split(k5, 8))
            inp = {"pi": np.asarray(pi), "V": np.asarray(p.V), "gamma": gamma, "n_episodes": 8,
                   "h_state": np.transpose(np.asarray(hindsight(p.psi)), (2, 0, 1)),
                   "h_return": np.transpose(np.asarray(hindsight(p.phi)), (2, 0, 1)), "return_bins": centres}
            for i in range(8):
                T = int(np.asarray(eps.valid[i]).sum())
                for f, src_ in (("states", eps.obs), ("actions", eps.action), ("rewards", eps.reward)):
                    inp[f"ep{i}_{f}"] = np.asarray(src_[i, :T])
            with tempfile.TemporaryDirectory() as td:
                fi, fo = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
                np.savez(fi, **inp)
                subprocess.run([OFF_PY, "-I", os.path.join(ROOT, "scripts", "official", "run_official.py"), fi, fo], check=True,
                               capture_output=True)
                off = dict(np.load(fo))
            tag = f"{name}_g{gamma}"
            rec[f"{tag}/theta"], rec[f"{tag}/psi"], rec[f"{tag}/phi"], rec[f"{tag}/V"] = (np.asarray(x) for x in p[:4])
            rec[f"{tag}/centres"] = centres
            for f in eps._fields:
                rec[f"{tag}/ep_{f}"] = np.asarray(getattr(eps, f))
            for k, v in off.items():
                rec[f"{tag}/off_{k}"] = v
            # sanity: ours == official on this fixture before it is written
            gs = jax.vmap(lambda e: state_hca_grads(p, e, None, gamma, "official"))(eps)
            dec = np.unique(env.obs[~env.terminal])
            assert np.abs(np.asarray(gs.theta[0])[dec] - off["ep0_state_dlogits"][dec]).max() < 1e-12
            gr = jax.vmap(lambda e: return_hca_grads(p, e, gamma, jnp.asarray(centres)))(eps)
            assert np.abs(np.asarray(gr.theta[0])[dec] - off["ep0_return_dlogits"][dec]).max() < 1e-12
    np.savez_compressed(os.path.join(FIX, "official_grads.npz"), **rec)
    info("wrote fixture", path=os.path.join(FIX, "official_grads.npz"), cases=len(names) * 2)


LOCK_FIX = (("E1", 0, "state_hca", 40), ("E1", 0, "return_hca", 40), ("E4", 0, "state_hca", 60),
            ("E4", 0, "return_hca", 60), ("E7", 0, "return_hca", 40), ("E8", 0, "state_hca", 40))


def lockstep_fixture(cfg):
    rec = {}
    for exp, si, method, E in LOCK_FIX:
        kw = settings(cfg["experiments"][exp])[si]
        env, keys, draws, ours, off = lockstep_case(cfg, exp, si, kw, method, 3, 7, episodes=E)
        tag = f"{exp}_s{si}_{method}"
        rec[f"{tag}/keys"] = np.asarray(jax.random.key_data(keys)) if keys.dtype != jnp.uint32 else np.asarray(keys)
        rec[f"{tag}/episodes"] = np.asarray(E)
        rec[f"{tag}/u_env"], rec[f"{tag}/z"], rec[f"{tag}/u_act"] = draws
        for k in ("curve", "returns", "theta", "V"):
            rec[f"{tag}/off_{k}"] = off[k]
        err = compare(env, ours, off)
        assert max(err.values()) < TOL, (tag, err)
    np.savez_compressed(os.path.join(FIX, "official_lockstep.npz"), **rec)
    info("wrote fixture", path=os.path.join(FIX, "official_lockstep.npz"), cases=len(LOCK_FIX))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "official", "lockstep.json"))
    args = ap.parse_args()
    cfg = load_config()
    check_device()
    commit = subprocess.check_output(["git", "-C", os.path.join(OFFICIAL, "repo"), "rev-parse", "HEAD"], text=True).strip()
    res = {"official_commit": commit, "n_runs": args.n_runs, "seed": args.seed, "tol": TOL, "cases": {}}
    t = Timer()
    worst = 0.0
    for exp in EXPS:
        for si, kw in enumerate(settings(cfg["experiments"][exp])):
            for m in ("state_hca", "return_hca"):
                env, _, _, ours, off = lockstep_case(cfg, exp, si, kw, m, args.n_runs, args.seed)
                err = compare(env, ours, off)
                curve = np.asarray(ours["curve"])
                res["cases"][f"{exp}_s{si}_{m}"] = {"env_kwargs": kw, "max_abs_err": err,
                                                     "ours_final_mean": float(curve[:, -1].mean()),
                                                     "official_final_mean": float(off["curve"][:, -1].mean()),
                                                     "ours_avg_mean": float(curve.mean()),
                                                     "official_avg_mean": float(off["curve"].mean())}
                worst = max(worst, *err.values())
                metric("lockstep", exp=exp, setting=si, method=m, **{k: f"{v:.2e}" for k, v in err.items()},
                       ours_final=round(float(curve[:, -1].mean()), 4), off_final=round(float(off["curve"][:, -1].mean()), 4))
    res["worst_max_abs_err"], res["pass"], res["wall_s"] = worst, worst < TOL, t.lap()
    write_json(args.out, res)
    info("lockstep done", worst=worst, passed=res["pass"], out=args.out)
    if args.fixtures:
        os.makedirs(FIX, exist_ok=True)
        gradient_fixture()
        lockstep_fixture(cfg)


if __name__ == "__main__":
    main()
