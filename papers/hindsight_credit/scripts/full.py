"""full-scale run: full reproduction at the paper's scale (100 runs per curve), plus the PG lr tuning, the ablations,
the paper's theoretical claims tested empirically, and the divergence probes.

    uv run python -m scripts.full tune                 # PG lr per panel/sweep point, tuning seed 999
    uv run python -m scripts.full eval --seed 0        # every experiment, all arms, 100 runs -> runs/seed_0/
    uv run python -m scripts.full eval --seed 1        # replication seeds (descriptive)
    uv run python -m scripts.full ablation             # -> runs/full/ablation.json
    uv run python -m scripts.full theory               # -> runs/full/theory.json
    uv run python -m scripts.full divergence           # -> runs/full/divergence.json
    uv run python -m scripts.full cost                 # FLOPs + wall clock per arm -> runs/full/cost.json
    uv run python scripts/report.py               # tables, paired tests (no jax)

Metrics files are compact: for every (experiment, setting, arm) the per-run final value, curve mean (A-AVG) and the
values at a few episodes, plus the mean and std curves. Full per-run curves go to curves.npz next to metrics.json.
"""
from __future__ import annotations

import argparse
import os
import sys

import src  # noqa: F401  (GPU bootstrap + x64)
import jax
import jax.numpy as jnp
import numpy as np

from src import data as D
from src.algorithm import mc_returns, pg_grads, return_hca_grads, state_hca_grads
from src.eval import (dp_q, enumerate_trajectories, exact_pair_hindsight, exact_return_hindsight,
                      trajectories_to_batch, true_policy_gradient)
from src.model import init_params, policy
from src.train import (EXP_ID, METHODS, load_config, make_env, make_fig3r_runner, make_runner, method_lrs, provenance,
                       run_keys, settings)
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "full")
LEARN_EXPS = ("E1", "E3", "E4", "E5", "E6", "E7", "E8")
PROBE_EPS = {"E1": (20, 50, 100), "E3": (100, 500), "E4": (50, 100, 500), "E5": (500,), "E6": (100,), "E7": (100,),
             "E8": (100,)}
_RUNNERS: dict = {}
TUNING_SEED = 999  # = seed 0 + pg_tuning_seed_offset (A-PG-LR)


def summarise_curve(curve: np.ndarray, exp: str) -> dict:
    c = np.asarray(curve)
    out = {"final": c[:, -1], "avg": c.mean(1), "mean_curve": c.mean(0), "std_curve": c.std(0, ddof=1)}
    for e in PROBE_EPS[exp]:
        out[f"ep{e}"] = c[:, e]
    return out


def run_arm(cfg, exp, kw, method, seed, n_runs, lrs, setting_idx, **rk):
    """One (experiment, setting, method) on ``n_runs`` runs; extra runner kwargs override the config."""
    ec = cfg["experiments"][exp]
    env = make_env(ec, kw)
    opts = dict(episodes=ec["episodes"], seg_len=ec["seg_len"], n_bins=ec["n_bins"], bin_widen=cfg["hca"]["bin_widen"],
                gamma=cfg["gamma"], variant=cfg["hca"]["state_variant"])
    opts.update(rk)
    ck = (D.env_sha256(env), method, tuple(sorted(opts.items())))
    if ck not in _RUNNERS:
        _RUNNERS[ck] = make_runner(env, method, **opts)
    runner = _RUNNERS[ck]
    keys = run_keys(seed, exp, setting_idx, n_runs)
    return np.asarray(jax.block_until_ready(runner(keys, jnp.asarray(lrs)))["curve"])


# ---------------------------------------------------------------- PG lr tuning (tuning seeds only)
def cmd_tune(cfg, n_runs):
    out = {"seed": TUNING_SEED, "n_runs": n_runs, "criterion": "max mean training-curve value (A-PG-LR)",
           "grid": cfg["pg"]["lr_grid"], "experiments": {}}
    t = Timer()
    for exp in LEARN_EXPS:
        rows = []
        for si, kw in enumerate(settings(cfg["experiments"][exp])):
            scores = {}
            for lr in cfg["pg"]["lr_grid"]:
                c = run_arm(cfg, exp, kw, "pg", TUNING_SEED, n_runs, method_lrs(cfg, "pg", lr), si)
                scores[lr] = float(c.mean())
            best = max(scores, key=scores.get)
            rows.append({"setting": kw, "scores": scores, "best_lr": best})
            metric("tune", exp=exp, setting=si, best_lr=best, **{f"lr{k}": round(v, 4) for k, v in scores.items()})
        out["experiments"][exp] = rows
    out["wall_s"] = t.lap()
    write_json(os.path.join(OUT, "pg_tuning.json"), out)


def tuned_lrs():
    import json
    d = json.load(open(os.path.join(OUT, "pg_tuning.json")))
    return {exp: [r["best_lr"] for r in rows] for exp, rows in d["experiments"].items()}


# ---------------------------------------------------------------- full evaluation
def cmd_eval(cfg, seed, n_runs):
    tl = tuned_lrs()
    res = {"provenance": provenance(cfg), "seed": seed, "n_runs": n_runs, "pg_tuned_lr": tl, "experiments": {},
           "timing": {}}
    npz = {}
    t0 = Timer()
    for exp in LEARN_EXPS:
        ec = cfg["experiments"][exp]
        recs = []
        for si, kw in enumerate(settings(ec)):
            env = make_env(ec, kw)
            rec = {"env_kwargs": kw, "env_sha256": D.env_sha256(env), "arms": {}}
            arms = [(m, method_lrs(cfg, m)) for m in ("state_hca", "return_hca")]
            arms += [(f"pg_lr{lr}", method_lrs(cfg, "pg", lr)) for lr in cfg["pg"]["lr_grid"]]
            for name, lrs in arms:
                m = "pg" if name.startswith("pg") else name
                tm = Timer()
                c = run_arm(cfg, exp, kw, m, seed, n_runs, lrs, si)
                rec["arms"][name] = summarise_curve(c, exp) | {"wall_s": tm.lap()}
                npz[f"{exp}_s{si}_{name}"] = c.astype(np.float32)
            rec["arms"]["pg"] = dict(rec["arms"][f"pg_lr{tl[exp][si]}"]) | {"lr": tl[exp][si]}
            for a in ("state_hca", "return_hca", "pg"):
                metric(f"eval {exp}", setting=si, arm=a, final=round(float(rec['arms'][a]['final'].mean()), 4),
                       avg=round(float(rec['arms'][a]['avg'].mean()), 4))
            recs.append(rec)
        res["experiments"][exp] = recs
    # E2 (Fig. 3 right): primary immediate term = rhat (A-F3R-IMM), sensitivities 'official' and 'exact'
    ec = cfg["experiments"]["E2"]
    env = make_env(ec, ec["env_kwargs"])
    pts = []
    for si, p in enumerate(ec["p_long"]):
        keys = run_keys(seed, "E2", si, n_runs)
        rec = {"p_long": p, "arms": {}}
        for imm in ("rhat", "official", "exact"):
            for m in (METHODS if imm == "rhat" else ("state_hca",)):
                run, oracle = make_fig3r_runner(env, m, p, rollouts=ec["rollouts"], n_bins=ec["n_bins"],
                                                bin_widen=cfg["hca"]["bin_widen"], gamma=cfg["gamma"], immediate=imm,
                                                hca_lrs=method_lrs(cfg, "state_hca"))
                r = jax.block_until_ready(run(keys))
                name = m if imm == "rhat" else f"{m}_imm_{imm}"
                rec["arms"][name] = {"online": np.asarray(r["online"]), "final_h": np.asarray(r["final"])}
                rec["oracle"] = oracle
        metric("eval E2", p_long=round(p, 3), oracle=round(rec["oracle"], 4),
               **{a: round(float(v["online"].mean()), 4) for a, v in rec["arms"].items()})
        pts.append(rec)
    res["experiments"]["E2"] = pts
    res["wall_s"] = t0.lap()
    d = os.path.join(ROOT, "runs", f"seed_{seed}")
    write_json(os.path.join(d, "metrics.json"), res)
    np.savez_compressed(os.path.join(d, "curves.npz"), **npz)
    info("eval done", seed=seed, wall_s=res["wall_s"])


# ---------------------------------------------------------------- ablations and sensitivities
def cmd_ablation(cfg, n_runs, seed=0):
    tl = tuned_lrs()
    out = {"seed": seed, "n_runs": n_runs, "rows": []}

    def add(tag, exp, si, kw, method, lrs, **rk):
        c = run_arm(cfg, exp, kw, method, seed, n_runs, lrs, si, **rk)
        s = summarise_curve(c, exp)
        row = {"tag": tag, "exp": exp, "setting": kw, "method": method, "lrs": list(map(float, lrs)),
               "runner_kwargs": {k: v for k, v in rk.items()}, "final": s["final"], "avg": s["avg"]}
        out["rows"].append(row)
        metric("ablation", tag=tag, exp=exp, method=method, final=round(float(s["final"].mean()), 4),
               avg=round(float(s["avg"].mean()), 4))

    base = lambda exp: settings(cfg["experiments"][exp])  # noqa: E731
    hca = lambda m: np.asarray(method_lrs(cfg, m))  # noqa: E731
    # AB1: remove the hindsight mechanism (alpha_h = 0 -> h stays uniform = pi at init -> zero credit)
    for exp, si in (("E1", 0), ("E3", 0), ("E4", 0), ("E6", 0), ("E8", 0)):
        for m in ("state_hca", "return_hca"):
            lrs = hca(m).copy()
            lrs[1] = 0.0
            add("AB1_no_hindsight_learning", exp, si, base(exp)[si], m, lrs)
    # AB2: Return-HCA conditioning resolution n_b (n_b = 1 removes the return information)
    for exp, si in (("E1", 0), ("E4", 0), ("E6", 0), ("E8", 0)):
        for nb in (1, 2, 3, 5, 10, 20):
            add(f"AB2_nbins_{nb}", exp, si, base(exp)[si], "return_hca", hca("return_hca"), n_bins=nb)
    # AB3: immediate-reward treatment of State HCA (official j=i term vs Algorithm 1 r-hat)
    for exp, si in (("E1", 0), ("E3", 0), ("E4", 0), ("E6", 0), ("E7", 0)):
        add("AB3_state_variant_paper", exp, si, base(exp)[si], "state_hca", hca("state_hca"), variant="paper")
    # S1: bins on the noise-free return range (A-BINS sensitivity)
    for exp, si in (("E4", 0), ("E5", 5), ("E6", 0), ("E7", 0), ("E8", 0)):
        add("S1_bins_noise_free", exp, si, base(exp)[si], "return_hca", hca("return_hca"), bin_widen=0.0)
    # S2: one shared chain observation (A-DE-ALIAS sensitivity)
    for exp, si in (("E3", 0), ("E4", 0), ("E5", 5)):
        kw = dict(base(exp)[si], aliasing="shared")
        for m in ("state_hca", "return_hca", "pg"):
            lrs = method_lrs(cfg, m, tl[exp][si]) if m == "pg" else hca(m)
            add("S2_one_chain_obs", exp, si, kw, m, lrs)
    # S3: crossover eps = 0 for Fig. 5 left/middle (A-AB-EPS sensitivity)
    for exp in ("E6", "E7"):
        kw = dict(base(exp)[0], eps=0.0)
        for m in ("state_hca", "return_hca", "pg"):
            lrs = method_lrs(cfg, m, tl[exp][0]) if m == "pg" else hca(m)
            add("S3_eps0", exp, 0, kw, m, lrs)
    # S4: E3 with Monte-Carlo returns for every arm (removes the 3-step bootstrap the paper's panel is about)
    for m in ("state_hca", "pg"):
        lrs = method_lrs(cfg, m, tl["E3"][0]) if m == "pg" else hca(m)
        add("S4_E3_monte_carlo", "E3", 0, base("E3")[0], m, lrs, seg_len=None)
    write_json(os.path.join(OUT, "ablation.json"), out)


# ---------------------------------------------------------------- theoretical claims, tested empirically
def exact_moments(env, theta, batch_probs, per_traj):
    """Exact mean and variance of a per-trajectory scalar under the enumerated trajectory distribution."""
    m = float(np.sum(batch_probs * per_traj))
    return m, float(np.sum(batch_probs * (per_traj - m) ** 2))


def cmd_theory(cfg):
    """T1 noise-variance claim (Eq. 3 discussion, §5 Delayed Effect): per-episode x0 logit-update variance of
    State HCA (exact hindsight), Return HCA (exact binned h_z) and REINFORCE with the exact V(x0) baseline (MC
    PG) on Delayed Effect n = 3, sigma in 0..5. Noise integrated by Monte Carlo (2e5 episodes per point, the exact
    tables are computed by enumeration on the noise-free env, which is valid because the chain observations carry no
    noise and h(a|x0, chain) = pi exactly). Fit Var(sigma) - Var(0) = c sigma^k: theory k_PG = 2, State HCA flat.
    T2 counterfactual claim (Fig. 3 right): exact expectation of the per-rollout estimators with exact h and the
    PG indicator estimator at every p_long, vs the oracle (State HCA unbiased; PG = 2p(1-p) * oracle)."""
    out = {}
    n_mc = 200_000
    key = jax.random.PRNGKey(31)
    rows = []
    for sigma in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        env = D.delayed_effect(n=3, sigma=sigma)
        env0 = D.delayed_effect(n=3, sigma=0.0)
        ea = D.env_arrays(env)
        rng = np.random.default_rng(7)
        theta = np.zeros((env.n_obs, 2))
        theta[0] = [0.4, -0.4]  # a non-uniform policy at x0 (so pi(a0) != 1/2), uniform elsewhere
        pi = np.asarray(policy(jnp.asarray(theta)))
        h_star, _ = exact_pair_hindsight(env0, pi)
        cent = D.bin_centres(env, 3, 2.0)
        keys = jax.random.split(jax.random.fold_in(key, int(sigma * 10)), n_mc)
        eps_ = jax.vmap(lambda k: D.sample_episode(ea, env.s0, env.L, lambda o: jnp.asarray(pi)[o], k))(keys)
        eps_np = jax.tree_util.tree_map(np.asarray, eps_)
        # exact pair hindsight gathered per episode (obs-based, noise-free)
        hp = np.full((n_mc, env.L, env.L, 2), 0.5)
        for (ox, oy), v in h_star.items():
            mask = (eps_np.obs[:, :, None] == ox) & (eps_np.obs[:, None, :] == oy)
            hp[mask] = v
        params = init_params(env.n_obs, 3)._replace(theta=jnp.asarray(theta))
        st = jax.vmap(lambda e, h: state_hca_grads(params, e, None, 1.0, "official", h_pair_override=h).theta[0, 0])(
            eps_, jnp.asarray(hp))
        # exact h_z by Monte Carlo over the noisy returns (fixed point of Alg. 2 training), per (x0, bin)
        Z = np.asarray(jax.vmap(lambda e: mc_returns(e, 1.0))(eps_))[:, 0]
        b = np.argmin(np.abs(cent[None, :] - Z[:, None]), 1)
        a0 = eps_np.action[:, 0]
        hz = np.full((len(cent), 2), 0.5)
        for k in range(len(cent)):
            sel = b == k
            if sel.sum():
                hz[k] = [(a0[sel] == 0).mean(), (a0[sel] == 1).mean()]
        hz_ep = np.full((n_mc, env.L, 2), 0.5)
        hz_ep[:, 0] = np.clip(hz[b], 1e-12, 1.0)
        rt = jax.vmap(lambda e, h: return_hca_grads(params, e, 1.0, jnp.asarray(cent), hz_override=h).theta[0, 0])(
            eps_, jnp.asarray(hz_ep))
        V0 = float(dp_q(ea, jnp.asarray(theta), env.L)[0] @ pi[0])
        pgv = params._replace(V=params.V.at[0].set(V0))
        pg = jax.vmap(lambda e: pg_grads(pgv, e, None, 1.0).theta[0, 0])(eps_)
        true = float(true_policy_gradient(ea, jnp.asarray(theta), env.s0, env.L)[0, 0])
        row = {"sigma": sigma, "true_grad": true}
        for name, x in (("state_hca", st), ("return_hca", rt), ("pg", pg)):
            x = np.asarray(x)
            row[name] = {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(n_mc)), "var": float(x.var(ddof=1)),
                         "var_se": float(x.var(ddof=1) * np.sqrt(2.0 / (n_mc - 1)))}
        rows.append(row)
        metric("T1", sigma=sigma, true=round(true, 4), **{f"var_{k}": round(row[k]["var"], 4) for k in ("state_hca", "return_hca", "pg")},
               **{f"mean_{k}": round(row[k]["mean"], 4) for k in ("state_hca", "return_hca", "pg")})
    # power-law fit of the noise-induced excess variance
    fits = {}
    for name in ("state_hca", "return_hca", "pg"):
        s = np.array([r["sigma"] for r in rows if r["sigma"] > 0])
        ex = np.array([r[name]["var"] - rows[0][name]["var"] for r in rows if r["sigma"] > 0])
        ok = ex > 0
        fit = np.polyfit(np.log(s[ok]), np.log(ex[ok]), 1) if ok.sum() >= 2 else [np.nan, np.nan]
        fits[name] = {"exponent": float(fit[0]), "excess_var": ex.tolist(), "n_points_positive": int(ok.sum())}
    out["T1_noise_variance"] = {"rows": rows, "fits": fits, "n_mc": n_mc}
    # T2: exact expectations of the Fig. 3-right estimators with exact hindsight (enumeration, no sampling)
    env = D.shortcut()
    ea = D.env_arrays(env)
    t2 = []
    for p in cfg["experiments"]["E2"]["p_long"]:
        theta = np.tile([0.0, np.log(p / (1 - p))], (env.n_obs, 1))
        pi = np.asarray(policy(jnp.asarray(theta)))
        trajs = enumerate_trajectories(env, pi)
        probs, batch = trajectories_to_batch(env, trajs)
        h_star, _ = exact_pair_hindsight(env, pi)
        # state HCA with exact pair hindsight from x0: h(a|x0, X_k) = h_star[(0, o_k)]
        q_state = []
        for k in range(len(trajs)):
            obs = np.asarray(batch.obs[k])
            hrow = np.stack([h_star.get((0, int(o)), pi[0]) for o in obs])
            w = (np.arange(env.L) >= 1) & np.asarray(batch.valid[k])
            q = np.sum((w * batch.reward[k])[:, None] * hrow, 0) / pi[0] + env.r_mean[0]
            q_state.append(q[0] - q[1])
        q_state = np.array(q_state)
        cent = D.bin_centres(env, 10, 2.0)
        hz_star, _ = exact_return_hindsight(env, pi, cent)
        Zs = np.array([float(np.sum(batch.reward[k] * batch.valid[k])) for k in range(len(trajs))])
        hz0 = np.stack([hz_star[(0, int(np.argmin(np.abs(cent - z))))] for z in Zs])
        q_ret = Zs * (hz0[:, 0] / pi[0, 0] - hz0[:, 1] / pi[0, 1])
        a0 = np.asarray(batch.action[:, 0])
        Vx = float(np.sum(probs * Zs))
        q_pg = np.where(a0 == 0, 1.0, -1.0) * (Zs - Vx)
        oracle = np.asarray(dp_q(ea, jnp.asarray(theta), env.L)[0])
        row = {"p_long": p, "oracle": float(oracle[0] - oracle[1])}
        for name, x in (("state_hca", q_state), ("return_hca", q_ret), ("pg", q_pg)):
            m, v = exact_moments(env, theta, probs, x)
            row[name] = {"mean": m, "var_per_rollout": v, "se_1000_rollouts": float(np.sqrt(v / 1000))}
        row["pg_theory_2p1mp"] = 2 * p * (1 - p) * row["oracle"]
        t2.append(row)
        metric("T2", p=round(p, 3), oracle=round(row["oracle"], 4), state=round(row["state_hca"]["mean"], 4),
               ret=round(row["return_hca"]["mean"], 4), pg=round(row["pg"]["mean"], 4), pg_th=round(row["pg_theory_2p1mp"], 4))
    out["T2_counterfactual_exact"] = t2
    write_json(os.path.join(OUT, "theory.json"), out)


# ---------------------------------------------------------------- divergence probes
def cmd_divergence(cfg, n_runs, seed=0):
    """Each probe is a paper-text-justified alternative reading of an unstated detail; recorded, never adopted."""
    tl = tuned_lrs()
    out = {"seed": seed, "n_runs": n_runs, "rows": [], "e2": []}
    hca = lambda m: np.asarray(method_lrs(cfg, m))  # noqa: E731

    def add(tag, exp, si, method, lrs, kw=None, **rk):
        kw = kw or settings(cfg["experiments"][exp])[si]
        tm = Timer()
        c = run_arm(cfg, exp, kw, method, seed, n_runs, lrs, si, **rk)
        s = summarise_curve(c, exp)
        out["rows"].append({"tag": tag, "exp": exp, "setting": kw, "method": method, "runner_kwargs": rk,
                            "final": s["final"], "avg": s["avg"], **{k: v for k, v in s.items() if k.startswith("ep")},
                            "mean_curve": s["mean_curve"], "wall_s": tm.lap()})
        metric("div", tag=tag, exp=exp, setting=si, method=method, final=round(float(s["final"].mean()), 4),
               avg=round(float(s["avg"].mean()), 4))

    # D-A: literal Algorithm 1/2 order (train h, V, r-hat on the episode, then compute the policy credit) 
    for exp in LEARN_EXPS:
        for si, _ in enumerate(settings(cfg["experiments"][exp])):
            for m in ("state_hca", "return_hca"):
                add("DA_alg_order", exp, si, m, hca(m), order="alg")
    # D-B: return-bin range (A-BINS) on the Shortcut: wider generic ranges that merge returns into shared bins
    for lo, hi in ((-5.0, 0.0), (-1.0, 1.0), (-5.0, 5.0), (-10.0, 10.0), (-2.0, 0.0)):
        add(f"DB_shortcut_centres_{lo}_{hi}", "E1", 0, "return_hca", hca("return_hca"),
            centres_override=tuple(np.linspace(lo, hi, 10)))
    # D-C: E2 estimate after the per-rollout hindsight update (Alg. 1/2 order) and the immediate-term variants
    ec = cfg["experiments"]["E2"]
    env = make_env(ec, ec["env_kwargs"])
    for si, p in enumerate(ec["p_long"]):
        keys = run_keys(seed, "E2", si, n_runs)
        rec = {"p_long": p}
        for m in ("state_hca", "return_hca"):
            run, oracle = make_fig3r_runner(env, m, p, rollouts=ec["rollouts"], n_bins=ec["n_bins"],
                                            bin_widen=cfg["hca"]["bin_widen"], gamma=cfg["gamma"], immediate="rhat",
                                            hca_lrs=method_lrs(cfg, "state_hca"), estimate_after_update=True)
            r = jax.block_until_ready(run(keys))
            rec[f"{m}_after_update"] = np.asarray(r["online"])
            rec["oracle"] = oracle
            # fewer rollouts per repetition (the h transient dominates): 100 rollouts
            run, _ = make_fig3r_runner(env, m, p, rollouts=100, n_bins=ec["n_bins"], bin_widen=cfg["hca"]["bin_widen"],
                                       gamma=cfg["gamma"], immediate="rhat", hca_lrs=method_lrs(cfg, "state_hca"))
            rec[f"{m}_100_rollouts"] = np.asarray(jax.block_until_ready(run(keys))["online"])
        metric("div E2", p=round(p, 3), **{k: round(float(np.mean(v)), 4) for k, v in rec.items() if k != "p_long"})
        out["e2"].append(rec)
    write_json(os.path.join(OUT, "divergence.json"), out)


# ---------------------------------------------------------------- compute cost
def cmd_cost(cfg, n_runs):
    tl = tuned_lrs()
    out = {"n_runs": n_runs, "rows": []}
    for exp in ("E1", "E3", "E4", "E6"):
        ec = cfg["experiments"][exp]
        kw = settings(ec)[0]
        env = make_env(ec, kw)
        keys = run_keys(0, exp, 0, n_runs)
        for m in METHODS:
            runner = make_runner(env, m, episodes=ec["episodes"], seg_len=ec["seg_len"], n_bins=ec["n_bins"],
                                 bin_widen=cfg["hca"]["bin_widen"], gamma=cfg["gamma"], variant=cfg["hca"]["state_variant"])
            lrs = method_lrs(cfg, m, tl[exp][0])
            ca = runner.lower(keys, lrs).compile().cost_analysis()
            ca = ca[0] if isinstance(ca, (list, tuple)) else ca
            flops = float(ca.get("flops", float("nan"))) if ca else float("nan")
            tm = Timer()
            jax.block_until_ready(runner(keys, lrs))
            t_first = tm.lap()
            ts = []
            for _ in range(3):
                jax.block_until_ready(runner(keys, lrs))
                ts.append(tm.lap())
            row = {"exp": exp, "method": m, "flops_per_call_xla": flops, "episodes": ec["episodes"],
                   "flops_per_episode": flops / (n_runs * ec["episodes"]), "first_call_s": t_first,
                   "steady_s_median": float(np.median(ts)), "episodes_per_s": n_runs * ec["episodes"] / float(np.median(ts))}
            out["rows"].append(row)
            metric("cost", **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    write_json(os.path.join(OUT, "cost.json"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "eval", "ablation", "theory", "divergence", "cost", "oracle_hz"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_runs", type=int, default=100)
    args = ap.parse_args()
    cfg = load_config()
    check_device()
    os.makedirs(OUT, exist_ok=True)
    t = Timer()
    {"tune": lambda: cmd_tune(cfg, args.n_runs), "eval": lambda: cmd_eval(cfg, args.seed, args.n_runs),
     "ablation": lambda: cmd_ablation(cfg, args.n_runs, args.seed), "theory": lambda: cmd_theory(cfg),
     "divergence": lambda: cmd_divergence(cfg, args.n_runs, args.seed), "cost": lambda: cmd_cost(cfg, args.n_runs),
     "oracle_hz": lambda: cmd_oracle_hz(cfg, args.n_runs, args.seed)}[args.cmd]()
    info("full done", cmd=args.cmd, wall_s=t.lap())



# ---------------------------------------------------------------- oracle-hindsight Return HCA on the bandits
def cmd_oracle_hz(cfg, n_runs, seed=0):
    """Upper-bound probe for the Return-HCA bandit divergence (E6-E8): Return HCA with the EXACT continuous
    hindsight h_z(a|x0,z) = pi(a) p(z|a) / sum_b pi(b) p(z|b) of the current policy (no bins, no learning transient),
    at the paper's alpha = 0.3. G1.6a shows its expected update equals the true gradient, so this isolates what the
    learned/binned h_z costs and bounds what any hindsight-learning detail could buy at alpha = 0.3."""
    from src.eval import start_value
    from src.train import apply_update
    out = {"seed": seed, "n_runs": n_runs, "rows": []}
    for exp in ("E6", "E7", "E8"):
        ec = cfg["experiments"][exp]
        for si, kw in enumerate(settings(ec)):
            env = make_env(ec, kw)
            ea = D.env_arrays(env)
            eps, sig = kw["eps"], kw["sigma"]

            def hz_exact(pi0, z):
                n1 = jnp.exp(-0.5 * ((z - 1.0) / sig) ** 2)
                n2 = jnp.exp(-0.5 * ((z - 2.0) / sig) ** 2)
                p = jnp.array([(1 - eps) * n1 + eps * n2, eps * n1 + (1 - eps) * n2]) * pi0
                return p / jnp.sum(p)

            def one_run(key):
                params = init_params(env.n_obs, ec["n_bins"])
                v0 = start_value(ea, params.theta, env.s0, env.L)

                def body(p, e):
                    pi = policy(p.theta)
                    ep = D.sample_episode(ea, env.s0, env.L, lambda o: pi[o], jax.random.fold_in(key, e))
                    Z = mc_returns(ep, 1.0)
                    hz = jnp.stack([hz_exact(pi[ep.obs[t]], Z[t]) for t in range(env.L)])
                    g = return_hca_grads(p, ep, 1.0, jnp.zeros(ec["n_bins"]), hz_override=hz)
                    p = apply_update(p, g, jnp.asarray(method_lrs(cfg, "return_hca")))
                    return p, start_value(ea, p.theta, env.s0, env.L)

                _, vals = jax.lax.scan(body, params, jnp.arange(ec["episodes"]))
                return jnp.concatenate([v0[None], vals])

            c = np.asarray(jax.jit(jax.vmap(one_run))(run_keys(seed, exp, si, n_runs)))
            s = summarise_curve(c, exp)
            out["rows"].append({"exp": exp, "setting": kw, "final": s["final"], "avg": s["avg"]})
            metric("oracle_hz", exp=exp, setting=si, final=round(float(s["final"].mean()), 4), avg=round(float(s["avg"].mean()), 4))
    write_json(os.path.join(OUT, "oracle_hz.json"), out)


if __name__ == "__main__":
    main()
