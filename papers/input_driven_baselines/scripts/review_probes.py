"""Review probes. Each part prints [METRIC] lines and the whole
result goes to runs/review/review_probes.json.

    python scripts/review_probes.py --parts det,resume,causal,unbiased,budget

det rerun the first 50 iterations of the full-scale paper-config runs (seed 0; state, multi, maml) and diff
          every logged training metric and every test evaluation against runs/full/<arm>/seed_0/metrics.json
resume 50 iterations uninterrupted vs 30 iterations + resume to 50 (multi, maml): history, evals, and the
          final orbax checkpoints (all parameters and optimizer states) must be bit-identical
causal the POLICY observation (and the critic features) at decision t must not depend on any input after t:
          two input sequences that agree on jobs 0..t but differ afterwards, same action noise
unbiased on the real env with the trained full-scale seed-0 policies: the baseline's contribution to the
          policy gradient, d = grad sum_t log pi(a_t|o_t) b_t, must have mean 0 (multi-value own head, MAML
          cross-fitted, state critic), with two negative controls that MUST be detected (in-sample group mean
          incl. own rollout; MAML adapted on the rollout's own half = no cross-fitting)
budget environment steps, rollouts, distinct inputs and critic capacity per iteration, per arm
"""
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src import algorithm as alg  # noqa: E402
from src.common import check_device, metric  # noqa: E402
from src.data import LBInputs, lb_input_kwargs, sample_lb_batch  # noqa: E402
from src.envs import lb_rollout  # noqa: E402
from src.model import n_params  # noqa: E402
from src.train import _ckpt_manager, batch_shape, build, load_config, restore_ckpt, run  # noqa: E402

OUT = os.path.join(ROOT, "runs", "review")
PAPER = os.path.join(ROOT, "src", "configs", "load_balance.yaml")


def _fresh(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def _rows_equal(a, b, keys=None):
    """Exact equality of two lists of metric dicts (NaN == NaN)."""
    bad = []
    for ra, rb in zip(a, b):
        for k in (keys or ra.keys()):
            x, y = ra[k], rb[k]
            if not ((isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y)) or x == y):
                bad.append((ra.get("it"), k, x, y))
    return bad


# ---------------------------------------------------------------- determinism vs the full-scale runs
def part_det(res):
    out = {}
    d = _fresh(os.path.join(OUT, "det"))
    for arm in ("state", "multi", "maml"):
        cfg = load_config(PAPER, ["iterations=50", "eval_every=25"])
        blob = run(copy.deepcopy(cfg), arm, 0, d)
        ref = json.load(open(f"runs/full/{arm}/seed_0/metrics.json"))
        hist_bad = _rows_equal(blob["history"], ref["history"][:50])
        ev_ref = [e for e in ref["evals"] if e["it"] <= 50]
        ev_bad = _rows_equal(blob["evals"], ev_ref, keys=("it", "test_mean", "test_std", "greedy_mean"))
        out[arm] = {"history_rows": len(blob["history"]), "history_mismatches": len(hist_bad),
                    "first_mismatch": [str(x) for x in hist_bad[:3]], "evals_compared": len(ev_ref),
                    "eval_mismatches": len(ev_bad), "eval_first_mismatch": [str(x) for x in ev_bad[:3]],
                    "evals": [(e["it"], e["test_mean"]) for e in blob["evals"]]}
        metric("review det", arm=arm, history_mismatches=len(hist_bad), eval_mismatches=len(ev_bad))
    res["det"] = out


# ---------------------------------------------------------------- resume
def part_resume(res):
    out = {}
    for arm in ("multi", "maml"):
        u = _fresh(os.path.join(OUT, "resume_uninterrupted"))
        r = _fresh(os.path.join(OUT, "resume_interrupted"))
        base = ["eval_every=10"]
        bu = run(load_config(PAPER, base + ["iterations=50"]), arm, 0, u)
        run(load_config(PAPER, base + ["iterations=30"]), arm, 0, r)
        br = run(load_config(PAPER, base + ["iterations=50"]), arm, 0, r)  # resumes from the it-30 checkpoint
        hist_bad = _rows_equal(br["history"], bu["history"])
        ev_bad = _rows_equal(br["evals"], bu["evals"], keys=("it", "test_mean", "test_std", "greedy_mean"))
        ts0, _, _ = build(load_config(PAPER, base), arm, 0)
        pu = restore_ckpt(_ckpt_manager(os.path.join(u, arm, "seed_0", "ckpt")), ts0)
        pr = restore_ckpt(_ckpt_manager(os.path.join(r, arm, "seed_0", "ckpt")), ts0)
        leaves_u, leaves_r = jax.tree_util.tree_leaves(pu), jax.tree_util.tree_leaves(pr)
        n_diff = sum(int(not np.array_equal(np.asarray(a), np.asarray(b))) for a, b in zip(leaves_u, leaves_r))
        out[arm] = {"history_rows": [len(bu["history"]), len(br["history"])], "history_mismatches": len(hist_bad),
                    "eval_its": [[e["it"] for e in bu["evals"]], [e["it"] for e in br["evals"]]],
                    "eval_mismatches": len(ev_bad), "ckpt_leaves": len(leaves_u), "ckpt_leaves_differ": n_diff,
                    "ckpt_it": [int(pu["it"]), int(pr["it"])]}
        metric("review resume", arm=arm, history_mismatches=len(hist_bad), eval_mismatches=len(ev_bad),
               ckpt_leaves_differ=n_diff, ckpt_leaves=len(leaves_u))
    res["resume"] = out


# ---------------------------------------------------------------- causality of the observation
def part_causal(res):
    cfg = json.load(open("runs/full/multi/seed_0/metrics.json"))["config"]
    ts, consts, models = build(cfg, "multi", 0)
    ts = restore_ckpt(_ckpt_manager("runs/full/multi/seed_0/ckpt"), ts)
    spec, am = models["spec"], models["actor"]
    policy = lambda o: am.apply(ts["actor"], o)
    kw = lb_input_kwargs(cfg)
    a = sample_lb_batch(jax.random.PRNGKey(777), 1, **kw)
    b = sample_lb_batch(jax.random.PRNGKey(778), 1, **kw)
    out = []
    for t in (0, 1, 50, 250, 498):
        # A' = A on jobs 0..t, then B's sizes and B's gaps (shifted so arrivals stay non-decreasing)
        arr_a, siz_a = np.asarray(a.arrivals[0]), np.asarray(a.sizes[0])
        gaps_b = np.diff(np.concatenate([[0], np.asarray(b.arrivals[0])]))
        arr2 = arr_a.copy()
        arr2[t + 1:] = arr_a[t] + np.cumsum(gaps_b[t + 1:])
        siz2 = siz_a.copy()
        siz2[t + 1:] = np.asarray(b.sizes[0])[t + 1:]
        A = LBInputs(jnp.asarray(arr_a), jnp.asarray(siz_a))
        A2 = LBInputs(jnp.asarray(arr2, jnp.int32), jnp.asarray(siz2, jnp.int32))
        key = jax.random.PRNGKey(5)
        ta, tb = lb_rollout(spec, policy, A, key), lb_rollout(spec, policy, A2, key)
        n = t + 1
        same_obs = bool(np.array_equal(np.asarray(ta.obs[:n]), np.asarray(tb.obs[:n])))
        same_cobs = bool(np.array_equal(np.asarray(ta.cobs[:n]), np.asarray(tb.cobs[:n])))
        same_act = bool(np.array_equal(np.asarray(ta.actions[:n]), np.asarray(tb.actions[:n])))
        later_differs = bool(t >= 498 or not np.array_equal(np.asarray(ta.obs[n:]), np.asarray(tb.obs[n:])))
        out.append({"t": t, "obs_0..t_identical": same_obs, "critic_feats_0..t_identical": same_cobs,
                    "actions_0..t_identical": same_act, "obs_after_t_differ": later_differs,
                    "inputs_differ_after_t": int(np.sum((arr2 != arr_a) | (siz2 != siz_a)))})
        metric("review causal", t=t, obs_same=same_obs, cobs_same=same_cobs, act_same=same_act,
               later_differs=later_differs)
    # the policy network's input is obs only: confirm its first kernel has obs_dim rows (no index / critic feature)
    k0 = jax.tree_util.tree_leaves(ts["actor"])
    res["causal"] = {"cases": out, "actor_input_dim": int([x for x in k0 if x.ndim == 2][0].shape[0]),
                     "obs_dim": spec.obs_dim, "critic_dim": spec.critic_dim}


# ---------------------------------------------------------------- unbiasedness on the real env
def _zstats(units, u):
    """units [N, P] i.i.d. gradient contributions of the baseline; u [P] unit direction."""
    m = jnp.mean(units, axis=0)
    se = jnp.std(units, axis=0, ddof=1) / np.sqrt(units.shape[0])
    z = m / jnp.maximum(se, 1e-30)
    proj = units @ u
    zp = float(jnp.mean(proj) / (jnp.std(proj, ddof=1) / np.sqrt(proj.shape[0])))
    return {"n_units": int(units.shape[0]), "frac_abs_z_gt_3": float(jnp.mean(jnp.abs(z) > 3)),
            "max_abs_z": float(jnp.max(jnp.abs(z))), "z_proj_true_grad": zp}


def part_unbiased(res, n_groups=512, k=16, ro_seed=99, name="unbiased"):
    out = {}
    for arm in ("multi", "maml", "state"):
        cfg = json.load(open(f"runs/full/{arm}/seed_0/metrics.json"))["config"]
        ts, consts, models = build(cfg, arm, 0)
        ts = restore_ckpt(_ckpt_manager(f"runs/full/{arm}/seed_0/ckpt"), ts)
        spec, am, cm = models["spec"], models["actor"], models["critic"]
        gamma, scale = float(cfg["gamma"]), float(cfg["reward_scale"])
        policy = lambda o: am.apply(ts["actor"], o)
        if arm == "multi":
            z = jax.tree_util.tree_map(lambda x: x[0:1], consts["fixed"])  # fixed sequence 0, its own trained net 0
        else:
            z = sample_lb_batch(jax.random.PRNGKey(424242), 1, **models["inkw"])
        E = n_groups * k
        rep = jax.tree_util.tree_map(lambda x: jnp.repeat(x, E, axis=0), z)
        roll = jax.jit(jax.vmap(lambda inp, kk: lb_rollout(spec, policy, inp, kk)))
        tr = roll(rep, jax.random.split(jax.random.PRNGKey(ro_seed), E))
        G = alg.discounted_returns(tr.rewards * scale, gamma)
        # reference direction: the no-baseline gradient from an INDEPENDENT batch on the same z
        tr2 = roll(rep, jax.random.split(jax.random.PRNGKey(100), E))
        G2 = alg.discounted_returns(tr2.rewards * scale, gamma)
        pgv = jax.jit(lambda o, a, adv: alg.per_rollout_pg(am.apply, ts["actor"], o, a, adv))

        def vecs(o, a, adv, chunk=1024):
            return jnp.concatenate([pgv(o[i:i + chunk], a[i:i + chunk], adv[i:i + chunk])
                                    for i in range(0, o.shape[0], chunk)])

        g_ref = jnp.mean(vecs(tr2.obs, tr2.actions, G2), axis=0)
        u = g_ref / jnp.linalg.norm(g_ref)
        g_full = vecs(tr.obs, tr.actions, G)  # no-baseline per-rollout PG on the test batch
        bases = {}
        if arm == "multi":
            bases["own_head_V0 (as trained)"] = (cm.apply(jax.tree_util.tree_map(lambda p: p[0], ts["critic"]), tr.cobs), 1)
            bases["mismatched_head_V1"] = (cm.apply(jax.tree_util.tree_map(lambda p: p[1], ts["critic"]), tr.cobs), 1)
        elif arm == "state":
            bases["state_critic (as trained)"] = (cm.apply(ts["critic"], tr.cobs), 1)
        else:
            h = k // 2
            xs, Gz = tr.cobs.reshape(n_groups, k, *tr.cobs.shape[1:]), G.reshape(n_groups, k, -1)
            fn = lambda xx, gg, swap: alg.maml_crossfit(
                cm.apply, ts["critic"], xx[:h], gg[:h], xx[h:], gg[h:], steps=int(cfg["meta_inner_steps"]),
                lr=float(cfg["meta_inner_lr"]), inner_reduction=str(cfg["meta_inner_reduction"]), first_order=True,
                clip=float(cfg.get("meta_inner_clip", 0.0)))[:2]
            b1, b2 = jax.vmap(lambda xx, gg: fn(xx, gg, False))(xs, Gz)
            bases["maml_crossfit (as trained)"] = (jnp.concatenate([b1, b2], axis=1).reshape(G.shape), k)
            # NEGATIVE control: adapt on the rollout's OWN half (no cross-fitting)
            adapt = lambda xx, gg: alg.sgd_adapt(cm.apply, ts["critic"], xx, gg, int(cfg["meta_inner_steps"]),
                                                 float(cfg["meta_inner_lr"]), str(cfg["meta_inner_reduction"]),
                                                 float(cfg.get("meta_inner_clip", 0.0)))
            own1 = jax.vmap(lambda xx, gg: cm.apply(adapt(xx[:h], gg[:h]), xx[:h]))(xs, Gz)
            own2 = jax.vmap(lambda xx, gg: cm.apply(adapt(xx[h:], gg[h:]), xx[h:]))(xs, Gz)
            bases["NEG maml_adapted_on_own_half"] = (jnp.concatenate([own1, own2], axis=1).reshape(G.shape), k)
        # NEGATIVE control (every arm): in-sample per-group mean of returns, own rollout INCLUDED (bias = g/k)
        Gm = jnp.repeat(jnp.mean(G.reshape(n_groups, k, -1), axis=1), k, axis=0)
        bases["NEG in-sample group mean (incl. own)"] = (Gm, k)
        # POWER controls with a known bias = c * (true gradient): the baseline reads c x the rollout's own return
        for c in (0.1, 0.25, 1.0):
            bases[f"NEG {c} x own return (bias = {c} g)"] = (c * G, 1)
        gp = g_full @ u
        res_arm = {"z_fingerprint": int(np.asarray(z.sizes).sum()), "rollouts": E,
                   "true_grad_norm": float(jnp.linalg.norm(g_ref)),
                   "signal_z_of_no_baseline_grad_on_u": float(jnp.mean(gp) / (jnp.std(gp, ddof=1) / np.sqrt(E)))}
        metric("review unbiased signal", arm=arm, z=round(res_arm["signal_z_of_no_baseline_grad_on_u"], 2))
        for nm, (b, grp) in bases.items():
            d = g_full - vecs(tr.obs, tr.actions, G - b)  # = per-rollout grad of sum_t log pi * b_t
            units = d.reshape(n_groups, k, -1).sum(1) if grp > 1 else d  # groups are the i.i.d. unit when b couples rollouts
            st = _zstats(units, u)
            st["bias_norm_over_true_grad_norm"] = float(jnp.linalg.norm(jnp.mean(d, 0)) / jnp.linalg.norm(g_ref))
            st["mse_vs_G"] = float(jnp.mean((G - b) ** 2))
            res_arm[nm] = st
            metric("review unbiased", rep=name, arm=arm, baseline=nm, **{kk: round(v, 4) if isinstance(v, float) else v
                                                                  for kk, v in st.items()})
        out[arm] = res_arm
    res[name] = out


def part_unbiased_reps(res):
    """Three more independent rollout batches (same z, same reference direction u); combined with the first
    batch by Stouffer's method in (4 x 8192 rollouts per policy)."""
    for r, sd in enumerate((1099, 2099, 3099)):
        part_unbiased(res, ro_seed=sd, name=f"unbiased_rep{r + 1}")


# ---------------------------------------------------------------- budget accounting
def part_budget(res):
    out = {}
    for tag, sets in (("paper", []), ("tuned", ["lr_critic=1e-2", "meta_outer_lr=1e-2", "entropy_decay_iters=1000"])):
        cfg = load_config(PAPER, sets)
        for arm in ("none", "state", "multi", "maml", "multi_shuffled"):
            ts, consts, models = build(cfg, arm, 0)
            g, kk = batch_shape(cfg, arm)
            lr_c = cfg["meta_outer_lr"] if arm == "maml" else cfg["lr_critic"]
            out[f"{tag}/{arm}"] = {"rollouts_per_iter": g * kk, "distinct_inputs_per_iter": g, "rollouts_per_input": kk,
                                   "env_steps_per_iter": g * kk * models["spec"].n_jobs,
                                   "distinct_training_inputs_total": (10 if arm.startswith("multi") else g * int(cfg["iterations"])),
                                   "actor_params": n_params(ts["actor"]), "critic_params": n_params(ts["critic"]),
                                   "lr_actor": cfg["lr_actor"], "critic_lr_used": lr_c,
                                   "critic_updates_per_iter_per_net": (round(cfg["mv_groups"] / cfg["n_value_nets"], 2)
                                                                       if arm.startswith("multi") else (1 if arm != "none" else 0)),
                                   "entropy_decay_iters": cfg["entropy_decay_iters"], "gamma": cfg["gamma"]}
    res["budget"] = out
    for k, v in out.items():
        metric("review budget", cfg_arm=k, **{a: b for a, b in v.items() if not isinstance(b, (list, dict))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="det,resume,causal,unbiased,budget")
    a = ap.parse_args()
    check_device()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "review_probes.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for p in a.parts.split(","):
        globals()[f"part_{p}"](res)
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)
    print("REVIEW_PROBES_DONE", flush=True)


if __name__ == "__main__":
    main()
