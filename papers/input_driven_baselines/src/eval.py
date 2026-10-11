"""Evaluation utilities: heuristics on the test set, the matched-policy gradient-variance probe,
and the toy / small-scale drivers.

    python -m src.eval --toys  # App. A toy (writes runs/toys/metrics.json)
    python -m src.eval --heuristics --config...  # SPT / LRT / EFT / JSQ on the 100 test inputs
    python -m src.eval --probe runs/small  # gradient-variance probe on the trained small-scale policies
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from . import algorithm as alg  # noqa: E402
from .common import check_device, info, metric, write_json  # noqa: E402
from .data import PROBE_SEED, full_sha, lb_input_kwargs, sample_lb_batch  # noqa: E402
from .envs import lb_rollout, make_lb_spec  # noqa: E402
from .model import init_critic  # noqa: E402


# ---------------------------------------------------------------- gradient-variance probe (matched policy)
def grad_variance_probe(cfg: dict, actor_m, actor_p, key, n_z: int = 64, k: int = 8, n_fit_z: int = 256,
                        fit_steps: int = 3000, meta=None, probe_inputs=None, mv=None):
    """At ONE fixed policy, compare the per-rollout policy-gradient estimator under several baselines on
    the SAME batch of n_z input sequences x k rollouts each.  Variance = trace of the covariance over all
    n_z*k rollouts (so it includes the input-driven spread -- unlike the within-iteration Var(PG) of a
    MAML/multi batch, whose rollouts share inputs).

    baselines: none | state (fresh critic fitted to this policy on n_fit_z other inputs) |
               input_loo (mean return-to-go of the other k-1 rollouts on the same z, same step) |
               state+loo (state critic + same-z leave-one-out mean of the residual G - V) |
               maml (cross-fitted adapted meta-critic, if ``meta=(critic_m, phi, cfg)`` is given)
               multi_own / multi_mismatched (if ``mv=(critic_m, stack)`` and ``probe_inputs`` = the arm's N fixed
               sequences: the trained value net of the rollout's own sequence i, or of sequence (i+1) mod N)
    All are unbiased (Lemma 1: they never read this rollout's own actions)."""
    spec = make_lb_spec(cfg)
    inkw = lb_input_kwargs(cfg)
    gamma, scale = float(cfg["gamma"]), float(cfg["reward_scale"])
    k_fit_in, k_fit_ro, k_in, k_ro, k_c = jax.random.split(key, 5)
    policy = lambda o: actor_m.apply(actor_p, o)
    roll = jax.jit(jax.vmap(lambda inp, kk: lb_rollout(spec, policy, inp, kk)))

    # --- fit a state critic to THIS policy on separate inputs
    fit_inp = sample_lb_batch(k_fit_in, n_fit_z, **inkw)
    tr = roll(fit_inp, jax.random.split(k_fit_ro, n_fit_z))
    Gf = alg.discounted_returns(tr.rewards * scale, gamma)
    cm, cp = init_critic(k_c, tuple(cfg["hidden"]), spec.critic_dim, cfg["activation"])
    tx = optax.adam(1e-3)

    @jax.jit
    def fit(cp):
        def body(c, _):
            p, o = c
            g = jax.grad(lambda q: alg.value_loss(cm.apply, q, tr.cobs, Gf, "mean"))(p)
            u, o = tx.update(g, o, p)
            return (optax.apply_updates(p, u), o), None

        (p, _), _ = jax.lax.scan(body, (cp, tx.init(cp)), None, length=fit_steps)
        return p

    cp = fit(cp)

    # --- probe batch
    if probe_inputs is not None:
        inp = probe_inputs
        n_z = int(inp.arrivals.shape[0])
    else:
        inp = sample_lb_batch(k_in, n_z, **inkw)
    inp_rep = jax.tree_util.tree_map(lambda x: jnp.repeat(x, k, axis=0), inp)
    tr = roll(inp_rep, jax.random.split(k_ro, n_z * k))
    G = alg.discounted_returns(tr.rewards * scale, gamma)  # [n_z*k, T]
    Vs = cm.apply(cp, tr.cobs)
    Gz = G.reshape(n_z, k, -1)
    loo = (jnp.sum(Gz, axis=1, keepdims=True) - Gz) / (k - 1)
    Rz = (G - Vs).reshape(n_z, k, -1)
    loo_res = (jnp.sum(Rz, axis=1, keepdims=True) - Rz) / (k - 1)
    bases = {"none": jnp.zeros_like(G), "state": Vs, "input_loo": loo.reshape(G.shape),
             "state+loo": Vs + loo_res.reshape(G.shape)}
    if meta is not None:
        mcm, phi, mcfg = meta
        h = k // 2
        xs = tr.cobs.reshape(n_z, k, *tr.cobs.shape[1:])
        b1, b2, _, aux = jax.vmap(lambda xx, gg: alg.maml_crossfit(
            mcm.apply, phi, xx[:h], gg[:h], xx[h:], gg[h:], steps=int(mcfg["meta_inner_steps"]),
            lr=float(mcfg["meta_inner_lr"]), inner_reduction=str(mcfg["meta_inner_reduction"]),
            first_order=True, clip=float(mcfg.get("meta_inner_clip", 0.0))))(xs, Gz)
        bases["maml"] = jnp.concatenate([b1, b2], axis=1).reshape(G.shape)
        bases["maml_unadapted"] = mcm.apply(phi, tr.cobs)
        if h >= 8:  # same rollouts, but each half's baseline adapted on only h/2 of the OTHER half's rollouts
            q = h // 2  # (isolates the adaptation-set size; an earlier probe adapted on 4, training on 8)
            phi_on = lambda xx, gg: alg.sgd_adapt(mcm.apply, phi, xx, gg, int(mcfg["meta_inner_steps"]),
                                                  float(mcfg["meta_inner_lr"]), str(mcfg["meta_inner_reduction"]),
                                                  float(mcfg.get("meta_inner_clip", 0.0)))
            bh1 = jax.vmap(lambda xx, gg: mcm.apply(phi_on(xx[h:h + q], gg[h:h + q]), xx[:h]))(xs, Gz)  # H1 <- H2[:q]
            bh2 = jax.vmap(lambda xx, gg: mcm.apply(phi_on(xx[:q], gg[:q]), xx[h:]))(xs, Gz)  # H2 <- H1[:q]
            bases[f"maml_adapt_on_{q}"] = jnp.concatenate([bh1, bh2], axis=1).reshape(G.shape)

    if mv is not None:
        mvm, stack = mv
        own = jnp.repeat(jnp.arange(n_z), k)
        for nm, net in (("multi_own", own), ("multi_mismatched", (own + 1) % n_z)):
            pe = jax.tree_util.tree_map(lambda p: p[net], stack)
            bases[nm] = jax.vmap(mvm.apply)(pe, tr.cobs)

    pgv = jax.jit(lambda adv: alg.per_rollout_pg(actor_m.apply, actor_p, tr.obs, tr.actions, adv))
    out = {}
    ref_mean = None
    for name, b in bases.items():
        vecs = pgv(G - b)
        mean = jnp.mean(vecs, axis=0)
        out[name] = {"var_pg": float(alg.pg_variance(vecs)), "baseline_mse": float(jnp.mean((G - b) ** 2)),
                     "mean_grad": np.asarray(mean)}
        if name == "none":
            ref_mean, ref_vecs = mean, vecs
    # unbiasedness: paired difference of per-rollout gradients vs the no-baseline estimator
    for name in bases:
        if name == "none":
            continue
        diff = pgv(G - bases[name]) - ref_vecs  # = -psi * b summed; zero-mean if b unbiased
        dm = jnp.mean(diff, axis=0)
        dse = jnp.std(diff, axis=0, ddof=1) / np.sqrt(diff.shape[0])
        z = dm / jnp.maximum(dse, 1e-12)
        out[name]["bias_max_abs_z"] = float(jnp.max(jnp.abs(z)))
        out[name]["bias_frac_abs_z_gt_3"] = float(jnp.mean(jnp.abs(z) > 3.0))
    # heavy-tailed returns make the trace dominated by a few rollouts: also report a robust per-coordinate
    # summary (median over parameters of the per-coordinate variance ratio vs the state baseline)
    vs = jnp.var(pgv(G - bases["state"]), axis=0, ddof=1)
    for name, b in bases.items():
        vb = jnp.var(pgv(G - b), axis=0, ddof=1)
        out[name]["median_coord_var_ratio_vs_state"] = float(jnp.median(vb / jnp.maximum(vs, 1e-30)))
    for v in out.values():
        v.pop("mean_grad")
    s = out["state"]["var_pg"]
    for name in out:
        out[name]["var_ratio_vs_state"] = out[name]["var_pg"] / s
    return out


# ---------------------------------------------------------------- small-scale probe driver
def probe_runs(run_root: str, seeds=None, arms=None, maml_k=None):
    """maml_k: rollouts per input for a MAML policy's probe. Default (None) = the arm's training meta_k, so the
    cross-fitted baseline adapts on halves of meta_k/2 rollouts exactly as in training (the small-scale/3
    probes before the fix used the generic k = 8, i.e. halves of 4)."""
    from .train import build, restore_ckpt, _ckpt_manager

    results = {}
    for arm in sorted(os.listdir(run_root)):
        arm_dir = os.path.join(run_root, arm)
        if not os.path.isdir(arm_dir) or arm.startswith("_") or (arms is not None and arm not in arms):
            continue
        for sd in sorted(os.listdir(arm_dir)):
            if not sd.startswith("seed_"):
                continue
            seed = int(sd.split("_")[1])
            if seeds is not None and seed not in seeds:
                continue
            with open(os.path.join(arm_dir, sd, "metrics.json")) as f:
                cfg = json.load(f)["config"]
            ts, consts, models = build(cfg, arm, seed)
            init_actor = ts["actor"]
            init_critic_p = ts["critic"]
            ts = restore_ckpt(_ckpt_manager(os.path.join(arm_dir, sd, "ckpt")), ts)
            key = jax.random.fold_in(jax.random.PRNGKey(PROBE_SEED), seed)
            meta = (models["critic"], ts["critic"], cfg) if arm == "maml" else None
            kk = (int(maml_k) if maml_k else int(cfg["meta_k"])) if arm == "maml" else 8
            nz = 512 // kk  # same 512 probe rollouts whatever k is (k = 8: 64 inputs; k = 16: 32 inputs)
            r_final = grad_variance_probe(cfg, models["actor"], ts["actor"], key, n_z=nz, k=kk, meta=meta)
            r_final["_probe_k"], r_final["_probe_n_z"] = kk, nz
            res = {"final_policy": r_final}
            if arm in ("multi", "multi_shuffled"):  # the arm's own N value nets on its N fixed sequences
                res["fixed_set"] = grad_variance_probe(cfg, models["actor"], ts["actor"], key, k=16,
                                                       probe_inputs=consts["fixed"], mv=(models["critic"], ts["critic"]))
                metric("probe_fixed", arm=arm, seed=seed, **{f"{b}_ratio": round(v["var_ratio_vs_state"], 4)
                                                            for b, v in res["fixed_set"].items()})
            if arm == "state":  # the init policy is shared by every arm with the same seed: probe it once
                res["init_policy"] = grad_variance_probe(cfg, models["actor"], init_actor, key)
            results.setdefault(arm, {})[seed] = res
            metric("probe", arm=arm, seed=seed, k=kk,
                   **{f"{b}_ratio": round(v["var_ratio_vs_state"], 4) for b, v in r_final.items()
                      if isinstance(v, dict)})
            del init_critic_p
    return results


# ---------------------------------------------------------------- heuristics
def heuristics_table(cfg: dict) -> dict:
    from .train import heuristic_returns, test_inputs_for

    spec = make_lb_spec(cfg)
    ti = test_inputs_for(cfg)
    hr = heuristic_returns(cfg, spec, ti)
    out = {"test_sha256": full_sha(ti)}
    for k, v in hr.items():
        out[k] = {"mean": float(v.mean()), "std": float(v.std()), "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                  "median": float(np.median(v))}
        metric("heuristic", name=k, **out[k])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--toys", action="store_true")
    ap.add_argument("--heuristics", action="store_true")
    ap.add_argument("--probe", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default=None, help="comma list of arms to probe (default: all)")
    ap.add_argument("--seeds", default=None, help="comma list of seeds to probe (default: all)")
    ap.add_argument("--maml-k", type=int, default=None, help="MAML probe rollouts per input (default: training meta_k)")
    a = ap.parse_args(argv)
    check_device()
    if a.toys:
        from .toys import run_toys

        run_toys(a.out or "runs/toys")
    if a.heuristics:
        from .train import load_config

        cfg = load_config(a.config, a.set)
        res = heuristics_table(cfg)
        if a.out:
            write_json(a.out, res)
    if a.probe:
        res = probe_runs(a.probe, arms=a.arms.split(",") if a.arms else None,
                         seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None, maml_k=a.maml_k)
        write_json(a.out or os.path.join(a.probe, "probe.json"), res)
        info("probe written")


if __name__ == "__main__":
    main()
