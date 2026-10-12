"""Instrument runner (E1 gradient quality, E2 value floor, E3 baselines, E4 landscapes) on saved checkpoints.

All instrument rollouts use FROZEN normalisers  and run on the GPU as wide vmapped batches; the
checkpoint holds every training seed stacked, and the instruments are vmapped over seeds.

Sampling design : a sample of n <= chunk pairs is a fresh single-actor n-step rollout
from reset; a sample of n > chunk pairs is n/chunk independent fresh chunk-step actors. The m estimates at all
sizes come from ONE batch of m x (n_max/chunk) actors: the n <= chunk estimates are the n-step PREFIXES of each
estimate's first actor (a prefix of a fresh rollout from reset IS a fresh n-step rollout from reset; GAE is
recomputed on the prefix). Estimates at different n are therefore nested, which leaves every per-n statistic
unchanged. The reference uses a disjoint key branch (k_ref) from the estimates (k_est).

Usage: python -m src.eval --run runs/small/hopper_ppo --its 0 48 --exps grad value baseline landscape [...]
"""
from __future__ import annotations

import argparse
import json
import os
from functools import partial

from . import common
from .common import Timer, info, metric, warn, write_json

import jax
import jax.numpy as jnp
import numpy as np

from . import algorithm as A
from . import data as D
from .configs.default import INSTRUMENTS
from .train import build, load_ckpt


def run_overrides(run_dir: str | None) -> dict:
    """Config overrides that make the instruments use the env a run was TRAINED in : the stored
    config of the run's first seed; a run without the key (every full-scale run) is the legacy env."""
    import glob
    if run_dir:
        for f in sorted(glob.glob(os.path.join(run_dir, "seed_*", "metrics.json"))):
            cfg = json.load(open(f)).get("config", {})
            return {"env_post_step_kinematics": bool(cfg.get("env_post_step_kinematics", False))}
    return {"env_post_step_kinematics": False}


class Ctx:
    def __init__(self, env_name, alg, overrides=None, run_dir=None):
        """Without run_dir and without an explicit override the LEGACY env is used, because every checkpoint
        written before the env fix was trained in it ; pass run_dir to follow the run's stored config."""
        ov = {**run_overrides(run_dir), **(overrides or {})}
        (self.cfg, self.env, self.spec, self.nets, self.rollout, self.update,
         self.init_state) = build(env_name, alg, ov)
        self.pi_tx, self.v_tx = A.make_optimizers(self.cfg)
        # value net with one extra input, the episode step / VSTAR_T_HORIZON (recheck: official
        # StateWithTime(horizon=T=2048) appends counter/T; counter = steps since reset)
        from .model import Nets
        self.nets_t = Nets(self.spec.obs_dim + 1, self.spec.act_dim, self.cfg)


def guard(tree, tag: int):
    """NaN/Inf guard inside every jitted instrument: [ERROR] + dump + abort on a non-finite leaf.
    Tags: 11 gradient block, 12 V* fit, 13 landscape step/surface/true-reward sums."""
    common.nan_guard(tree, {}, tag=tag, dump_dir=os.path.join(common.REPO, "nan_dumps"))


# ------------------------------------------------------------------ sampling
VSTAR_STEPS = 0   # >0: override INSTRUMENTS['vstar_steps'] (divergence test: is V* under-fit?)
POINT_BATCH = 0   # >0: landscape grid points per lax.map batch (GPU memory bound)
LAND_NO_ANNEAL = False   # recheck (labelled arm): PPO landscape step at the un-annealed Table-1 LR
ACTOR_BATCH = 0   # >0: run actors in lax.map groups of this width (CPU: MJX throughput peaks at ~32-64 envs/process)


def collect(ctx: Ctx, pi, norm, keys, T: int):
    """keys [..., A]: independent fresh T-step single-actor rollouts, frozen normalisers -> Traj [..., T].
    Actor batching changes only the execution order; every actor keeps its own key, so results are identical."""
    one = lambda k: ctx.rollout(pi, norm, k, T, False)[0]
    K = int(np.prod(keys.shape[:-1]))
    B = ACTOR_BATCH
    if B and K > B:
        flat = keys.reshape((K,) + keys.shape[-1:])
        pad = (-K) % B
        flat = jnp.concatenate([flat, flat[:pad]]) if pad else flat
        tr = jax.lax.map(jax.vmap(one), flat.reshape((-1, B) + keys.shape[-1:]))
        return jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:])[:K].reshape(keys.shape[:-1] + x.shape[2:]), tr)
    f = one
    for _ in range(keys.ndim - 1):
        f = jax.vmap(f)
    return f(keys)


def advantages(ctx: Ctx, traj, values, kind: str):
    """Advantages for actors [A, T] (or [T]): 'gae' with the supplied values, or 'zero' = MC return ."""
    g = partial(A.gae, gamma=ctx.cfg["gamma"], lam=ctx.cfg["gae_lambda"])
    f = g if traj.rew.ndim == 1 else jax.vmap(g)
    adv, ret = f(traj.rew, values, traj.not_done)
    return (ret if kind == "zero" else adv), ret


def sub_sample(traj, n: int, T: int):
    """The first n pairs of an estimate's actors [A, T]: prefix of actor 0 if n <= T, else the first n/T actors."""
    if n <= T:
        return jax.tree_util.tree_map(lambda x: x[0, :n], traj)
    assert n % T == 0, (n, T)
    return jax.tree_util.tree_map(lambda x: x[: n // T], traj)


MC_KINDS = {"mc_vtheta": "gae", "mc_vstar": "vstar", "mc_vstar_t": "vstar_t"}
VSTAR_T_HORIZON = 2048.0   # official torch_utils.StateWithTime(horizon=self.T), T = 2048


def time_feature(traj):
    """counter / T with counter = steps since reset at the observed state (ep_len is t+1 at step t)."""
    return ((traj.ep_len - 1.0) / VSTAR_T_HORIZON)[..., None]


def grad_of(ctx, pi, v, traj, kind="gae", vstar=None, normalize=True, vstar_t=None):
    """Policy-gradient estimate from one sample traj ([T] or [A, T]) with baseline kind in
    {gae (agent V), vstar, zero (A := G_t, then batch-normalised: (G - mean G)/std G, i.e. a CONSTANT batch-mean
    baseline), zero_raw (A := G_t with no centring at all: the literal 'no baseline'; the scale is irrelevant to a
    cosine), zero_gae (GAE with V == 0: the ambiguity ablation),
    vstar_t (GAE with the time-aware V*(s, t/2048)), mc_vtheta / mc_vstar / mc_vstar_t
    (A := G_t - V(s_t), the literal eq. (2)/(9) baseline, with V = V_theta / V* / V*(s, t); normalised like every other kind)}."""
    vk = MC_KINDS.get(kind, kind)
    if vk == "vstar":
        values = vstar_apply(ctx, vstar, traj.obs)
    elif vk == "vstar_t":
        values = vstar_t_apply(ctx, vstar_t, traj)
    elif kind == "zero_gae":
        values = jnp.zeros_like(traj.rew)
    else:
        values = ctx.nets.v_apply(v, traj.obs)
    adv, ret = advantages(ctx, traj, values, "zero" if kind in ("zero", "zero_raw") else "gae")
    if kind in MC_KINDS:
        adv = ret - values
    if kind == "zero_raw":
        normalize = False
    obs = traj.obs.reshape((-1, traj.obs.shape[-1]))
    act = traj.act.reshape((-1, traj.act.shape[-1]))
    return A.pg_vector(ctx.nets, pi, obs, act, adv.reshape(-1), normalize)


# ------------------------------------------------------------------ E1 / E3
def gradient_block(ctx, pi, v, norm, root, it, sizes, m, chunk, ref_pairs, kinds=("gae",), vstar=None,
                   raw_ablation=True, pool_ablation=False, vstar_t=None):
    """Returns dict: g_ref [d], G[kind][n] [m, d] (+ 'gae_raw' and 'gae_pool' ablations)."""
    traj, tr_ref = grad_collect(ctx, pi, norm, root, it, sizes, m, chunk, ref_pairs)
    return grad_compute(ctx, pi, v, traj, tr_ref, root, it, sizes, m, chunk, kinds, vstar, raw_ablation, pool_ablation,
                        vstar_t)


def grad_collect(ctx, pi, norm, root, it, sizes, m, chunk, ref_pairs):
    """The rollouts of one gradient block: m estimates x (n_max/chunk) fresh actors (k_est) and the reference
    (ref_pairs/chunk fresh actors, k_ref; None if ref_pairs == 0)."""
    n_max = max(sizes)
    T = min(n_max, chunk)
    A_est = max(n_max // chunk, 1)
    k_est = common.k_instr(root, it, common.I_EST)
    keys = jax.vmap(lambda i: jax.random.split(jax.random.fold_in(k_est, i), A_est))(jnp.arange(m))
    traj = collect(ctx, pi, norm, keys, T)                                   # [m, A_est, T]
    tr_ref = None
    if ref_pairs:
        k_ref = common.k_instr(root, it, common.I_REF)
        T_ref = min(ref_pairs, chunk)
        tr_ref = collect(ctx, pi, norm, jax.random.split(k_ref, max(ref_pairs // chunk, 1)), T_ref)
    return traj, tr_ref


def grad_compute(ctx, pi, v, traj, tr_ref, root, it, sizes, m, chunk, kinds=("gae",), vstar=None,
                 raw_ablation=True, pool_ablation=False, vstar_t=None):
    n_max = max(sizes)
    T = min(n_max, chunk)
    k_est = common.k_instr(root, it, common.I_EST)
    out = {"G": {}}
    for kind in kinds:
        for n in sizes:
            sub = jax.vmap(lambda tr: sub_sample(tr, n, T))(traj)
            out["G"].setdefault(kind, {})[n] = jax.vmap(lambda tr: grad_of(ctx, pi, v, tr, kind, vstar,
                                                                           vstar_t=vstar_t))(sub)
            if raw_ablation and kind == "gae":
                out["G"].setdefault("gae_raw", {})[n] = jax.vmap(
                    lambda tr: grad_of(ctx, pi, v, tr, "gae", None, normalize=False))(sub)
    if pool_ablation:
        # n pairs drawn uniformly (with replacement; duplicates negligible at n <= pool/10) from the POOL
        # of all m x n_max estimate pairs, GAE computed first on the full chunk-length actors. Pool estimates share
        # no key with the reference (k_ref); they reuse the estimates' rollouts (so they are not independent of the
        # fresh-from-reset estimates, only of each other up to pool overlap).
        flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), traj)     # [m*A_est, T]
        adv_pool, _ = advantages(ctx, flat, ctx.nets.v_apply(v, flat.obs), "gae")
        P = adv_pool.size
        obs_p = flat.obs.reshape((P, -1))
        act_p = flat.act.reshape((P, -1))
        adv_p = adv_pool.reshape(-1)
        k_pool = jax.random.fold_in(k_est, 1_000_003)
        for n in sizes:
            if n > P // 10:
                continue
            ix = jax.vmap(lambda i: jax.random.randint(jax.random.fold_in(jax.random.fold_in(k_pool, n), i),
                                                       (n,), 0, P))(jnp.arange(m))
            out["G"].setdefault("gae_pool", {})[n] = jax.vmap(
                lambda j: A.pg_vector(ctx.nets, pi, obs_p[j], act_p[j], adv_p[j], True))(ix)
    if tr_ref is not None:
        out["g_ref"] = grad_of(ctx, pi, v, tr_ref, "gae")
        out["g_ref_raw"] = grad_of(ctx, pi, v, tr_ref, "gae", normalize=False)
    guard(out, 11)
    return out


def fit_vstar(ctx, pi, norm, root, it, pairs, chunk, with_time=False):
    """V* fit to complete-episode MC returns on `pairs` fresh pairs . with_time=True also fits, on the SAME
    pairs with the SAME fit key, the time-aware V*(s, t/2048)  and returns
    ((vs, mu, sd), info, (vs_t, mu_t, sd_t), info_t); the obs-only fit is unchanged by the flag."""
    k = common.k_instr(root, it, common.I_VSTAR)
    k_data, k_fit = jax.random.split(k)
    T = min(pairs, chunk)
    tr = collect(ctx, pi, norm, jax.random.split(k_data, max(pairs // chunk, 1)), T)   # [A, T]
    _, ret = advantages(ctx, tr, jnp.zeros_like(tr.rew), "zero")
    mask = A.complete_episode_mask(tr.not_done)
    obs = tr.obs.reshape((-1, tr.obs.shape[-1]))
    I = INSTRUMENTS
    vs, mu, sd, inf = A.fit_value(ctx.nets, k_fit, obs, ret.reshape(-1), mask.reshape(-1), I["vstar_lr"],
                                  VSTAR_STEPS or I["vstar_steps"], I["vstar_batch"])
    guard((vs, mu, sd), 12)
    if not with_time:
        return (vs, mu, sd), inf
    obs_t = jnp.concatenate([tr.obs, time_feature(tr)], -1).reshape((-1, tr.obs.shape[-1] + 1))
    vt, mut, sdt, inft = A.fit_value(ctx.nets_t, k_fit, obs_t, ret.reshape(-1), mask.reshape(-1), I["vstar_lr"],
                                     VSTAR_STEPS or I["vstar_steps"], I["vstar_batch"])
    guard((vt, mut, sdt), 12)
    return (vs, mu, sd), inf, (vt, mut, sdt), inft


def vstar_apply(ctx, vstar, obs):
    vs, mu, sd = vstar
    return mu + sd * ctx.nets.v_apply(vs, obs)


def vstar_t_apply(ctx, vstar_t, traj):
    vs, mu, sd = vstar_t
    return mu + sd * ctx.nets_t.v_apply(vs, jnp.concatenate([traj.obs, time_feature(traj)], -1))


def value_floor(ctx, pi, v, vstar, norm, root, it, n_rollouts=10):
    """C7: Returns-MRE / MSE of V_theta vs the V* floor on the same fresh held-out states ."""
    k = jax.random.fold_in(common.k_instr(root, it, common.I_HELDOUT), 7)
    tr = collect(ctx, pi, norm, jax.random.split(k, n_rollouts), int(INSTRUMENTS["heldout_pairs"]))
    vals = ctx.nets.v_apply(v, tr.obs)
    adv, ret = advantages(ctx, tr, vals, "gae")
    vsv = vstar_apply(ctx, vstar, tr.obs)
    nd = tr.not_done
    return dict(returns_mre_vtheta=A.masked_mean(jnp.abs(vals - ret) / jnp.abs(ret), nd),
                returns_mre_vstar=A.masked_mean(jnp.abs(vsv - ret) / jnp.abs(ret), nd),
                returns_mse_vtheta=A.masked_mean((vals - ret) ** 2, nd),
                returns_mse_vstar=A.masked_mean((vsv - ret) ** 2, nd),
                gae_mre_true_vtheta_self=A.masked_mean(jnp.abs(adv) / jnp.abs(vals + adv), nd))


# ------------------------------------------------------------------ E4
def policy_step(ctx, ts, norm, root, it, n_pairs, chunk, branch):
    """The update step the algorithm takes from theta_k on a fresh n_pairs sample (k_land branch 2 or 3)."""
    k = jax.random.fold_in(common.k_instr(root, it, common.I_LAND), branch)
    k_data, k_upd = jax.random.split(k)
    n_act = n_pairs // chunk if (n_pairs > chunk and n_pairs % chunk == 0) else 1   # 2048 -> one actor (training batch)
    tr = collect(ctx, ts.pi, norm, jax.random.split(k_data, n_act), n_pairs // n_act)
    vals = ctx.nets.v_apply(ts.v, tr.obs)
    adv, ret = advantages(ctx, tr, vals, "gae")
    fl = lambda x: x.reshape((-1,) + x.shape[2:])
    b = A.Batch(fl(tr.obs), fl(tr.act), fl(tr.logp), fl(adv), fl(ret), fl(vals), fl(tr.not_done))
    cfg = ctx.cfg
    if cfg["alg"] == "ppo":
        lr = cfg["policy_lr"] * (1.0 if LAND_NO_ANNEAL else A.lr_scale(cfg, it))
        pi1, _, st = A.ppo_policy_update(ctx.nets, cfg, ts.pi, ts.pi_opt, ctx.pi_tx, b, lr, k_upd)
    else:
        pi1, st = A.trpo_policy_update(ctx.nets, cfg, ts.pi, b, k_upd)
    return pi1 - ts.pi, b, st


def start_selected(not_done, ep_len, epl: int):
    """[..., T] mask of episode ENDS whose episode started at step <= T - epl (recheck 'start' estimator):
    exactly the episodes that start early enough to be complete whatever their length."""
    T = not_done.shape[-1]
    start = jnp.arange(T, dtype=jnp.float32) - ep_len + 1.0              # first step of the episode ending at t
    return (not_done < 0.5) & (start <= T - epl)


def true_reward(ctx, pi_grid, norm, keys, pairs, T_chunk):
    """Per grid point and per actor, two estimators of the mean episode return from the same rollouts:
    'all'    = EVERY episode completed within the actor's T_chunk pairs (the paper's/official-code estimator: the
               true reward at n pairs uses all trajectories that fit in n pairs). Also returns the
               pairs actually used (sum of completed-episode lengths); the truncated tail of each actor is unused.
    'firstk' = only the first floor(T/1000) episodes per actor (length-unbiased but spends ~2 episodes per actor,
               i.e. far fewer pairs than the label; the pre- full-scale estimator, kept for comparison).
    'start'  = every episode that STARTS at step <= T - episode_length of its actor . Such an
               episode always completes inside the actor, and whether an episode starts before the cut-off depends
               only on the episodes before it (a stopping time), so by Wald's identity sum(R)/count is unbiased up to
               O(1/count); 'all' instead keeps a truncated episode only if it happened to be short (length bias).
    Subsets of the actors give the Fig 5 concentration levels (2e3 / 2e4 / 1e5 pairs) from one run."""
    n_act = max(pairs // T_chunk, 1)
    T = min(pairs, T_chunk)
    epl = int(ctx.cfg["episode_length"])

    def one(p, k):
        tr = collect(ctx, p, norm, jax.random.split(k, n_act), T)       # [n_act, T]
        sel, rets = D.unbiased_episode_returns(tr, epl)
        selc = sel.astype(jnp.float32)
        allc = (tr.not_done < 0.5).astype(jnp.float32)
        r = tr.ep_ret
        stc = start_selected(tr.not_done, tr.ep_len, epl).astype(jnp.float32)
        return dict(all_sum=jnp.sum(allc * r, -1), all_sq=jnp.sum(allc * r ** 2, -1), all_cnt=jnp.sum(allc, -1),
                    all_pairs=jnp.sum(allc * tr.ep_len, -1),
                    k_sum=jnp.sum(selc * rets, -1), k_sq=jnp.sum(selc * rets ** 2, -1), k_cnt=jnp.sum(selc, -1),
                    st_sum=jnp.sum(stc * r, -1), st_sq=jnp.sum(stc * r ** 2, -1), st_cnt=jnp.sum(stc, -1))

    if POINT_BATCH:      # bound GPU memory: grid points in lax.map batches (same keys, same numbers)
        return jax.lax.map(lambda a: one(*a), (pi_grid, keys), batch_size=POINT_BATCH)
    return jax.vmap(one)(pi_grid, keys)                                 # keys [grid]: one per grid point


def reduce_true(sums, sqs, cnts, n_actors):
    """Mean and SE of the episode return over the first n_actors actors -> arrays [grid]."""
    s, q, c = (np.asarray(x)[:, :n_actors].sum(1) for x in (sums, sqs, cnts))
    m = s / np.maximum(c, 1)
    var = (q - c * m ** 2) / np.maximum(c - 1, 1)
    return m, np.sqrt(np.maximum(var, 0) / np.maximum(c, 1)), c


def land_steps(ctx, ts, norm, root, it, few, many, chunk, grid):
    """E4 phase 1: random direction u, both regimes' update steps and surrogate surfaces (no true-reward rollouts)."""
    alphas = jnp.linspace(0.0, INSTRUMENTS["landscape_max"], grid)
    k_land = common.k_instr(root, it, common.I_LAND)
    u = jax.random.normal(jax.random.fold_in(k_land, 0), ts.pi.shape)
    u = u * INSTRUMENTS["landscape_random_dir_norm"] / jnp.linalg.norm(u)
    out = {"u": u, "alphas": alphas}
    cfg = ctx.cfg
    for name, n_pairs, br in LAND_REGIMES(few, many):
        delta, b, st = policy_step(ctx, ts, norm, root, it, n_pairs, chunk, br)
        S = A.surrogate_grid(ctx.nets, cfg["alg"], cfg.get("clip_eps", 0.2), ts.pi, delta, u, alphas, alphas, b)
        out[name] = dict(S=S, delta=delta, delta_norm=jnp.linalg.norm(delta), u_norm=jnp.linalg.norm(u), step_stats=st)
        if cfg["alg"] == "ppo":     # recheck: the unclipped ratio surrogate (eq. 14) on the same step/sample
            out[name]["S_unclip"] = A.surrogate_grid(ctx.nets, "trpo", 0.0, ts.pi, delta, u, alphas, alphas, b)
        if cfg["alg"] == "ppo":     # the objective as PPO optimised it (per-minibatch adv-norm, the update's partitions)
            _, k_upd = jax.random.split(jax.random.fold_in(k_land, br))
            idx = A._minibatch_perms(k_upd, b.obs.shape[0], cfg["policy_epochs"], cfg["minibatches"])
            out[name]["S_mb"] = A.surrogate_grid(ctx.nets, "ppo", cfg["clip_eps"], ts.pi, delta, u, alphas, alphas, b,
                                                 mb_idx=idx)
    guard({k: (out[k]["S"], out[k]["delta"]) for k in ("few", "many")}, 13)
    return out


def LAND_REGIMES(few, many):
    return (("few", few, 3), ("many", many, 2))


def land_true(ctx, pi, norm, root, it, deltas, u, true_pairs, true_chunk, grid):
    """E4 phase 2: ONE true-reward rollout over both regimes' grids (keys k_land/1/regime -> split(grid^2)).
    deltas [2, d] in regime order (few, many). Returns a dict of true_reward() arrays, each [2, grid^2, n_actors]."""
    alphas = jnp.linspace(0.0, INSTRUMENTS["landscape_max"], grid)
    k_land = common.k_instr(root, it, common.I_LAND)
    Aa, Bb = jnp.meshgrid(alphas, alphas, indexing="ij")
    grids, keys = [], []
    for r, (_, _, br) in enumerate(LAND_REGIMES(0, 0)):
        grids.append(pi[None] + Aa.reshape(-1, 1) * deltas[r][None] + Bb.reshape(-1, 1) * u[None])
        keys.append(jax.random.split(jax.random.fold_in(jax.random.fold_in(k_land, 1), br), grid * grid))
    tr = true_reward(ctx, jnp.concatenate(grids), norm, jnp.concatenate(keys), true_pairs, true_chunk)
    out = {k: x.reshape((2, grid * grid) + x.shape[1:]) for k, x in tr.items()}
    guard(out, 13)
    return out


def landscape(ctx, ts, norm, root, it, few, many, chunk, true_pairs, true_chunk, grid):
    """Both phases in one call (steps, then one true-reward rollout over both 7x7 grids)."""
    out = land_steps(ctx, ts, norm, root, it, few, many, chunk, grid)
    tr = land_true(ctx, ts.pi, norm, root, it, jnp.stack([out["few"]["delta"], out["many"]["delta"]]),
                   out["u"], true_pairs, true_chunk, grid)
    for r, name in enumerate(("few", "many")):
        out[name].update({k: v[r] for k, v in tr.items()})
    return out


# ------------------------------------------------------------------ driver
def summarize_grad(G, g_ref, sizes, n_boot, seed):
    res = {}
    for n in sizes:
        res[str(n)] = A.cos_stats(np.asarray(G[n]), None if g_ref is None else np.asarray(g_ref), n_boot, seed)
    return res


def parse_run_name(run_dir: str):
    """runs/<exp>/<env>_<alg>[-suffix] -> (env, alg). 'ppo-nobaseline' -> 'ppo' (recheck: a suffixed name once
    fell through to the TRPO config silently)."""
    from .configs.default import ALGS, ENVS
    env_name, arm = os.path.basename(run_dir.rstrip("/")).split("_", 1)
    alg = arm.split("-")[0]
    if alg not in ALGS or env_name not in ENVS:
        raise ValueError(f"cannot parse env/alg from run dir {run_dir!r}: got ({env_name}, {alg})")
    return env_name, alg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<exp>/<env>_<alg> (contains ckpt/itNNNN)")
    ap.add_argument("--its", type=int, nargs="+", required=True)
    ap.add_argument("--exps", nargs="+", default=["grad", "value", "baseline", "landscape"])
    ap.add_argument("--grad_sizes", type=int, nargs="+", default=list(INSTRUMENTS["grad_sample_sizes"]))
    ap.add_argument("--base_sizes", type=int, nargs="+", default=list(INSTRUMENTS["baseline_sample_sizes"]))
    ap.add_argument("--m", type=int, default=INSTRUMENTS["grad_n_estimates"])
    ap.add_argument("--ref_pairs", type=int, default=INSTRUMENTS["grad_reference_pairs"])
    ap.add_argument("--chunk", type=int, default=INSTRUMENTS["rollout_chunk"])
    ap.add_argument("--vstar_pairs", type=int, default=INSTRUMENTS["vstar_fit_pairs"])
    ap.add_argument("--land_few", type=int, default=INSTRUMENTS["landscape_few_step_pairs"])
    ap.add_argument("--land_many", type=int, default=INSTRUMENTS["landscape_many_step_pairs"])
    ap.add_argument("--land_true_pairs", type=int, default=INSTRUMENTS["landscape_true_reward_pairs"])
    ap.add_argument("--land_true_chunk", type=int, default=2000)
    ap.add_argument("--land_grid", type=int, default=INSTRUMENTS["landscape_grid"])
    ap.add_argument("--land_phase", default="all", choices=["all", "steps", "true"],
                    help="E4 split: 'steps' (update steps + surrogate surfaces, saved to <tag>_it<k>.npz; CPU) and "
                         "'true' (true-reward rollouts from --land_steps_from files; GPU)")
    ap.add_argument("--land_steps_from", default=None, help="tag of the 'steps' phase files")
    ap.add_argument("--vstar_steps", type=int, default=0, help="override the V* fit length (0 = config)")
    ap.add_argument("--land_no_anneal", action="store_true",
                    help="recheck (labelled arm): the PPO landscape step at the un-annealed Table-1 LR")
    ap.add_argument("--land_point_batch", type=int, default=0, help="grid points per lax.map batch (0 = all vmapped)")
    ap.add_argument("--land_step_chunk", type=int, default=0,
                    help="actor length for the many-sample step's data (0 = --chunk); placement/latency choice")
    ap.add_argument("--ref_from", default=None,
                    help="tag of an earlier run whose saved reference gradients (<tag>_gref_it<k>.npz) are reused "
                         "when --ref_pairs 0 (adds estimate sizes without re-rolling the 1e6-pair reference)")
    ap.add_argument("--n_boot", type=int, default=INSTRUMENTS["bootstrap_resamples"])
    ap.add_argument("--seed_idx", type=int, nargs="*", default=None, help="subset of stacked seeds (indices)")
    ap.add_argument("--land_seed_idx", type=int, nargs="*", default=None,
                    help="run the landscape instrument only on these seed indices (its own agent pass)")
    ap.add_argument("--pool_ablation", action=argparse.BooleanOptionalAction, default=True,
                    help="also n pairs subsampled from the pool (default ON since the review, "
                         "fresh-from-reset sampling is our assumption, so E1 is always reported under both schemes)")
    ap.add_argument("--base_kinds", nargs="+", default=["zero", "zero_raw", "gae", "vstar", "zero_gae"],
                    choices=["zero", "zero_raw", "gae", "vstar", "zero_gae", "vstar_t", "mc_vtheta", "mc_vstar",
                             "mc_vstar_t"],
                    help="E3 baselines; zero = A:=G_t batch-centred (constant baseline), zero_raw = A:=G_t un-centred "
                         "(no baseline at all), zero_gae = GAE with V == 0 ablation); recheck additions "
                         "(labelled, not in the declared set): vstar_t = GAE with the time-aware V*(s, t/2048) "
                         ", mc_* = A := G_t - V(s_t) with V_theta / V* / V*(s,t) ")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="training seeds stacked in the checkpoint, in order (default: the run's seed_<n> dirs)")
    ap.add_argument("--agent_batch", type=int, default=0,
                    help="(checkpoint, seed) agents vmapped per instrument call (0 = all in one call)")
    ap.add_argument("--actor_batch", type=int, default=0,
                    help=">0: roll out actors in lax.map groups of this width (CPU placement; numbers unchanged)")
    ap.add_argument("--tag", default="instruments")
    args = ap.parse_args(argv)

    common.check_device()
    global ACTOR_BATCH, POINT_BATCH, VSTAR_STEPS, LAND_NO_ANNEAL
    LAND_NO_ANNEAL = args.land_no_anneal
    ACTOR_BATCH = args.actor_batch
    POINT_BATCH = args.land_point_batch
    VSTAR_STEPS = args.vstar_steps
    run_dir = os.path.join(common.REPO, args.run)
    env_name, alg = parse_run_name(run_dir)
    ctx = Ctx(env_name, alg, run_dir=run_dir)
    info("env", post_step_kinematics=bool(ctx.cfg["env_post_step_kinematics"]))
    seeds = args.seeds or sorted(int(d.split("_")[1]) for d in os.listdir(run_dir) if d.startswith("seed_"))
    info("instruments", run=args.run, seeds=str(seeds), its=str(args.its), exps=str(args.exps))
    timer = Timer()
    results = {s: {} for s in seeds}
    # The vmap unit is an AGENT = (checkpoint it, training seed): all checkpoints x seeds of a run go through ONE wide
    # rollout batch (GPU throughput saturates only at >~1e4 envs). Every key still derives from (root, it) exactly as
    # in the per-checkpoint runs, so the grouping changes execution only, never the numbers.
    V = lambda f: jax.jit(jax.vmap(f))
    need_t = any(k in ("vstar_t", "mc_vstar_t") for k in args.base_kinds)
    need_vs = any(k in ("vstar", "mc_vstar", "vstar_t", "mc_vstar_t") for k in args.base_kinds)
    j_vstar = V(lambda p, nm, root, it: fit_vstar(ctx, p, nm, root, it, args.vstar_pairs, args.chunk))
    j_vstar_t = V(lambda p, nm, root, it: fit_vstar(ctx, p, nm, root, it, args.vstar_pairs, args.chunk, True))
    j_floor = V(lambda p, v, vs, nm, root, it: value_floor(ctx, p, v, vs, nm, root, it))
    gsz = tuple(args.grad_sizes)
    j_gcol = V(lambda p, nm, root, it: grad_collect(ctx, p, nm, root, it, gsz, args.m, args.chunk, args.ref_pairs))
    # gradients agent by agent (lax.map): a 1e6-pair gradient vmapped over agents does not fit in GPU memory
    j_gcomp = jax.jit(lambda p, v, tr, trr, root, it: jax.lax.map(
        lambda a: grad_compute(ctx, a[0], a[1], a[2], a[3], a[4], a[5], gsz, args.m, args.chunk,
                               pool_ablation=args.pool_ablation), (p, v, tr, trr, root, it)))
    j_base = V(lambda p, v, nm, vs, vst, root, it: gradient_block(ctx, p, v, nm, root, it, tuple(args.base_sizes),
                                                             args.m, args.chunk, 0, kinds=tuple(args.base_kinds),
                                                             vstar=vs, vstar_t=vst,
                                                             raw_ablation=False))
    j_land = V(lambda t, nm, root, it: landscape(ctx, t, nm, root, it, args.land_few, args.land_many,
                                                 args.land_step_chunk or args.chunk,
                                                 args.land_true_pairs, args.land_true_chunk, args.land_grid))
    j_lsteps = V(lambda t, nm, root, it: land_steps(ctx, t, nm, root, it, args.land_few, args.land_many,
                                                    args.land_step_chunk or args.chunk, args.land_grid))
    j_ltrue = V(lambda p, nm, root, it, dl, u: land_true(ctx, p, nm, root, it, dl, u, args.land_true_pairs,
                                                         args.land_true_chunk, args.land_grid))
    pick = lambda tree, i: jax.tree_util.tree_map(lambda x: np.asarray(x[i]), tree)
    st0 = ctx.init_state(jax.random.PRNGKey(0))
    like = {"ts": jax.tree_util.tree_map(lambda x: jnp.stack([x] * len(seeds)), st0),
            "norm": jax.tree_util.tree_map(lambda x: jnp.stack([x] * len(seeds)), D.norm_init(ctx.spec.obs_dim)),
            "it": jnp.int32(0)}
    cks = {it: load_ckpt(os.path.join(run_dir, "ckpt", f"it{it:04d}"), like) for it in args.its}
    sel_all = args.seed_idx if args.seed_idx is not None else list(range(len(seeds)))
    passes = [([(it, si) for it in args.its for si in sel_all], [x for x in args.exps if x != "landscape"]
               if args.land_seed_idx is not None else list(args.exps))]
    if args.land_seed_idx is not None and "landscape" in args.exps:      # landscapes on their own (fewer) seeds
        passes.append(([(it, si) for it in args.its for si in args.land_seed_idx], ["landscape"]))
    work = []
    for agents, exps in passes:
        B = args.agent_batch or len(agents)
        work += [(agents[g0:g0 + B], exps) for g0 in range(0, len(agents), B)]
    for grp, exps in work:
        if exps:
            stack = lambda key: jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                                       *[jax.tree_util.tree_map(lambda x: x[si], cks[it][key])
                                                         for it, si in grp])
            ts, norm = stack("ts"), stack("norm")
            roots = jnp.stack([jax.random.PRNGKey(seeds[si]) for _, si in grp])
            itj = jnp.asarray([it for it, _ in grp], jnp.int32)
            # a real frozen-normaliser check -- a short instrument rollout (update_norm=False) must return the
            # statistics it was given (the old field compared the input with itself and was always True)
            _, norm_after = jax.jit(jax.vmap(lambda p_, n_, r_: ctx.rollout(p_, n_, jax.random.fold_in(r_, 99), 64, False)))(
                ts.pi, norm, jnp.stack([jax.random.PRNGKey(seeds[si]) for _, si in grp]))
            frozen_ok = [bool(all(np.array_equal(np.asarray(a)[j], np.asarray(b)[j]) for a, b in
                                  zip(jax.tree_util.tree_leaves(norm), jax.tree_util.tree_leaves(norm_after))))
                         for j in range(len(grp))]
            tag_it = str(sorted(set(int(i) for i, _ in grp)))
            vstar = vinf = vf = og = ob = ol = vstar_t = vinf_t = None
            if "baseline" in exps and need_t:
                timer.lap()
                vstar, vinf, vstar_t, vinf_t = j_vstar_t(ts.pi, norm, roots, itj)
                jax.block_until_ready(vstar_t)
                info("vstar + vstar_t fit", its=tag_it, agents=len(grp), seconds=round(timer.lap(), 1))
            elif "value" in exps or ("baseline" in exps and need_vs):
                timer.lap()
                vstar, vinf = j_vstar(ts.pi, norm, roots, itj)
                jax.block_until_ready(vstar)
                info("vstar fit", its=tag_it, agents=len(grp), seconds=round(timer.lap(), 1))
            if "value" in exps:
                vf = j_floor(ts.pi, ts.v, vstar, norm, roots, itj)
            if "grad" in exps:
                timer.lap()
                tr, trr = j_gcol(ts.pi, norm, roots, itj)
                jax.block_until_ready(tr)
                info("grad rollouts", its=tag_it, agents=len(grp), seconds=round(timer.lap(), 1))
                og = j_gcomp(ts.pi, ts.v, tr, trr, roots, itj)
                jax.block_until_ready(og)
                del tr, trr
                info("grad compute", its=tag_it, seconds=round(timer.lap(), 1))
            if "baseline" in exps:
                timer.lap()
                ob = j_base(ts.pi, ts.v, norm, vstar, vstar_t, roots, itj)
                jax.block_until_ready(ob)
                info("baseline block", its=tag_it, agents=len(grp), seconds=round(timer.lap(), 1))
            if "landscape" in exps:
                timer.lap()
                if args.land_phase == "all":
                    ol = j_land(ts, norm, roots, itj)
                elif args.land_phase == "steps":
                    ol = j_lsteps(ts, norm, roots, itj)
                else:                                   # 'true': steps/surfaces from the saved phase-1 files
                    zs = [np.load(os.path.join(run_dir, f"seed_{seeds[si]}", f"{args.land_steps_from}_it{it}.npz"))
                          for it, si in grp]
                    deltas = jnp.asarray(np.stack([np.stack([z["delta_few"], z["delta_many"]]) for z in zs]))
                    uu = jnp.asarray(np.stack([z["u"] for z in zs]))
                    trw = j_ltrue(ts.pi, norm, roots, itj, deltas, uu)
                    ol = {"alphas": jnp.stack([jnp.asarray(zs[0]["alphas"])] * len(zs))}
                    for r, name in enumerate(("few", "many")):
                        ol[name] = dict(S=jnp.asarray(np.stack([z[f"S_{name}"] for z in zs])),
                                        **({"S_mb": jnp.asarray(np.stack([z[f"S_mb_{name}"] for z in zs]))}
                                           if f"S_mb_{name}" in zs[0].files else {}),
                                        **({"S_unclip": jnp.asarray(np.stack([z[f"S_unclip_{name}"] for z in zs]))}
                                           if f"S_unclip_{name}" in zs[0].files else {}),
                                        delta_norm=jnp.asarray(np.stack([z[f"delta_norm_{name}"] for z in zs])),
                                        u_norm=jnp.asarray(np.stack([z["u_norm"] for z in zs])),
                                        step_stats={k[len(f"st_{name}_"):]: jnp.asarray(np.stack([z[k] for z in zs]))
                                                    for k in zs[0].files if k.startswith(f"st_{name}_")},
                                        **{k: v[:, r] for k, v in trw.items()})
                jax.block_until_ready(ol)
                info("landscape block", phase=args.land_phase, its=tag_it, agents=len(grp), seconds=round(timer.lap(), 1))
                if args.land_phase == "steps":
                    for j, (it, si) in enumerate(grp):
                        o = pick(ol, j)
                        f = os.path.join(run_dir, f"seed_{seeds[si]}", f"{args.tag}_it{it}.npz")
                        os.makedirs(os.path.dirname(f), exist_ok=True)
                        np.savez(f, u=o["u"], alphas=o["alphas"], u_norm=o["few"]["u_norm"],
                                 **{f"{k}_{nm}": o[nm][kk] for nm in ("few", "many")
                                    for k, kk in (("delta", "delta"), ("S", "S"), ("delta_norm", "delta_norm"))},
                                 **{f"st_{nm}_{k}": v for nm in ("few", "many") for k, v in o[nm]["step_stats"].items()},
                                 **{f"S_mb_{nm}": o[nm]["S_mb"] for nm in ("few", "many") if "S_mb" in o[nm]},
                                 **{f"S_unclip_{nm}": o[nm]["S_unclip"] for nm in ("few", "many") if "S_unclip" in o[nm]})
                    ol = None
            for j, (it, si) in enumerate(grp):
                s = seeds[si]
                root = jax.random.PRNGKey(s)
                r = results[s].setdefault(f"it{it}", {})
                boot_seed = int(jax.random.randint(jax.random.fold_in(jax.random.fold_in(root, common.K_BOOT), it), (), 0, 2**31 - 1))
                if vinf is not None:
                    vi = pick(vinf, j)
                    r["vstar_fit"] = {k: np.asarray(v).tolist() for k, v in vi.items()}
                    info("vstar fit", seed=s, it=it, best_step=int(vi["best_step"]), heldout_r2=float(vi["heldout_r2"]))
                if vinf_t is not None:
                    r["vstar_t_fit"] = {k: np.asarray(v).tolist() for k, v in pick(vinf_t, j).items()}
                    info("vstar_t fit", seed=s, it=it, heldout_r2=float(r["vstar_t_fit"]["heldout_r2"]))
                if vf is not None:
                    r["value_floor"] = {k: float(v) for k, v in pick(vf, j).items()}
                    metric("value floor", seed=s, it=it, **r["value_floor"])
                if og is not None:
                    o = pick(og, j)
                    gpath = os.path.join(run_dir, f"seed_{s}", f"{args.tag}_gref_it{it}.npz")
                    if "g_ref" in o:
                        os.makedirs(os.path.dirname(gpath), exist_ok=True)
                        np.savez(gpath, g_ref=o["g_ref"], g_ref_raw=o["g_ref_raw"])
                    elif args.ref_from:
                        z = np.load(os.path.join(run_dir, f"seed_{s}", f"{args.ref_from}_gref_it{it}.npz"))
                        o["g_ref"], o["g_ref_raw"] = z["g_ref"], z["g_ref_raw"]
                    else:
                        o["g_ref"] = o["g_ref_raw"] = None
                    r["grad"] = summarize_grad(o["G"]["gae"], o["g_ref"], args.grad_sizes, args.n_boot, boot_seed)
                    r["grad_raw_adv"] = summarize_grad(o["G"]["gae_raw"], o["g_ref_raw"], args.grad_sizes, args.n_boot, boot_seed)
                    if "gae_pool" in o["G"]:
                        ns = sorted(o["G"]["gae_pool"])
                        r["grad_pool"] = summarize_grad(o["G"]["gae_pool"], o["g_ref"], ns, args.n_boot, boot_seed)
                    for n in args.grad_sizes:
                        g = r["grad"][str(n)]
                        metric("grad", seed=s, it=it, n=n, pairwise=g["pairwise"], to_true=g.get("to_true"))
                if ob is not None:
                    o = pick(ob, j)
                    r["baseline"] = {kind: summarize_grad(o["G"][kind], None, args.base_sizes, args.n_boot, boot_seed)
                                     for kind in args.base_kinds}
                    for n in args.base_sizes:
                        metric("baseline", seed=s, it=it, n=n, **{k: r["baseline"][k][str(n)]["pairwise"]
                                                                   for k in args.base_kinds})
                if ol is not None:
                    out = pick(ol, j)
                    alphas = np.asarray(out["alphas"])
                    r["landscape"] = {"alphas": alphas.tolist()}
                    for name in ("few", "many"):
                        o = out[name]
                        s00 = float(o["S"][0, 0])
                        if abs(s00) > 1e-5:
                            warn("surrogate S(0,0) != 0", value=s00)
                        G = args.land_grid
                        n_act = np.asarray(o["all_sum"]).shape[1]
                        conc, conc_k, conc_s = {}, {}, {}
                        for pairs in sorted(set([p for p in INSTRUMENTS["landscape_concentration_pairs"]
                                                 if p <= args.land_true_pairs] + [args.land_true_pairs])):
                            na = max(min(pairs // args.land_true_chunk, n_act), 1)
                            # primary : every episode completed within the pair budget
                            m, se, c = reduce_true(o["all_sum"], o["all_sq"], o["all_cnt"], na)
                            used = np.asarray(o["all_pairs"])[:, :na].sum(1)
                            st_ = A.landscape_stats(o["S"], m.reshape(G, G), se.reshape(G, G), alphas)
                            if "S_mb" in o:
                                st_mb = A.landscape_stats(o["S_mb"], m.reshape(G, G), se.reshape(G, G), alphas)
                                st_.update(spearman_Smb_T=st_mb["spearman_S_T"], surr_gain_step_mb=st_mb["surr_gain_step"])
                            if "S_unclip" in o:
                                st_u = A.landscape_stats(o["S_unclip"], m.reshape(G, G), se.reshape(G, G), alphas)
                                st_.update(spearman_Sunclip_T=st_u["spearman_S_T"],
                                           spearman_Sunclip_T_step=st_u["spearman_S_T_step"],
                                           surr_gain_step_unclip=st_u["surr_gain_step"])
                            conc[str(pairs)] = {"T": m.reshape(G, G).tolist(), "T_se": se.reshape(G, G).tolist(),
                                                "n_ep": c.reshape(G, G).tolist(), "pairs_used": used.reshape(G, G).tolist(),
                                                "pairs_rolled": int(na * min(args.land_true_pairs, args.land_true_chunk)),
                                                "n_actors": na, **st_}
                            if "st_sum" in o:
                                ms, ses, cs_ = reduce_true(o["st_sum"], o["st_sq"], o["st_cnt"], na)
                                st_s = A.landscape_stats(o["S"], ms.reshape(G, G), ses.reshape(G, G), alphas)
                                if "S_unclip" in o:
                                    st_su = A.landscape_stats(o["S_unclip"], ms.reshape(G, G), ses.reshape(G, G), alphas)
                                    st_s.update(spearman_Sunclip_T=st_su["spearman_S_T"],
                                                spearman_Sunclip_T_step=st_su["spearman_S_T_step"])
                                conc_s[str(pairs)] = {"T": ms.reshape(G, G).tolist(), "T_se": ses.reshape(G, G).tolist(),
                                                      "n_ep": cs_.reshape(G, G).tolist(), "n_actors": na, **st_s}
                            mk, sek, ck = reduce_true(o["k_sum"], o["k_sq"], o["k_cnt"], na)
                            conc_k[str(pairs)] = {"T": mk.reshape(G, G).tolist(), "T_se": sek.reshape(G, G).tolist(),
                                                  "n_ep": ck.reshape(G, G).tolist(), "n_actors": na,
                                                  **A.landscape_stats(o["S"], mk.reshape(G, G), sek.reshape(G, G), alphas)}
                        stats = conc[str(args.land_true_pairs)]
                        r["landscape"][name] = {"S": np.asarray(o["S"]).tolist(), "by_pairs": conc,
                                                "by_pairs_firstk": conc_k,
                                                **({"by_pairs_start": conc_s} if conc_s else {}),
                                                **({"S_unclip": np.asarray(o["S_unclip"]).tolist()} if "S_unclip" in o else {}),
                                                **({"S_mb": np.asarray(o["S_mb"]).tolist()} if "S_mb" in o else {}),
                                                "S00": s00, "delta_norm": float(o["delta_norm"]), "u_norm": float(o["u_norm"]),
                                                "step_stats": {k: float(v) for k, v in o["step_stats"].items()}}
                        metric("landscape", seed=s, it=it, regime=name, delta_norm=float(o["delta_norm"]),
                               **{k: v for k, v in stats.items() if not isinstance(v, list)},
                               rough_by_pairs=str({p: round(v["roughness_rel_se"], 2) for p, v in conc.items()}))
                r["normaliser_frozen_ok"] = frozen_ok[j]
                if not frozen_ok[j]:
                    warn("instrument rollout changed the normaliser statistics", seed=s, it=it)
    for s in seeds:
        if results[s]:
            write_json(os.path.join(run_dir, f"seed_{s}", f"{args.tag}.json"),
                       {"env": env_name, "alg": alg, "seed": s, "args": vars(args), **common.git_info(), **results[s]})
    info("done")


if __name__ == "__main__":
    main()
