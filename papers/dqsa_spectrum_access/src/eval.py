"""Evaluation: frozen-network test rollouts (Eq. 11, alpha = 0, beta = 20, SAMPLED --), baseline rollouts on the SAME
worlds, per-experiment reducers, and the paper's metrics (README).

Metric definitions:
  util_t   : fraction of channels with exactly one SU transmitter (and PU free) at slot t   (Fig. 3 "channel throughput")
  rate_t   : mean over active users of delivered Mbps at slot t                              (Figs. 4, 6)
  lograte_t: per experiment max(-5, mean_n ln(cumulative Mbit_n(t))) (-inf if any user at zero) -- ORDINAL ONLY
  pareto   : last 50 slots with zero collisions and zero idle channels (strict) / >= 0.95 successful (tolerant).  CAVEAT :
             `pareto_strict` / `pareto_tol` demand K successes per slot, which is infeasible when the users' random 2-channel subsets leave a
             channel uncovered or a PU blocks one (E2/E3) -- it is meaningful only for the subset == "all" cases E1/E6-E8.  The
             feasibility-relative versions are `pareto_feas_strict` / `pareto_feas_tol` (successes == maximum matching size of active users to
             PU-free channels they may use, data.max_feasible) and `util_bound_t` = feasible_t / K is the per-slot achievable utilization bound.
Eval worlds: world keys come from a CONSTANT eval root (identical for every arm and training seed); action keys additionally fold in
the replicate index so arms are paired by replicate.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import algorithm as alg  # noqa: E402
import data as D  # noqa: E402
import model as M  # noqa: E402

EVAL_ROOT_SEED = 20261020
TEST_ALPHA, TEST_BETA = 0.0, 20.0     # (asserted below)
PARETO_WINDOW = 50                    #
LOG_FLOOR = math.log(0.01)            # paper Fig. 5 curves start at ~-4.6 (rendered p.11: above the -5 axis line) = ln(0.01),
                                      # i.e. log(max(x, 0.01)).  (Was -5.0 before the review.)  Curve index i is the state AFTER slot i+1 (paper t = 1..).


def eval_keys(n_exp: int, replicate: int, offset: int = 0):
    root = jax.random.PRNGKey(EVAL_ROOT_SEED)
    wk = D.world_keys(jax.random.fold_in(root, 0), offset + n_exp)[offset:]
    ak = D.world_keys(jax.random.fold_in(jax.random.fold_in(root, 1), replicate), offset + n_exp)[offset:]
    return wk, ak


def jain(x, active):
    x = jnp.where(active, x, 0.0)
    n = active.sum()
    return jnp.where(x.sum() > 0, x.sum() ** 2 / (n * (x ** 2).sum()), 0.0)


# (recheck): the log-rate metric has a FREE unit (slot_s ASSUMED) and a free base.  Because a floored experiment
# is set to the constant ln(0.01) while an unfloored one shifts with the unit, an ORDER between two arms with different floored fractions is
# not unit-invariant.  `LR_VARIANTS` reports the same per-experiment metric under every declared (unit, base) convention, floor = log_base(0.01)
# (the paper's axis constant); H3 is graded only if the t = 100 ordering is the same under ALL of them (README).
LR_VARIANTS = {            # name: (unit scale applied to cumulative Mbit-equivalents, log base)
    "ln_mbit": (1.0, "ln"), "ln_bit": (1.0e6, "ln"), "ln_mbpssum": (None, "ln"),
    "log2_mbit": (1.0, "log2"), "log2_bit": (1.0e6, "log2"), "db_mbit": (1.0, "db"), "db_bit": (1.0e6, "db"),
}
_LOGF = {"ln": lambda x: jnp.log(x), "log2": lambda x: jnp.log2(x), "db": lambda x: 10.0 * jnp.log10(x)}


def lr_variants(spec: D.EnvSpec, rate, active, n):
    """rate [T,N] Mbps -> {name: [T]} per-experiment mean-over-users log of cumulative delivered amount, any-zero-user -> floor."""
    out = {}
    for name, (scale, base) in LR_VARIANTS.items():
        cum = jnp.cumsum(rate, 0) * (spec.slot_s * scale if scale is not None else 1.0)   # ln_mbpssum: sum over slots of Mbps (no slot_s)
        f = _LOGF[base]
        floor = f(jnp.asarray(0.01, jnp.float32))
        lg = jnp.where(cum > 0, f(jnp.maximum(cum, 1e-30)), -jnp.inf)
        v = jnp.maximum(floor, jnp.where(active, lg, 0.0).sum(1) / n)
        out["lrv_" + name] = jnp.where(jnp.any(active[None] & jnp.isinf(lg), axis=1), floor, v)
    return out


def reduce_one(spec: D.EnvSpec, o: dict):
    """Reduce one experiment's rollout (success/tx/g/rate [T,N]) to curves and scalars."""
    K, W = spec.n_channels, PARETO_WINDOW
    s, tx, g, rate, active, n = o["success"].astype(jnp.float32), o["tx"], o["g"], o["rate"], o["active"], o["n_active"]
    counts = (jax.nn.one_hot(g, K) * tx[..., None]).sum(1)                       # [T,K] SU transmitters per channel
    util_t = s.sum(1) / K
    coll_t = (counts >= 2).mean(1)
    idle_t = (counts == 0).mean(1)
    rate_t = rate.sum(1) / n
    cum = jnp.cumsum(rate, 0) * spec.slot_s                                       # Mbit (Mbps x s)
    lg = jnp.where(cum > 0, jnp.log(jnp.maximum(cum, 1e-30)), -jnp.inf)
    lr_strict = jnp.maximum(LOG_FLOOR, jnp.where(active, lg, 0.0).sum(1) / n)
    lr_strict = jnp.where(jnp.any(active[None] & jnp.isinf(lg), axis=1), LOG_FLOOR, lr_strict)
    lr_user = jnp.where(active, jnp.maximum(lg, LOG_FLOOR), 0.0).sum(1) / n       # per-user floored variant
    last = s[-W:]
    sw = last.sum(1)
    if spec.pu:
        feas_t = jax.vmap(lambda pf: D.max_feasible(spec, D.World(active, n, o["subset"]), pf))(o["pu_free"]).astype(jnp.float32)
    else:
        feas_t = jnp.broadcast_to(D.max_feasible(spec, D.World(active, n, o["subset"]), jnp.ones((K,), bool)).astype(jnp.float32), s.shape[:1])
    fl = feas_t[-W:]
    return dict(
        feasible_t=feas_t, util_bound_t=feas_t / K,
        pareto_feas_strict=jnp.all(sw == fl), pareto_feas_tol=(sw.sum() >= 0.95 * fl.sum()) & (fl.sum() > 0),
        util_t=util_t, coll_t=coll_t, idle_t=idle_t, rate_t=rate_t, lr_strict_t=lr_strict, lr_user_t=lr_user,
        pareto_strict=jnp.all(sw == K), pareto_tol=(sw.mean() / K) >= 0.95,
        succ_frac_last=jnp.where(active, last.mean(0), 0.0), tx_frac_last=jnp.where(active, tx[-W:].mean(0), 0.0),
        succ_tot=s.sum(0), jain_all=jain(s.sum(0), active), jain_last=jain(last.sum(0), active), n_active=n,
        active=active, zero_user_end=jnp.any(active & (cum[-1] <= 0)), **lr_variants(spec, rate, active, n))


_ROLL_KEYS = ("success", "tx", "g", "rate", "active", "n_active", "subset", "pu_free")


@lru_cache(maxsize=None)
def _learned_fn(spec: D.EnvSpec, T: int, reward: str, dueling: bool, lstm: int, hid: int, greedy: bool, beta: float = TEST_BETA):
    net = M.make_net(spec.n_actions, lstm, hid, dueling)

    @jax.jit
    def fn(params, wk, ak):
        out = alg.rollout_learned(spec, net, params, wk, ak, T, TEST_ALPHA, beta, reward, greedy)
        return jax.vmap(lambda o: reduce_one(spec, o))({k: out[k] for k in _ROLL_KEYS})
    return fn


@lru_cache(maxsize=None)
def _policy_fn(spec: D.EnvSpec, T: int, name: str, kw: tuple):
    policy = alg.BASELINES[name](**dict(kw))

    @jax.jit
    def fn(wk, ak):
        out = alg.rollout_policy(spec, policy, wk, ak, T)
        return jax.vmap(lambda o: reduce_one(spec, o))(out)
    return fn


def _run_chunks(fn, n_exp, replicate, chunk, *args, world_offset=0):
    parts = []
    for off in range(world_offset, world_offset + n_exp, chunk):
        m = min(chunk, world_offset + n_exp - off)
        wk, ak = eval_keys(m, replicate, off)
        parts.append(jax.tree.map(np.asarray, fn(*args, wk, ak)))
    return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}


MONITOR_WORLD_OFFSET = 100_000     # monitor / tuning / patience evaluations read worlds >= this index, disjoint from the reporting worlds 0..n_exp-1


def eval_learned(spec, tc: alg.TrainCfg, params, n_exp=1000, T=200, replicate=0, chunk=100, greedy=False, world_offset=0):
    assert TEST_ALPHA == 0.0 and TEST_BETA == 20.0            # README: test policy = Eq. 11 sampled, alpha=0, beta=20
    beta = TEST_BETA / tc.reward_scale if tc.rescale_beta else TEST_BETA        # policy-invariant under a reward rescaling
    fn = _learned_fn(spec, T, tc.reward, tc.dueling, tc.lstm, tc.hid, greedy, beta)
    return _run_chunks(fn, n_exp, replicate, chunk, params, world_offset=world_offset)


def eval_baseline(spec, name, n_exp=1000, T=200, replicate=0, chunk=100, world_offset=0, **kw):
    """world_offset > 0 selects eval worlds disjoint from the reporting worlds 0..n_exp-1 (use >= 100000 for tuning)."""
    fn = _policy_fn(spec, T, name, tuple(sorted(kw.items())))
    return _run_chunks(fn, n_exp, replicate, chunk, world_offset=world_offset)


# ----------------------------------------------------------------------------------------------- summaries
def summarize(spec: D.EnvSpec, r: dict, window=(60, 200), kind: str | None = None) -> dict:
    """Experiment-mean curves + scalars from the stacked reducer output (numpy)."""
    lo, hi = window
    out = {k.replace("_t", "_curve"): r[k].mean(0).tolist() for k in ("util_t", "coll_t", "idle_t", "rate_t", "lr_strict_t", "lr_user_t")}
    out["util_bound_curve"] = r["util_bound_t"].mean(0).tolist()
    fsum = float(r["feasible_t"][:, lo:hi].sum())
    out.update(
        util_bound_plateau=float(r["util_bound_t"][:, lo:hi].mean()),
        efficiency_plateau=float(r["util_t"][:, lo:hi].sum() * spec.n_channels / fsum) if fsum > 0 else 0.0,   # successes / feasible successes
        pareto_feas_strict=float(r["pareto_feas_strict"].mean()), pareto_feas_tol=float(r["pareto_feas_tol"].mean()),
        plateau_util=float(r["util_t"][:, lo:hi].mean()), plateau_coll=float(r["coll_t"][:, lo:hi].mean()),
        plateau_idle=float(r["idle_t"][:, lo:hi].mean()), plateau_rate=float(r["rate_t"][:, lo:hi].mean()),
        pareto_strict=float(r["pareto_strict"].mean()), pareto_tol=float(r["pareto_tol"].mean()),
        jain_all=float(r["jain_all"].mean()), jain_last=float(r["jain_last"].mean()),
        lograte_final=float(r["lr_strict_t"][:, -1].mean()), lograte_user_final=float(r["lr_user_t"][:, -1].mean()),
        frac_zero_user_end=float(r["zero_user_end"].mean()),
        n_exp=int(r["util_t"].shape[0]), T=int(r["util_t"].shape[1]), window=list(window))
    if kind:
        out.update(patterns(kind, spec, r))
    return out


def patterns(kind: str, spec: D.EnvSpec, r: dict) -> dict:
    """Operational 'Pareto-optimal sharing' tests of the small cases."""
    K = spec.n_channels
    strict = r["pareto_strict"].astype(bool)
    sf, tf = r["succ_frac_last"], r["tx_frac_last"]
    act = r["active"].astype(bool)
    out = {}
    if kind == "E6":      # sum rate: K users always transmit, the others always silent, throughput > 0.9
        always = ((tf >= 0.99) & act).sum(1)
        silent = ((tf <= 0.01) & act).sum(1)
        out["pattern_E6"] = float((strict & (always == K) & (silent == act.sum(1) - K)).mean())
        out["pattern_E6_tol"] = float((r["pareto_tol"].astype(bool) & (always == K) & (silent == act.sum(1) - K)).mean())   # reported, not graded
        out["throughput_gt_0.9"] = float((r["util_t"][:, -PARETO_WINDOW:].mean(1) > 0.9).mean())
    elif kind == "E7":    # competitive: one user alone on a channel, the other two alternate ~50/50 on the other
        srt = np.sort(np.where(act, sf, -1.0), axis=1)[:, ::-1]
        ok = (srt[:, 0] >= 0.99) & (srt[:, 1] >= 0.4) & (srt[:, 1] <= 0.6) & (srt[:, 2] >= 0.4) & (srt[:, 2] <= 0.6)
        out["pattern_E7"] = float((strict & ok).mean())
        out["pattern_E7_tol"] = float((r["pareto_tol"].astype(bool) & ok).mean())                  # reported, not graded
    elif kind == "E8":    # PF: near-equal sharing (Jain >= 0.9 over the test episode), throughput ~0.6
        out["pattern_E8_jain09"] = float((r["jain_all"] >= 0.9).mean())
    return out


# ----------------------------------------------------------------------------------------------- analytic references
def aloha_expected(n):
    """Paper §VI-B: expected clique throughput with optimal p = 1/n:  (1-1/n)^(n-1)."""
    return (1.0 - 1.0 / n) ** (n - 1)


def aloha_expected_mean(n_min=3, n_max=11):
    return float(np.mean([aloha_expected(n) for n in range(n_min, n_max + 1)]))   # 0.403308 for 3..11


def aloha_multichannel(N, K):
    """Per-channel throughput at p = min(1, K/N), uniform channel:  (Np/K)(1-p/K)^(N-1)."""
    p = min(1.0, K / N)
    return N * p / K * (1 - p / K) ** (N - 1)
