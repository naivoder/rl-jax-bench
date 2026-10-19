"""Evaluation: run any policy (baseline, hand rule, ceiling or learned) for one full episode on each of a set of
scenarios (vmapped), and reduce the per-step quantities to the per-episode metrics of README:

  SR      per-step mean of r_SV (and its episode sum)
  M_cov   mean over track-bearing steps of sum_tracks ||P||_F (Stone Soup SumofCovarianceNorms); NaN if the
          episode never holds a track (excluded and counted). Reading (i) 'live': the tracks alive after the step.
  M_cov_lib  DESCRIPTIVE, reading (ii) 'library' (official-code phase, fixture P9): the Stone Soup tutorial pipeline
          accumulates every track ever produced, so a track deleted at step t also contributes its last (predicted,
          trace > threshold) state at t. Equal to Stone Soup's MultiManager + SumofCovarianceNormsMetric to 1e-12.
  GOSPA   per-episode sums of distance / localisation / missed / false (p = 2, c = 10 km, alpha = 2, full state)
  diagnostics: per-track mean ||P||_F, mean live tracks, initiations, deletions, lifetime histogram, returns under
          both r_TL signs, look-concentration (central cells), detections.

The common-random-number design: the scenario key drives truth, measurement noise and P_D draws; the policy key drives
only the policy's own randomness.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.configs.config import EnvConfig
from src import data


def rollout_eval(pinit, policy, scen_key, pol_key, cfg: EnvConfig):
    """One episode of T steps. Returns a dict of per-episode scalars (+ the lifetime histogram)."""
    s = data.reset(scen_key, cfg)
    obs = data.observe(s, cfg)
    p = pinit(jax.random.fold_in(pol_key, 0xFFFF))
    c = cfg.n_a // 2

    def body(carry, i):
        s, obs, p = carry
        a, p = policy(p, s, obs, jax.random.fold_in(pol_key, i))
        s, obs, info = data.step(s, a, cfg)
        info = dict(info)
        info["central1"] = ((a[0] == c) & (a[1] == c)).astype(jnp.float64)
        info["central3"] = ((jnp.abs(a[0] - c) <= 1) & (jnp.abs(a[1] - c) <= 1)).astype(jnp.float64)
        return (s, obs, p), info

    (sT, _, _), info = jax.lax.scan(body, (s, obs, p), jnp.arange(cfg.T))
    has = info["n_tracks"] > 0
    n_has = jnp.sum(has)
    has_lib = (info["n_tracks"] + info["n_killed"]) > 0
    n_lib = jnp.sum(has_lib)
    out = dict(
        sr=jnp.mean(info["r_sv"]), sr_sum=jnp.sum(info["r_sv"]),
        m_cov=jnp.where(n_has > 0, jnp.sum(jnp.where(has, info["cov_sum"], 0.0)) / jnp.maximum(n_has, 1), jnp.nan),
        m_cov_lib=jnp.where(n_lib > 0, jnp.sum(jnp.where(has_lib, info["cov_sum"] + info["cov_killed"], 0.0))
                            / jnp.maximum(n_lib, 1), jnp.nan),
        cov_per_track=jnp.sum(info["cov_sum"]) / jnp.maximum(jnp.sum(info["n_tracks"]), 1),
        frac_steps_tracked=n_has / cfg.T,
        mean_tracks=jnp.mean(info["n_tracks"].astype(jnp.float64)),
        ret_flip=jnp.sum(info["r_sv"] + info["rtl_flip"]), ret_lit=jnp.sum(info["r_sv"] - info["rtl_flip"]),
        rtl_flip_sum=jnp.sum(info["rtl_flip"]),
        n_det=jnp.sum(info["n_det"]), n_hit=jnp.sum(info["n_hit"]),
        n_init=sT.n_init, n_del=sT.n_del, overflow=sT.overflow,
        central1=jnp.mean(info["central1"]), central3=jnp.mean(info["central3"]),
        min_S_minor=jnp.min(info["min_S_minor"]),
        life_hist=data.final_life_hist(sT),
    )
    if cfg.gospa_metrics:
        out.update(gospa_d=jnp.sum(info["gospa_d"]), gospa_loc=jnp.sum(info["gospa_loc"]),
                   gospa_miss=jnp.sum(info["gospa_miss"]), gospa_false=jnp.sum(info["gospa_false"]))
        if cfg.gospa_lib:
            out.update(gospa_lib_d=jnp.sum(info["gospa_lib_d"]), gospa_lib_loc=jnp.sum(info["gospa_lib_loc"]),
                       gospa_lib_miss=jnp.sum(info["gospa_lib_miss"]), gospa_lib_false=jnp.sum(info["gospa_lib_false"]),
                       lib_kill_ovf=jnp.sum(info["lib_kill_ovf"]))
    if cfg.n_fa_slots > 0:
        out.update(n_fa=jnp.sum(info["n_fa"]))
    return out


def eval_many(pinit, policy, scen_keys, pol_keys, cfg: EnvConfig):
    """vmap over scenarios (pol_keys aligned with scen_keys)."""
    return jax.vmap(lambda sk, pk: rollout_eval(pinit, policy, sk, pk, cfg))(scen_keys, pol_keys)


def scenario_keys(root_key, tag: int, n: int):
    base = jax.random.fold_in(root_key, tag)
    return jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.arange(n))


def make_learned(net, params, deterministic: bool = False):
    """Policy from an ActorCritic and its params: sample (or argmax) each factor."""
    from src.algorithm import sample_action

    def pinit(key):
        return jnp.int32(0)

    def policy(p, s, obs, key):
        la, le, _ = net.apply(params, obs)
        return sample_action(key, la, le, deterministic), p
    return pinit, policy


def life_median(hist) -> float:
    """Median of the lifetime (updates after initiation) distribution from a histogram summed over episodes."""
    h = np.asarray(hist).reshape(-1, data.LIFE_BINS).sum(0)
    if h.sum() == 0:
        return float("nan")
    cdf = np.cumsum(h) / h.sum()
    return float(np.searchsorted(cdf, 0.5))


M_COV_PENALTY = 5000.0   # ASSUMED ( reading 'pen'): a step without any track counts at the deletion threshold, the
                         # largest covariance statistic the tracker admits before it gives a track up


def derived_episode_metrics(pe: dict, T: int, n_tgt: int, gospa_c: float) -> dict:
    """Readings added after the review, computed from stored per-episode values (no re-run needed):
      m_cov_zero   per-step mean of sum ||P||_F with no-track steps counted as 0 (never NaN)
      m_cov_pen    the same with no-track steps counted at M_COV_PENALTY (constructed; makes 'no track' expensive)
      unassigned_tracks_per_step  GOSPA-unassigned live tracks per step = sum false / (c^2/2) / T (duplicates and
                  tracks farther than c from every target); assigned_tracks_per_step likewise from sum missed
      dup_frac     unassigned / (assigned + unassigned) track-steps
    """
    frac = np.asarray(pe["frac_steps_tracked"], float)
    m = np.asarray(pe["m_cov"], float)
    mz = np.where(np.isfinite(m), m * frac, 0.0)
    res = dict(m_cov_zero=mz, m_cov_pen=mz + (1.0 - frac) * M_COV_PENALTY)
    if "gospa_false" in pe:
        h = gospa_c ** 2 / 2.0
        un = np.asarray(pe["gospa_false"], float) / h / T
        asg = n_tgt - np.asarray(pe["gospa_miss"], float) / h / T
        res.update(unassigned_tracks_per_step=un, assigned_tracks_per_step=asg,
                   dup_frac=np.where(un + asg > 0, un / np.maximum(un + asg, 1e-300), np.nan))
    return res


def reduce_episodes(out: dict) -> dict:
    """Per-arm summary over episodes: means (M_cov over episodes that held a track) and episode SDs."""
    res = {}
    for k, v in out.items():
        v = np.asarray(v)
        if k == "life_hist":
            res["life_median"] = life_median(v)
            res["n_tracks_total"] = int(v.sum())
            continue
        if k == "min_S_minor":
            res[k] = float(np.min(v))
            continue
        fin = v[np.isfinite(v)]
        res[k] = float(fin.mean()) if fin.size else float("nan")
        res[k + "_sd"] = float(fin.std(ddof=1)) if fin.size > 1 else float("nan")
        if k == "m_cov":
            res["n_episodes_no_track"] = int((~np.isfinite(v)).sum())
    return res
