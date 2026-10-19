"""Fixed policies: the paper's three baselines (Static, Random, Coverage, paper §3), the public hand rule
`raster+revisit` (added by this reproduction) and the privileged truth-cued ceiling.

Policy interface (shared with the learned policies in eval.py):
    pinit(key) -> pstate;  policy(pstate, s: EnvState, obs: Obs, key) -> (action int32[2], pstate)
`s` is the full environment state. Public policies read only the tracker part of it (s.tr, s.t), never s.truth;
the ceiling reads s.truth and is therefore a ceiling only, never a teacher.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from src.configs.config import EnvConfig
from src.data import D2R, action_centres_deg


def nearest_cell(az_deg, el_deg, cfg: EnvConfig):
    pitch = 2 * cfg.for_half_deg / (cfg.n_a - 1)
    ia = jnp.clip(jnp.round((az_deg + cfg.for_half_deg) / pitch), 0, cfg.n_a - 1)
    ie = jnp.clip(jnp.round((el_deg + cfg.for_half_deg) / pitch), 0, cfg.n_a - 1)
    return jnp.stack([ia, ie]).astype(jnp.int32)


def direction_deg(x):
    """(az, el) in degrees of state(s) x [..., 6] (position part)."""
    px, py, pz = x[..., 0], x[..., 2], x[..., 4]
    return jnp.arctan2(py, px) / D2R, jnp.arctan2(pz, jnp.sqrt(px ** 2 + py ** 2)) / D2R


def raster_action(r, cfg: EnvConfig):
    """Row-major raster: az fastest, el lines bottom to top, from cell (0, 0)."""
    n = cfg.n_a
    return jnp.stack([r % n, (r // n) % n]).astype(jnp.int32)


def boustrophedon_action(r, cfg: EnvConfig):
    n = cfg.n_a
    row = (r // n) % n
    col = r % n
    return jnp.stack([jnp.where(row % 2 == 0, col, n - 1 - col), row]).astype(jnp.int32)


def make_static(cfg: EnvConfig):
    def pinit(key):
        return jax.random.randint(key, (2,), 0, cfg.n_a).astype(jnp.int32)

    def policy(p, s, obs, key):
        return p, p
    return pinit, policy


def make_random(cfg: EnvConfig):
    def pinit(key):
        return jnp.int32(0)

    def policy(p, s, obs, key):
        return jax.random.randint(key, (2,), 0, cfg.n_a).astype(jnp.int32), p
    return pinit, policy


def make_coverage(cfg: EnvConfig, boustrophedon: bool = False):
    f = boustrophedon_action if boustrophedon else raster_action

    def pinit(key):
        return jnp.int32(0)

    def policy(r, s, obs, key):
        return f(r, cfg), r + 1
    return pinit, policy


def make_hand_rule(cfg: EnvConfig, k_rev, p_rev):
    """raster+revisit: continue the raster unless a live track is overdue (steps since its last update >= k_rev
    at the coming look, or trace P >= p_rev); then look at the cell nearest the predicted direction of the most overdue
    track (largest age, ties -> largest trace). The raster pointer pauses during revisits. Reads the tracker only."""
    def pinit(key):
        return jnp.int32(0)

    def policy(r, s, obs, key):
        tr = s.tr
        age = (s.t + 1 - tr.last_upd).astype(jnp.float64)
        trace = jnp.trace(tr.P, axis1=1, axis2=2)
        overdue = tr.valid & ((age >= k_rev) | (trace >= p_rev))
        score = jnp.where(overdue, age * 1e6 + trace, -jnp.inf)
        j = jnp.argmax(score)
        xp = tr.x[j] + jnp.array([tr.x[j, 1], 0, tr.x[j, 3], 0, tr.x[j, 5], 0]) * cfg.dt
        az, el = direction_deg(xp)
        revisit = jnp.any(overdue)
        a = jnp.where(revisit, nearest_cell(az, el, cfg), raster_action(r, cfg))
        return a, jnp.where(revisit, r, r + 1)
    return pinit, policy


def make_ceiling(cfg: EnvConfig, match_m: float = 1000.0):
    """truth-cued: look at the true (CV-predicted) direction of the reachable target whose track has the largest
    trace P; a target with no live track within match_m metres (position) counts as +inf; ties (several untracked) go to
    the target least recently inside the beam. Targets out of beam reach of every cell are skipped (raster if none).
    match_m = 1 km is ASSUMED ( leaves the truth-to-track matching unstated)."""
    cen = action_centres_deg(cfg)

    def pinit(key):
        return jnp.int32(0)

    def policy(r, s, obs, key):
        tr = s.tr
        tp = s.truth[:, [0, 2, 4]]
        xp = s.truth + jnp.stack([s.truth[:, 1], jnp.zeros(cfg.n_tgt), s.truth[:, 3], jnp.zeros(cfg.n_tgt),
                                  s.truth[:, 5], jnp.zeros(cfg.n_tgt)], -1) * cfg.dt
        az, el = direction_deg(xp)
        d = jnp.linalg.norm(tp[:, None, :] - tr.x[None, :, [0, 2, 4]], axis=-1)          # [n, K]
        d = jnp.where(tr.valid[None, :], d, jnp.inf)
        trace = jnp.trace(tr.P, axis1=1, axis2=2)
        j = jnp.argmin(d, axis=1)
        tracked = jnp.min(d, axis=1) < match_m
        cells = jax.vmap(lambda a, e: nearest_cell(a, e, cfg))(az, el)                  # [n, 2]
        # reachable: within the beam radius of the chosen cell centre (great circle)
        ca, ce = cen[cells[:, 0]] * D2R, cen[cells[:, 1]] * D2R
        ub = jnp.stack([jnp.cos(ce) * jnp.cos(ca), jnp.cos(ce) * jnp.sin(ca), jnp.sin(ce)], -1)
        ut = xp[:, [0, 2, 4]] / jnp.linalg.norm(xp[:, [0, 2, 4]], axis=-1, keepdims=True)
        reach = jnp.sum(ub * ut, -1) >= jnp.cos(cfg.beam_r_deg * D2R)
        prio = jnp.where(tracked, trace[j], 1e12 - s.last_look.astype(jnp.float64))
        prio = jnp.where(reach, prio, -jnp.inf)
        i = jnp.argmax(prio)
        a = jnp.where(jnp.any(reach), cells[i], raster_action(r, cfg))
        return a, r + 1
    return pinit, policy
