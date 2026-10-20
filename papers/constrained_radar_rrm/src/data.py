"""Scenario generation (the paper has no dataset): target tables for the random spawn model and the digitised
test schedules. A scenario is a fixed-size table of M targets whose truth is a pure function of the slot index
(noise-free CV lines, A-03, or the Case III reflecting arc, A-37), so presence and position need no state.
"""
from __future__ import annotations

import json
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .common import CFG_DIR, EnvCfg


class Scenario(NamedTuple):
    p0: jax.Array        # [M, 2] position at birth (m)
    v: jax.Array         # [M, 2] velocity (m/s)
    born: jax.Array      # [M] int32 slot of birth
    death: jax.Array     # [M] int32 first slot of absence (birth + lifetime or schedule)
    valid: jax.Array     # [M] bool
    circ: jax.Array      # [M] bool: reflecting-arc target (Case III)
    circ_r: jax.Array    # [M] radius (m)
    circ_lo: jax.Array   # [M] arc start (rad)
    circ_span: jax.Array  # [M] arc span (rad)
    circ_step: jax.Array  # [M] angle increment per slot (rad) = omega T0
    overflow: jax.Array  # [] int32: spawned targets dropped for lack of table space (logged)


def _empty(M: int) -> dict:
    z = np.zeros
    return dict(p0=z((M, 2)), v=z((M, 2)), born=z(M, np.int32), death=z(M, np.int32), valid=z(M, bool),
                circ=z(M, bool), circ_r=z(M), circ_lo=z(M), circ_span=z(M), circ_step=z(M))


def _to_scen(d: dict) -> Scenario:
    return Scenario(**{k: jnp.asarray(v) for k, v in d.items()}, overflow=jnp.zeros((), jnp.int32))


def truth(scen: Scenario, t, cfg: EnvCfg):
    """Truth positions [M, 2] and presence [M] at slot t (presence = alive, born, inside the 20-km area; A-15)."""
    dt = (t - scen.born).astype(jnp.float64)
    lin = scen.p0 + scen.v * (dt * cfg.T0)[:, None]
    phase = scen.circ_step * dt
    u = jnp.mod(phase, 2 * scen.circ_span)
    a = jnp.where(u <= scen.circ_span, u, 2 * scen.circ_span - u)
    ang = scen.circ_lo + a
    arc = scen.circ_r[:, None] * jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
    pos = jnp.where(scen.circ[:, None], arc, lin)
    r = jnp.hypot(pos[:, 0], pos[:, 1])
    present = scen.valid & (t >= scen.born) & (t < scen.death) & (r <= cfg.area_radius)
    return pos, present


# ----------------------------------------------------------------------------- random spawn model (§VII-A2, A-16/A-17)


def spawn_scenario(key, cfg: EnvCfg, n_slots: int, n0) -> Scenario:
    """n0 targets at slot 0, then every `spawn_every` slots one arrival w.p. spawn_prob. Range U(5,15) km, bearing
    U(0,2pi), speed U(0.5,2) m/s, heading U(0,2pi) (A-16). Arrivals are compacted into the first M table rows."""
    M = cfg.M
    n0max = 3
    n_checks = -(-n_slots // cfg.spawn_every)
    C = max(n0max + n_checks, M)                       # pad with never-arriving candidates if needed
    n_pad = C - n0max - n_checks
    k_arr, k_r, k_b, k_s, k_h = jax.random.split(key, 5)
    born = jnp.concatenate([jnp.zeros(n0max, jnp.int32), jnp.arange(n_checks, dtype=jnp.int32) * cfg.spawn_every,
                            jnp.zeros(n_pad, jnp.int32)])
    arrive = jnp.concatenate([jnp.arange(n0max) < n0, jax.random.uniform(k_arr, (n_checks,)) < cfg.spawn_prob,
                              jnp.zeros(n_pad, bool)])
    rng = jax.random.uniform(k_r, (C,), minval=cfg.spawn_range[0], maxval=cfg.spawn_range[1])
    brg = jax.random.uniform(k_b, (C,), maxval=2 * jnp.pi)
    spd = jax.random.uniform(k_s, (C,), minval=cfg.spawn_speed[0], maxval=cfg.spawn_speed[1])
    hdg = jax.random.uniform(k_h, (C,), maxval=2 * jnp.pi)
    order = jnp.argsort(jnp.where(arrive, born, 10 ** 9), stable=True)[:M]
    valid = arrive[order]
    p0 = jnp.stack([rng * jnp.cos(brg), rng * jnp.sin(brg)], -1)[order]
    v = jnp.stack([spd * jnp.cos(hdg), spd * jnp.sin(hdg)], -1)[order]
    b = born[order]
    zM = jnp.zeros(M)
    return Scenario(p0=p0, v=v, born=b, death=b + cfg.lifetime, valid=valid, circ=jnp.zeros(M, bool), circ_r=zM,
                    circ_lo=zM, circ_span=zM, circ_step=zM,
                    overflow=jnp.maximum(arrive.sum() - M, 0).astype(jnp.int32))


# ----------------------------------------------------------------------------- digitised test schedules


# Replica schedule file. scenario_case1.json = the re-evaluation's re-digitisation; scenario_case1_v1.json =
# the schedule behind every full-run number (kept so those numbers stay reproducible; scripts select it with --replica v1).
REPLICA_FILES = {"v2": "scenario_case1.json", "v1": "scenario_case1_v1.json",
                 "v2o": "scenario_case1_v2_overflight.json"}   # v2o: Target 1b overflies the radar (sensitivity)
REPLICA = "v2"


def replica_case1(cfg: EnvCfg, which: str | None = None) -> Scenario:
    """Case I / II replica (A-40): straight segments start->end over [born, removed)."""
    with open(os.path.join(CFG_DIR, REPLICA_FILES[which or REPLICA])) as f:
        sched = json.load(f)["targets"]
    d = _empty(cfg.M)
    for i, s in enumerate(sched):
        n = s["removed"] - s["born"]
        d["p0"][i] = s["start"]
        d["v"][i] = (np.array(s["end"]) - np.array(s["start"])) / (n * cfg.T0)
        d["born"][i], d["death"][i], d["valid"][i] = s["born"], s["removed"], True
    return _to_scen(d)


def case3(cfg: EnvCfg, omega: float) -> Scenario:
    """Case III (§VII-B3, A-37): targets 3 and 5 as printed + the reflecting-arc Target 4 at angular speed omega."""
    with open(os.path.join(CFG_DIR, "scenario_case3.json")) as f:
        sc = json.load(f)
    d = _empty(cfg.M)
    for i, s in enumerate(sc["linear"]):
        d["p0"][i], d["v"][i] = s["start"], s["v"]
        d["born"][i], d["death"][i], d["valid"][i] = s["born"], s["removed"], True
    c = sc["circle"]
    i = len(sc["linear"])
    d["born"][i], d["death"][i], d["valid"][i], d["circ"][i] = c["born"], c["removed"], True, True
    d["circ_r"][i] = c["radius"]
    d["circ_lo"][i] = np.deg2rad(c["deg"][0])
    d["circ_span"][i] = np.deg2rad(c["deg"][1] - c["deg"][0])
    d["circ_step"][i] = omega * cfg.T0
    return _to_scen(d)


def scenario_checksums() -> dict:
    from .common import sha256_file
    return {n: sha256_file(os.path.join(CFG_DIR, n)) for n in ("scenario_case1.json", "scenario_case1_v1.json", "scenario_case1_v2_overflight.json",
                                                                 "scenario_case3.json")}


def add_circle(scen: Scenario, key, cfg: EnvCfg, n_slots: int, radius: float = 4300.0, span_deg: float = 70.0,
               omegas=(0.005, 0.5), p_fast=None) -> Scenario:
    """Case III TRAINING scenario (A-37): the spawn table plus one reflecting-arc target present for the whole
    episode in the LAST table row (radius 4300 m and span 70 deg as read off Figs. 13-14; arc start bearing
    U(0, 2 pi) and omega drawn from {0.005, 0.5} rad/s per episode, ASSUMED). The row is overwritten (a spawn
    arrival there would need 12 arrivals in one 3000-slot episode). p_fast (traced, optional): probability of the fast
    omega; None = 0.5 (full-run); 1 / 0 = per-omega training."""
    k_lo, k_w = jax.random.split(key)
    i = cfg.M - 1
    fast = jax.random.bernoulli(k_w) if p_fast is None else jax.random.bernoulli(k_w, p_fast)
    om = jnp.where(fast, omegas[1], omegas[0])
    return scen._replace(
        p0=scen.p0.at[i].set(0.0), v=scen.v.at[i].set(0.0), born=scen.born.at[i].set(0),
        death=scen.death.at[i].set(n_slots), valid=scen.valid.at[i].set(True), circ=scen.circ.at[i].set(True),
        circ_r=scen.circ_r.at[i].set(radius), circ_lo=scen.circ_lo.at[i].set(jax.random.uniform(k_lo, (), maxval=2 * jnp.pi)),
        circ_span=scen.circ_span.at[i].set(jnp.deg2rad(span_deg)), circ_step=scen.circ_step.at[i].set(om * cfg.T0))


def replica_case4(cfg: EnvCfg) -> Scenario:
    """Case IV test schedule (A-39a, full-run): Fig. 17's six trajectories coincide with the Case I replica (Fig. 3)
    within the +-500 m reading precision (read at 320 dpi, 2026-10-19), and Fig. 18's track births match Fig. 4's,
    so the Case I replica schedule is reused; only the interference regions (cfg.interf) differ. Re-checked in the
    re-evaluation: Fig. 17 overlays Fig. 3 track for track. Residual: Fig. 18 shades Target 2 'in region 1'
    from ~700 to ~2000, while this replica keeps Target 2 inside region 1 from slot ~640 until its removal at 3000."""
    return replica_case1(cfg)
