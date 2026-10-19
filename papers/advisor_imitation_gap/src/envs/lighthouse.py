"""2D Lighthouse (paper Ex. 2, App. A.1, A.5.2) as a pure-JAX environment, ported from AllenAct v0.5.0
allenact_plugins/lighthouse_plugin/{lighthouse_environment,lighthouse_sensors,lighthouse_tasks}.py.

  * grid {-N..N}^2, N = WORLD_RADIUS = 15                                       [official code]
  * start (0, 0); goal = uniform corner                                         [official code; App. A.5.2]
  * actions: 0 = +1 on dim 0, 1 = +1 on dim 1, 2 = -1 on dim 0, 3 = -1 on dim 1 (clipped at +-N)   [official code]
  * reward -0.01 per step, +1 on reaching the goal (0.99 net); the 1000th step without success gets
    -0.01 / (1 - 0.99) = -1 instead                                             [official FindGoalLightHouseTask._step]
  * observation f_i = (4 corner codes, on-border code, last action) where a corner code is the corner's
    true value once the agent has ever been within Chebyshev distance i of it, else the value of the cell
    at the view-window corner in that direction (EMPTY 0, GOAL 1, WRONG_CORNER 2, WALL 3)  [official get_corner_observation]
  * the student's 6400-d input is the official FactorialDesignCornerSensor (DEGREE = -1) row for that tuple:
    patsy's full-rank coding of C(s0):C(s1):C(s2):C(s3):C(b0):C(a0) (NOT a one-hot).
    We regenerate it from the 13-subterm structure (configs/lighthouse_patsy_subterms.json) and store, per
    tuple, the <= 13 active columns; tests/test_official_parity.py checks the full matrix against patsy's sha256.
  * teacher = the official hand-coded f_j rule (query_expert, world_dim 2, deterministic)    [official code]
"""
from __future__ import annotations

import itertools
import json
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

N = 15                     # official WORLD_RADIUS
N_ACTIONS = 4              # App. A.8
MAX_STEPS = 1000           # App. A.10 / official MAX_STEPS
STEP_PENALTY = -0.01       # official
FOUND = 1.0                # official FOUND_TARGET_REWARD
GAMMA = 0.99               # official DISCOUNT_FACTOR (timeout reward)
EMPTY, GOAL, WRONG, WALL = 0, 1, 2, 3
NONE = 4                   # 'no last action' / 'not on border' code (2 * world_dim)
FEAT_DIM = 6400            # 4**4 * 5**2
CORNERS = np.array(sorted([(-N, -N), (-N, N), (N, -N), (N, N)]), np.int32)  # official sorted order
_SUBTERMS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "lighthouse_patsy_subterms.json")
LEVELS = (4, 4, 4, 4, 5, 5)
NAMES = ("s0", "s1", "s2", "s3", "b0", "a0")


def optimal_len(i: int) -> float:
    """official LightHouseEnvironment.optimal_ave_ep_length (= 5N - 3i)."""
    tau = 2 * (N - i)
    return 0.25 * (4 * 2 * i + 10 * tau)


# ------------------------------------------------------------------ factorial-design features
def design_active_columns(order: str = "first_fastest") -> np.ndarray:
    """[6400 tuples, 13 subterms] active column index (-1 if the subterm is zero for that tuple).

    Tuple index = itertools.product order over LEVELS (official tuple_to_ind). A subterm with factors F and coding
    'full' (one-hot over L levels) or 'reduced' (treatment: level 0 -> zero row) contributes the Kronecker product
    of its factors' coding rows; columns of successive subterms are concatenated in patsy's order."""
    sub = json.load(open(_SUBTERMS))
    tuples = np.array(list(itertools.product(*[range(l) for l in LEVELS])), np.int64)
    out = np.full((len(tuples), len(sub)), -1, np.int64)
    off = 0
    for k, (facs, kinds, ncol) in enumerate(sub):
        idxs = [NAMES.index(f) for f in facs]
        widths = [LEVELS[i] if kd == "full" else LEVELS[i] - 1 for i, kd in zip(idxs, kinds)]
        assert int(np.prod(widths)) == ncol, (facs, widths, ncol)
        col = np.zeros(len(tuples), np.int64)
        ok = np.ones(len(tuples), bool)
        # mixed-radix position; 'first_fastest' = first factor varies fastest
        mult = 1
        seq = list(zip(idxs, kinds, widths))
        if order != "first_fastest":
            seq = seq[::-1]
        for i, kd, wdt in seq:
            v = tuples[:, i] if kd == "full" else tuples[:, i] - 1
            ok &= v >= 0
            col += np.maximum(v, 0) * mult
            mult *= wdt
        out[:, k] = np.where(ok, off + col, -1)
        off += ncol
    assert off == FEAT_DIM
    return out


def design_matrix(order: str = "first_fastest") -> np.ndarray:
    act = design_active_columns(order)
    m = np.zeros((act.shape[0], FEAT_DIM), bool)
    r, c = np.nonzero(act >= 0)
    m[r, act[r, c]] = True
    return m


# ------------------------------------------------------------------ environment
class State(NamedTuple):
    pos: jax.Array        # int32 [2]
    goal: jax.Array       # int32 corner index 0..3
    closest: jax.Array    # int32 [4] closest-ever Chebyshev distance to each corner
    last: jax.Array       # int32 last action, NONE before the first step
    t: jax.Array          # int32
    found: jax.Array      # bool
    ret: jax.Array        # float32


def reset(key, goal=None) -> State:
    if goal is None:
        goal = jax.random.randint(key, (), 0, 4)
    pos = jnp.zeros(2, jnp.int32)
    closest = jnp.max(jnp.abs(jnp.asarray(CORNERS) - pos[None]), axis=1).astype(jnp.int32)
    z = jnp.zeros((), jnp.int32)
    return State(pos, jnp.asarray(goal, jnp.int32), closest, z + NONE, z, jnp.zeros((), bool), jnp.zeros((), jnp.float32))


def step(s: State, a):
    a = jnp.asarray(a, jnp.int32)
    delta = jnp.where(a >= 2, -1, 1)
    ind = a % 2
    old = s.pos[ind]
    new = jnp.clip(old + delta, -N, N)
    pos = s.pos.at[ind].set(new)
    moved = new != old
    d = jnp.max(jnp.abs(jnp.asarray(CORNERS) - pos[None]), axis=1).astype(jnp.int32)
    closest = jnp.where(moved, jnp.minimum(s.closest, d), s.closest)
    found = jnp.all(pos == jnp.asarray(CORNERS)[s.goal])
    r = jnp.where(found, STEP_PENALTY + FOUND, jnp.where(s.t == MAX_STEPS - 1, STEP_PENALTY / (1 - GAMMA), STEP_PENALTY))
    t = s.t + 1
    ns = State(pos, s.goal, closest, a, t, found, s.ret + r)
    return ns, r.astype(jnp.float32), found | (t >= MAX_STEPS)


def _cell_value(s: State, cell):
    """world_tensor value at grid coordinates `cell` (each in [-N, N])."""
    on_wall = jnp.any(jnp.abs(cell) == N)
    corner_hit = jnp.all(jnp.asarray(CORNERS) == cell[None], axis=1)  # [4]
    is_corner = jnp.any(corner_hit)
    is_goal = corner_hit[s.goal]
    return jnp.where(is_goal, GOAL, jnp.where(is_corner, WRONG, jnp.where(on_wall, WALL, EMPTY))).astype(jnp.int32)


def corner_obs(s: State, view_radius):
    """official get_corner_observation: (c0, c1, c2, c3, on_border, last_action)."""
    signs = jnp.sign(jnp.asarray(CORNERS))                      # corner > 0 -> +1 else -1
    cells = jnp.clip(s.pos[None] + view_radius * signs, -N, N)  # [4, 2]
    view_vals = jax.vmap(lambda c: _cell_value(s, c))(cells)
    true_vals = jnp.where(jnp.arange(4) == s.goal, GOAL, WRONG)
    seen = s.closest <= view_radius
    vals = jnp.where(seen, true_vals, view_vals)
    on_border = jnp.concatenate([s.pos == N, s.pos == -N])      # [4], indexed like actions
    first_border = jnp.argmax(on_border)
    border = jnp.where((s.last == NONE), s.last,
                       jnp.where(on_border[jnp.clip(s.last, 0, 3)], s.last,
                                 jnp.where(jnp.any(on_border), first_border, NONE)))
    return jnp.concatenate([vals, jnp.stack([border, s.last])]).astype(jnp.int32)


def tuple_index(tup):
    """itertools.product index of (s0..s3, b, a) over LEVELS (official tuple_to_ind)."""
    idx = jnp.zeros((), jnp.int32)
    for k, l in enumerate(LEVELS):
        idx = idx * l + tup[k]
    return idx


def teacher(s: State, view_radius):
    """official FindGoalLightHouseTask.query_expert for world_dim 2 (deterministic)."""
    o = corner_obs(s, view_radius)
    tl, tr, bl, br, hit, last = o[0], o[1], o[2], o[3], o[4], o[5]
    d, r, u, l, none = 0, 1, 2, 3, 4
    W = WRONG
    nw = lambda *xs: ~jnp.any(jnp.stack([x == W for x in xs]))
    aw = lambda *xs: jnp.all(jnp.stack([x == W for x in xs]))
    rot = jnp.where(last == r, u, jnp.where(last == u, l, jnp.where(last == l, d, r)))  # last==d -> r
    zero_vis = jnp.where((tr == W) | (last == hit), rot, last)
    conds = [
        (tr == GOAL, jnp.where(hit != r, r, u)),
        (br == GOAL, jnp.where(hit != d, d, r)),
        (bl == GOAL, jnp.where(hit != l, l, d)),
        (tl == GOAL, jnp.where(hit != u, u, l)),
        ((tr == W) & nw(br, bl, tl), l),
        ((br == W) & nw(bl, tl, tr), u),
        ((bl == W) & nw(tl, tr, br), r),
        ((tl == W) & nw(tr, br, bl), d),
        (aw(tr, br) & nw(bl, tl), l),
        (aw(br, bl) & nw(tl, tr), u),
        (aw(bl, tl) & nw(tr, br), r),
        (aw(tl, tr) & nw(br, bl), d),
        ((hit != none) & (tr == br) & (br == bl) & (bl == tl), zero_vis),
        ((last == r) & (tr == WALL), u),
        ((last == u) & (tl == WALL), l),
        ((last == l) & (bl == WALL), d),
        ((last == d) & (br == WALL), r),
        (last == none, r),
    ]
    out = last
    for c, v in conds[::-1]:
        out = jnp.where(c, v, out)
    return out.astype(jnp.int32)


def episode_info(s: State):
    return {"ret": s.ret, "len": s.t.astype(jnp.float32), "found": s.found.astype(jnp.float32)}
