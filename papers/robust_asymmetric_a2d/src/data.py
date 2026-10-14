"""Gridworld environments of the paper (Frozen Lake, Tiger Door, Tiger Door 1/2/3) as pure JAX functions.

There is no dataset in this paper; "data" is the environment. Re-implemented from the paper (§4, Fig. 3, Fig. B.1,
App. D.1) and the semantics of the official env (github.com/plai-group/a2d @368aaf5, `lavagap.py`, `tigerdoor.py`,
`tigerdoor_osc.py`, `minigrid.py`), verified step-by-step against it in tests/test_official_parity.py. No official code is
copied. Coordinates: (row, col) inside the wall border; row = MiniGrid y - 1, col = MiniGrid x - 1.

Actions (official DIR_TO_VEC): 0 -> +x (col+1), 1 -> +y (row+1), 2 -> -x (col-1), 3 -> -y (row-1).
Per step: reward -2 (also on the terminal step); entering the goal +20 (terminal); entering the hazard -100 (terminal);
bumping a wall = stay; entering the button sets `illum` (persistent); the episode ends at step 100 (time-out, terminal).

Observations (all computed from the latent (pos, hidden, illum) so buffers store latents, not tensors):
  state_vec   : omniscient compact expert input (FL 34 = 25 pos | 9 hazard; TD 30 = 25 | 2 goal | 2 lava | 1 illum;
                TD-k 54 = 49 | 2 | 2 | 1)                                                     [official gen_obs]
  partial_vec : compact-partial trainee input (FL 25 = pos; TD/TD-k = state_vec with the last 5 zeroed until illum)
  image       : 42x42x3 render of the 7x7 grid (6 px tiles) in [0,1] + N(0, 0.05) pixel noise, clipped, stored as
                float16 like the official buffer; FL hazard never drawn; TD goal/lava drawn once illuminated.
Position one-hot index = col * n + row (official `arange(n*n).reshape(n, n)[x-1, y-1]`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

STEP_R, GOAL_R, HAZ_R = -2.0, 20.0, -100.0       # paper §4 / App. D.1; official step()
MAX_STEPS = 100                                   # official _tmax (paper says 200)
GAMMA = 0.995                                     # App. D.1
NOISE_SD = 0.05                                   # official full_rendering_obs_noise
TILE = 6                                          # official tile_size -> 42 px for a 7x7 grid
N_ACT = 4
MOVES = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], np.int32)   # (drow, dcol) per action index


@dataclass(frozen=True)
class EnvSpec:
    name: str
    kind: str                 # 'fl' (hazard hidden, no button) or 'td' (goal side hidden, button)
    n: int                    # interior side
    floor: np.ndarray         # (n, n) bool, walkable (incl. goal/hazard/button cells)
    start: tuple
    button: tuple | None
    goal: np.ndarray          # (K, n, n) bool
    haz: np.ndarray           # (K, n, n) bool
    mdp_opt: float            # exact optima (see src/exact.py), for normalization / reporting
    pomdp_opt: float

    @property
    def K(self) -> int:
        return self.goal.shape[0]

    @property
    def n_pos(self) -> int:
        return self.n * self.n

    @property
    def state_dim(self) -> int:
        return self.n_pos + (9 if self.kind == 'fl' else 5)

    @property
    def partial_dim(self) -> int:
        return self.n_pos + (0 if self.kind == 'fl' else 5)

    @property
    def has_image(self) -> bool:
        return self.n == 5


def _fl() -> EnvSpec:
    n = 5
    floor = np.ones((n, n), bool)
    goal = np.zeros((9, n, n), bool)
    haz = np.zeros((9, n, n), bool)
    for i in range(9):                                  # official potential_lava_locs order: x = 2 + i%3, y = 2 + i//3
        goal[i, 2, 4] = True
        haz[i, 1 + i // 3, 1 + i % 3] = True
    return EnvSpec('FL', 'fl', n, floor, (2, 0), None, goal, haz, 32.0 / 3.0, 4.0)


def _td() -> EnvSpec:
    n = 5
    cells = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (3, 2), (3, 3), (3, 4)]
    floor = np.zeros((n, n), bool)
    for r, c in cells:
        floor[r, c] = True
    goal = np.zeros((2, n, n), bool)
    haz = np.zeros((2, n, n), bool)
    goal[0, 3, 4] = haz[0, 1, 4] = True                 # goal_idx 0: goal at MiniGrid (5,4), lava at (5,2)
    goal[1, 1, 4] = haz[1, 3, 4] = True
    return EnvSpec('TD', 'td', n, floor, (0, 0), (3, 0), goal, haz, 6.0, 2.0)


def _tdk(k: int) -> EnvSpec:
    """Tiger Door k (official TigerDoorEnvOSC lateral_steps k): 9x9 grid, 2-row corridor at MiniGrid y = 3, 4."""
    n = 7
    floor = np.zeros((n, n), bool)
    lo, hi = 3 - k, 3 + k                               # interior cols (MiniGrid x = 4 -/+ k)
    floor[2:4, lo:hi + 1] = True
    goal = np.zeros((2, n, n), bool)
    haz = np.zeros((2, n, n), bool)
    goal[0, 2:4, hi] = haz[0, 2:4, lo] = True           # goal_idx 0: goal on the high-x side
    goal[1, 2:4, lo] = haz[1, 2:4, hi] = True
    mdp = {1: 18.0, 2: 16.0, 3: 14.0}[k]
    return EnvSpec(f'TD{k}', 'td', n, floor, (3, 3), (2, 3), goal, haz, mdp, mdp - 2.0)


@lru_cache(maxsize=None)
def get_spec(name: str) -> EnvSpec:
    return {'FL': _fl, 'TD': _td, 'TD1': lambda: _tdk(1), 'TD2': lambda: _tdk(2), 'TD3': lambda: _tdk(3)}[name]()


# ----------------------------------------------------------------------------------------------- env dynamics
class EnvState(NamedTuple):
    r: jnp.ndarray
    c: jnp.ndarray
    h: jnp.ndarray        # hidden variable: FL hazard index (0..8), TD goal side (0/1)
    illum: jnp.ndarray    # bool
    t: jnp.ndarray        # steps taken in the episode


class Latent(NamedTuple):
    """Everything an observation is a function of (stored instead of tensors)."""
    pos: jnp.ndarray      # one-hot index col * n + row
    h: jnp.ndarray
    illum: jnp.ndarray


def reset(spec: EnvSpec, key) -> EnvState:
    h = jax.random.randint(key, (), 0, spec.K)
    z = jnp.zeros((), jnp.int32)
    return EnvState(z + spec.start[0], z + spec.start[1], h.astype(jnp.int32), jnp.zeros((), bool), z)


def step(spec: EnvSpec, st: EnvState, a):
    """One MiniGrid step. Returns (new_state, reward, done)."""
    floor = jnp.asarray(spec.floor)
    mv = jnp.asarray(MOVES)[a]
    nr, nc = st.r + mv[0], st.c + mv[1]
    inb = (nr >= 0) & (nr < spec.n) & (nc >= 0) & (nc < spec.n)
    nrc, ncc = jnp.clip(nr, 0, spec.n - 1), jnp.clip(nc, 0, spec.n - 1)
    ok = inb & floor[nrc, ncc]
    r2, c2 = jnp.where(ok, nrc, st.r), jnp.where(ok, ncc, st.c)
    is_goal = ok & jnp.asarray(spec.goal)[st.h, nrc, ncc]
    is_haz = ok & jnp.asarray(spec.haz)[st.h, nrc, ncc]
    illum = st.illum
    if spec.button is not None:
        illum = illum | (ok & (nrc == spec.button[0]) & (ncc == spec.button[1]))
    t = st.t + 1
    rew = STEP_R + jnp.where(is_goal, GOAL_R, 0.0) + jnp.where(is_haz, HAZ_R, 0.0)
    done = is_goal | is_haz | (t >= MAX_STEPS)
    return EnvState(r2, c2, st.h, illum, t), rew, done


def latent(spec: EnvSpec, st: EnvState) -> Latent:
    return Latent((st.c * spec.n + st.r).astype(jnp.int32), st.h, st.illum)


# ----------------------------------------------------------------------------------------------- observations
def state_vec(spec: EnvSpec, lat: Latent, dtype=jnp.float32):
    pos = jax.nn.one_hot(lat.pos, spec.n_pos, dtype=dtype)
    if spec.kind == 'fl':
        return jnp.concatenate([pos, jax.nn.one_hot(lat.h, 9, dtype=dtype)], -1)
    g = jax.nn.one_hot(lat.h, 2, dtype=dtype)
    lv = jax.nn.one_hot(1 - lat.h, 2, dtype=dtype)
    return jnp.concatenate([pos, g, lv, lat.illum.astype(dtype)[..., None]], -1)


def partial_vec(spec: EnvSpec, lat: Latent, dtype=jnp.float32):
    pos = jax.nn.one_hot(lat.pos, spec.n_pos, dtype=dtype)
    if spec.kind == 'fl':
        return pos
    rest = state_vec(spec, lat, dtype)[..., spec.n_pos:]
    return jnp.concatenate([pos, rest * lat.illum.astype(dtype)[..., None]], -1)


# ---- image renderer (own implementation of MiniGrid's 3x super-sampled tile painter; checked vs the official env)
_BG = np.array([0.84 * 255.0, 0.93 * 255.0, 0.94 * 255.0])
_COL = {'wall': (100, 100, 100), 'goal': (0, 255, 0), 'lava': (0, 0, 0.8 * 255.0), 'button': (255, 0, 255)}


def _tile(obj: str | None, agent: bool, ts: int = TILE, sub: int = 3) -> np.ndarray:
    S = ts * sub
    img = np.ones((S, S, 3)) * _BG
    yy, xx = np.meshgrid((np.arange(S) + 0.5) / S, (np.arange(S) + 0.5) / S, indexing='ij')
    img[xx <= 0.031] = (100, 100, 100)                  # grid lines (top and left edges)
    img[yy <= 0.031] = (100, 100, 100)
    if obj is not None:
        img[:, :] = _COL[obj]
    if agent:                                            # triangle, direction fixed to 1 in the official renderer
        th = 0.5 * math.pi
        x = 0.5 + (xx - 0.5) * math.cos(-th) - (yy - 0.5) * math.sin(-th)
        y = 0.5 + (yy - 0.5) * math.cos(-th) + (xx - 0.5) * math.sin(-th)
        a, b, c = np.array([0.12, 0.19]), np.array([0.87, 0.50]), np.array([0.12, 0.81])
        v0, v1 = c - a, b - a
        v2x, v2y = x - a[0], y - a[1]
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d02 = v0[0] * v2x + v0[1] * v2y
        d12 = v1[0] * v2x + v1[1] * v2y
        inv = 1.0 / (d00 * d11 - d01 * d01)
        u = (d11 * d02 - d01 * d12) * inv
        v = (d00 * d12 - d01 * d02) * inv
        img[(u >= 0) & (v >= 0) & ((u + v) < 1)] = (255, 0, 0)
    img = img.reshape(ts, sub, ts, sub, 3).mean(axis=3).mean(axis=1)
    return img.astype(np.uint8)                          # stored into a uint8 canvas (truncation)


@lru_cache(maxsize=None)
def image_table(name: str) -> np.ndarray:
    """Noiseless images in [0,1], shape (n_pos, K_img, 2, 42, 42, 3); index [pos, h (TD only), illum]."""
    spec = get_spec(name)
    assert spec.has_image, 'images only for FL and TD (as in the paper)'
    n, G = spec.n, spec.n + 2
    K_img = 1 if spec.kind == 'fl' else spec.K
    out = np.zeros((spec.n_pos, K_img, 2, G * TILE, G * TILE, 3), np.float32)
    for pos in range(spec.n_pos):
        pr, pc = pos % n, pos // n
        for h in range(K_img):
            for il in range(2):
                canvas = np.zeros((G * TILE, G * TILE, 3), np.uint8)
                for gy in range(G):
                    for gx in range(G):
                        r, c = gy - 1, gx - 1
                        obj = None
                        if not (0 <= r < n and 0 <= c < n) or not spec.floor[r, c]:
                            obj = 'wall'
                        elif spec.kind == 'fl':
                            obj = 'goal' if spec.goal[0, r, c] else None        # hazard never drawn (render_lava=False)
                        else:
                            if spec.button is not None and (r, c) == spec.button:
                                obj = 'button'
                            elif il and spec.goal[h, r, c]:
                                obj = 'goal'
                            elif il and spec.haz[h, r, c]:
                                obj = 'lava'
                        tile = _tile(obj, (r, c) == (pr, pc))
                        canvas[gy * TILE:(gy + 1) * TILE, gx * TILE:(gx + 1) * TILE] = tile
                out[pos, h, il] = canvas.astype(np.float64) / 255.0
    return out


def image(spec: EnvSpec, table: jnp.ndarray, lat: Latent, key, noise: bool = True, dtype=jnp.float32):
    """One 42x42x3 observation (HWC). `table` = jnp.asarray(image_table(name))."""
    hi = 0 if spec.kind == 'fl' else lat.h
    il = lat.illum.astype(jnp.int32) if spec.kind == 'td' else 0
    base = table[lat.pos, hi, il]
    if noise:
        base = jnp.clip(base + NOISE_SD * jax.random.normal(key, base.shape, base.dtype), 0.0, 1.0)
    return base.astype(jnp.float16).astype(dtype)        # official buffer stores observations as torch.half


def obs_fn(spec: EnvSpec, kind: str, table=None, dtype=jnp.float32):
    """Return f(latent, key) -> trainee input of the given kind ('state' | 'partial' | 'image')."""
    if kind == 'state':
        return lambda lat, key: state_vec(spec, lat, dtype)
    if kind == 'partial':
        return lambda lat, key: partial_vec(spec, lat, dtype)
    if kind == 'image':
        return lambda lat, key: image(spec, table, lat, key, True, dtype)
    raise ValueError(kind)


def layout_bytes(spec: EnvSpec) -> bytes:
    """Serialised layout for the layout checksum (README, "Checked against")."""
    parts = [spec.floor.astype(np.int8), spec.goal.astype(np.int8), spec.haz.astype(np.int8),
             np.array(spec.start, np.int8), np.array(spec.button if spec.button else (-1, -1), np.int8)]
    return b''.join(p.tobytes() for p in parts)
