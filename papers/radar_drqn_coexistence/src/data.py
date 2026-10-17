"""Environment of Thornton et al. (IEEE TCCN 2020, arXiv 2006.13173v3) as pure jax.lax functions.

There is no dataset: the "data" of this paper is the simulated spectrum. This module holds
  * the static tables (action masks, a*(theta), reward tables per reading, Table-II indicator tables),
  * the interference chains (Markov on/off, FH sweep) as finite chains over a latent z with a pattern theta(z),
  * the target model and the jitted, vmappable env step.

Timing: the observation at step t carries theta_obs,t = theta(z_t), the occupancy sensed in the
previous pulse; the radar transmits a_t in pulse t whose interference is theta(z_{t+1}); r_{t+1} = R(a_t, theta(z_{t+1})).

Bit convention: sub-band i (0..4, low to high frequency) is bit i of a theta index, theta_idx = sum_i theta_i << i.
Every table value cites its source in src/configs/sources.yaml (env:, interference:).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

N_SUB = 5                      # Table I: 100 MHz / 20 MHz sub-bands
SUB_MHZ = 20.0                 # Table I
N_THETA = 2 ** N_SUB           # paper §III-A: 2^N interference states
N_POS, N_VEL = 50, 10          # Table I (P, V)
CPI = 1000                     # Table I: pulses per CPI
Z_MAX = 5                      # padded chain size (Markov uses 2 states, FH 5)
OBS_DIM = 2 + N_SUB            # [x/49, v/9, theta_obs(5)]
FULL_BAND = 4                  # action index of 11111 (lexicographic order)
BLOCK_B = 11                   # action index of 00111
READINGS = ("eq3", "lin")      # PRIMARY Eq. 3 as printed, SECONDARY R_lin


# ----------------------------------------------------------------------------- static tables (numpy, float64)
def action_masks() -> np.ndarray:
    """[15, 5] contiguous masks in lexicographic (start, end) order (paper §III-A fn. 2)."""
    out = []
    for s in range(N_SUB):
        for e in range(s, N_SUB):
            m = np.zeros(N_SUB, np.int32)
            m[s:e + 1] = 1
            out.append(m)
    return np.stack(out)


MASKS = action_masks()
N_ACT = MASKS.shape[0]         # 15


def theta_bits(idx: int) -> np.ndarray:
    return np.array([(idx >> i) & 1 for i in range(N_SUB)], np.int32)


THETA_BITS = np.stack([theta_bits(i) for i in range(N_THETA)])          # [32, 5]


def astar_mask(theta: np.ndarray) -> np.ndarray:
    """Largest free contiguous block of theta; ties -> lowest start index; all-occupied -> empty mask."""
    best_s, best_len, s = 0, 0, None
    for i in range(N_SUB + 1):
        free = i < N_SUB and theta[i] == 0
        if free and s is None:
            s = i
        if not free and s is not None:
            if i - s > best_len:
                best_s, best_len = s, i - s
            s = None
    m = np.zeros(N_SUB, np.int32)
    m[best_s:best_s + best_len] = 1
    return m


ASTAR_MASK = np.stack([astar_mask(THETA_BITS[i]) for i in range(N_THETA)])   # [32, 5]


def _mask_to_action(m: np.ndarray) -> int:
    for k in range(N_ACT):
        if np.array_equal(MASKS[k], m):
            return k
    return FULL_BAND    # theta = 11111 has an empty a*; SAA then transmits the full band (never reached in E1-E5)


ASTAR_ACTION = np.array([_mask_to_action(ASTAR_MASK[i]) for i in range(N_THETA)], np.int32)   # SAA table [32]


def counts(a: int, th: int) -> tuple[int, int]:
    """(N_c, N_mo) for action index a against theta index th (overlap reading of Defs 1-2)."""
    m, t, s = MASKS[a], THETA_BITS[th], ASTAR_MASK[th]
    return int((m * t).sum()), int(s.sum() - (s * m).sum())


def reward_value(nc: int, nmo: int, reading: str, beta1: float = 5.0, beta2: float = 6.0) -> float:
    """Eq. 3 (reading 'eq3') or the figure-identified linear reading 'lin'; (beta1, beta2) = Table I."""
    if nc > 0:
        return 0.0
    if nmo == 0:
        return 1.0
    if reading == "eq3":
        return beta1 / (beta2 * nmo)
    if reading == "lin":
        return 1.0 - nmo * (beta2 - beta1) / beta2
    if reading == "pow":   # rejected reading; kept only for the identification table
        return (beta1 / beta2) ** nmo
    raise ValueError(reading)


def reward_table(reading: str, beta1: float = 5.0, beta2: float = 6.0) -> np.ndarray:
    """[15, 32] float64: R(a, theta_next)."""
    return np.array([[reward_value(*counts(a, th), reading, beta1, beta2) for th in range(N_THETA)]
                     for a in range(N_ACT)])


COLL_TABLE = np.array([[float(counts(a, th)[0] > 0) for th in range(N_THETA)] for a in range(N_ACT)])   # [15, 32]
# % missed (Table II) = collision-free pulses with N_mo > 0, i.e. the third branch of Eq. 3 (amendment 2026-10-16):
# counting N_mo > 0 on every pulse gives FH-SAA 80 % against the printed 40.0 %; the branch partition gives 40 % exactly.
MISS_TABLE = np.array([[float(counts(a, th)[0] == 0 and counts(a, th)[1] > 0) for th in range(N_THETA)]
                       for a in range(N_ACT)])   # [15, 32]
ACT_SIZE = MASKS.sum(1).astype(np.float64)                                                           # [15]


def theta_idx(bits) -> int:
    return int(sum(int(b) << i for i, b in enumerate(bits)))


# ----------------------------------------------------------------------------- interference chains
class Chain(NamedTuple):
    """A finite interference chain: latent z in 0..nz-1, pattern theta(z) (theta index), transition T[z, z']."""
    nz: int
    pattern: np.ndarray     # [nz] int theta indices
    T: np.ndarray           # [nz, nz] float64
    z0: int


ON_PATTERN = theta_idx([1, 1, 0, 0, 0])     # paper §IV-B: active interferer occupies [1,1,0,0,0]


def markov_chain(p: float) -> Chain:
    """Symmetric on/off chain, flip w.p. p every pulse, begins OFF (paper §IV-B)."""
    return Chain(2, np.array([0, ON_PATTERN]), np.array([[1 - p, p], [p, 1 - p]]), 0)


def fh_chain() -> Chain:
    """Deterministic single-sub-band sweep e1 -> e2 -> ... -> e5 -> e1 starting at e1 (paper §IV-C)."""
    T = np.zeros((5, 5))
    for k in range(5):
        T[k, (k + 1) % 5] = 1.0
    return Chain(5, np.array([1 << k for k in range(5)]), T, 0)


def pad_chain(c: Chain) -> tuple[np.ndarray, np.ndarray]:
    """Pad to Z_MAX: unreachable states self-loop with pattern 0."""
    pat = np.zeros(Z_MAX, np.int32)
    pat[:c.nz] = c.pattern
    T = np.eye(Z_MAX)
    T[:c.nz, :c.nz] = c.T
    return pat, T


# ----------------------------------------------------------------------------- experiments
def experiment(name: str) -> dict:
    """Train chain, eval chains per segment and the segment lengths in CPIs (eval_cpis total = 70)."""
    if name == "mc_p04":
        return dict(train=markov_chain(0.4), eval=[markov_chain(0.4)], seg_cpis=[70], seg_p=[0.4])
    if name == "mc_p02":
        return dict(train=markov_chain(0.2), eval=[markov_chain(0.2)], seg_cpis=[70], seg_p=[0.2])
    if name == "mc_sched":   # Figure 4; train p = 0.1; segments 15/15/15/15/10
        ps = [0.1, 0.2, 0.4, 0.6, 0.8]
        return dict(train=markov_chain(0.1), eval=[markov_chain(p) for p in ps], seg_cpis=[15, 15, 15, 15, 10],
                    seg_p=ps)
    if name == "fh":
        return dict(train=fh_chain(), eval=[fh_chain()], seg_cpis=[70], seg_p=[None])
    if name in ("mc_p01", "mc_p06", "mc_p08"):   # toy-check Monte-Carlo checks of Table D1 only
        p = {"mc_p01": 0.1, "mc_p06": 0.6, "mc_p08": 0.8}[name]
        return dict(train=markov_chain(p), eval=[markov_chain(p)], seg_cpis=[70], seg_p=[p])
    raise ValueError(name)


# global experiment ids (key tree): append only, never reorder
EXPERIMENTS = ("mc_p04", "mc_p02", "mc_sched", "fh", "mc_p01", "mc_p06", "mc_p08")
N_SEG = 5


class EnvParams(NamedTuple):
    """Per-lane env arrays (vmappable)."""
    pattern: jax.Array      # [Z] int32 theta index of each latent state
    T_train: jax.Array      # [Z, Z]
    T_eval: jax.Array       # [N_SEG, Z, Z]
    seg_end: jax.Array      # [N_SEG] cumulative eval-CPI end of each segment
    z0: jax.Array           # [] int32
    R: jax.Array            # [15, 32] reward table of the lane's reading
    p_move: jax.Array       # [] target move probability per pulse (0.1; 0 = static-target PI sensitivity)


def env_params(exp_name: str, reading: str, eval_cpis: int | None = None, p_move: float = 0.1) -> EnvParams:
    """Arrays for one (experiment, reading). eval_cpis rescales the segment lengths (small-scale proxy) keeping proportions."""
    ex = experiment(exp_name)
    pat, Ttr = pad_chain(ex["train"])
    Tev = [pad_chain(c)[1] for c in ex["eval"]]
    seg = list(ex["seg_cpis"])
    if eval_cpis is not None and eval_cpis != sum(seg):
        seg = scale_segments(seg, eval_cpis)
    while len(Tev) < N_SEG:
        Tev.append(Tev[-1])
        seg.append(0)
    seg_end = np.cumsum(seg)
    seg_end[len(ex["eval"]):] = 10 ** 9          # padded segments never start
    return EnvParams(jnp.asarray(pat), jnp.asarray(Ttr, jnp.float32), jnp.asarray(np.stack(Tev), jnp.float32),
                     jnp.asarray(seg_end, jnp.int32), jnp.asarray(ex["train"].z0, jnp.int32),
                     jnp.asarray(reward_table(reading), jnp.float32), jnp.asarray(p_move, jnp.float32))


def scale_segments(seg: list[int], total: int) -> list[int]:
    raw = np.array(seg, float) * total / sum(seg)
    out = np.maximum(1, np.floor(raw)).astype(int)
    out[-1] = total - out[:-1].sum()
    return out.tolist()


# ----------------------------------------------------------------------------- dynamics (pure jax)
class EnvState(NamedTuple):
    z: jax.Array      # latent interference state; theta_obs = pattern[z]
    x: jax.Array      # target position 0..49
    v: jax.Array      # target velocity 0..9
    d: jax.Array      # direction +-1


THETA_BITS_J = jnp.asarray(THETA_BITS, jnp.float32)
MASKS_J = jnp.asarray(MASKS, jnp.int32)
COLL_J = jnp.asarray(COLL_TABLE, jnp.float32)
MISS_J = jnp.asarray(MISS_TABLE, jnp.float32)
SIZE_J = jnp.asarray(ACT_SIZE, jnp.float32)
ASTAR_ACTION_J = jnp.asarray(ASTAR_ACTION, jnp.int32)
P_MOVE = 0.1    # probability per pulse that the target moves one cell


def target_reset(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """per-CPI x0 ~ U{0..49}, v ~ U{0..9}, direction sgn(v - 4.5)."""
    kx, kv = jax.random.split(key)
    x = jax.random.randint(kx, (), 0, N_POS)
    v = jax.random.randint(kv, (), 0, N_VEL)
    return x, v, jnp.where(v >= 5, 1, -1)


def target_step(x, v, d, key, p_move=P_MOVE):
    """move one cell w.p. p_move (0.1) in direction d; at the edge stay, flip d and mirror v."""
    move = jax.random.uniform(key) < p_move
    nx = x + d
    out = (nx < 0) | (nx >= N_POS)
    x2 = jnp.where(move & ~out, nx, x)
    d2 = jnp.where(move & out, -d, d)
    v2 = jnp.where(move & out, N_VEL - 1 - v, v)
    return x2, v2, d2


def env_reset(key_stream: jax.Array, params: EnvParams) -> EnvState:
    """Start of a stream (training or evaluation): interference at z0 (OFF / e1), target drawn for CPI 0."""
    x, v, d = target_reset(jax.random.fold_in(jax.random.fold_in(key_stream, 1), 0))
    return EnvState(params.z0, x, v, d)


def obs_of(s: EnvState, params: EnvParams) -> jax.Array:
    """[x/49, v/9, theta_obs bits]."""
    th = THETA_BITS_J[params.pattern[s.z]]
    return jnp.concatenate([jnp.stack([s.x / (N_POS - 1.0), s.v / (N_VEL - 1.0)]).astype(jnp.float32), th])


def full_state_index(s: EnvState, params: EnvParams) -> jax.Array:
    """Index of the full MDP state (x, v, theta_obs) in 0..15999, used by policy iteration."""
    return (s.x * N_VEL + s.v) * N_THETA + params.pattern[s.z]


def sample_next_z(z, T, key):
    """Inverse-CDF draw of z' ~ T[z, :]. Using one uniform per step gives common random numbers across policies."""
    u = jax.random.uniform(key)
    cdf = jnp.cumsum(T[z])
    return jnp.minimum(jnp.sum(cdf <= u), Z_MAX - 1).astype(jnp.int32)


def env_step(s: EnvState, a: jax.Array, t_stream: jax.Array, T: jax.Array, key_stream: jax.Array,
             params: EnvParams):
    """One pulse. t_stream = step index within this stream (CPI boundary when (t+1) % 1000 == 0).

    Keys: interference uniform = fold_in(fold_in(key_stream, 0), t); target = fold_in(fold_in(key_stream, 1), .).
    Returns (s', r, info) with info = (collision, missed, |a| in sub-bands, theta_next index).
    """
    z2 = sample_next_z(s.z, T, jax.random.fold_in(jax.random.fold_in(key_stream, 0), t_stream))
    th2 = params.pattern[z2]
    r = params.R[a, th2]
    kt = jax.random.fold_in(jax.random.fold_in(key_stream, 1), t_stream + 1)
    xm, vm, dm = target_step(s.x, s.v, s.d, jax.random.fold_in(kt, 1), params.p_move)
    new_cpi = (t_stream + 1) % CPI == 0
    xr, vr, dr = target_reset(jax.random.fold_in(kt, 0))
    s2 = EnvState(z2, jnp.where(new_cpi, xr, xm), jnp.where(new_cpi, vr, vm), jnp.where(new_cpi, dr, dm))
    return s2, r, (COLL_J[a, th2], MISS_J[a, th2], SIZE_J[a], th2)


def T_for_cpi(params: EnvParams, cpi: jax.Array, n_train_cpis: int) -> jax.Array:
    """Transition matrix in force during global CPI index `cpi` (training chain, then the eval segment schedule)."""
    e = cpi - n_train_cpis
    seg = jnp.sum(e >= params.seg_end)          # number of finished segments
    seg = jnp.minimum(seg, N_SEG - 1)
    return jnp.where(e < 0, params.T_train, params.T_eval[seg])


def segment_of(params: EnvParams, eval_cpi: jax.Array) -> jax.Array:
    return jnp.minimum(jnp.sum(eval_cpi >= params.seg_end), N_SEG - 1)
