"""Multi-user channel-access environment (paper §III, §VI) -- the paper has no dataset, this file IS the data/world generator.

Pure jax functions of (spec, world, state, key); vmap over experiments at the call site.

World  (fixed within an episode): which users are active (clique size n_j ~ U{n_min..n_max}, arrays padded to n_max),
                                  each user's K' allowed channels (global ids).
State  (changes per slot):        Rayleigh fading h[N,K] (complex AR(1)), primary-user ON/OFF per channel, the user's
                                  last local action / global channel / ACK, running success counts M_n(t).

Collision rule (paper §III, README):  success_n = tx_n AND exactly one SU transmitter on its channel AND channel free
of a primary user.  ACK = success (0 for failure AND for silence, §IV-A.1).

Randomness: world_key -> (world, init, step root); slot t uses fold_in(step_root, t).  Action sampling never touches
these keys, so two policies run on the SAME world_key see identical worlds (paired evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import j0

# paper §VI-C: B = 20 MHz, SNR = 35 dB.  Ergodic Rayleigh capacity (README, closed form) used ONLY to normalise
# the capacity input (ASSUMED): C = (B/ln2) e^{1/rho} E1(1/rho) = 215.957 Mbps.
C_ERGODIC_MBPS = 215.957


@dataclass(frozen=True)
class EnvSpec:
    name: str = "clique"
    n_min: int = 3                 # active users per episode ~ Uniform{n_min..n_max}   (E1: paper §VI-B 3..11)
    n_max: int = 11
    n_channels: int = 1            # K  (global channels)
    k_local: int = 1               # K' (channels a user may choose among; K'=2 in the 100x50 experiments, §VI)
    subset: str = "all"            # "all": user may use every channel (K'==K) | "random": distinct K' of K per user/episode 
                                   # | "fixed": ONE allocation (subset_seed) shared by every episode, covers every channel when n_max >= K (sensitivity arm)
    subset_seed: int = 0           # only for subset == "fixed"
    capacity: str = "const"        # "const": 1 (paper v1/v2 §II-B.1; E1,E6-E8) | "rate": instantaneous rate / C_ERGODIC 
    fading: bool = False           # Rayleigh AR(1) (E2/E3)
    pu: bool = False               # primary users (E3)
    pu_stay: float = 0.9           # ASSUMED
    doppler_hz: float = 100.0      # paper §VI-C
    slot_s: float = 1e-3           # ASSUMED 
    snr_db: float = 35.0           # paper §VI
    bw_hz: float = 20e6            # paper §VI

    @property
    def rho(self) -> float:        # Clarke/Jakes lag-1 correlation  J0(2 pi f_D T_s) = 0.903713
        return float(j0(2 * np.pi * self.doppler_hz * self.slot_s))

    @property
    def snr_lin(self) -> float:
        return float(10 ** (self.snr_db / 10))

    @property
    def obs_dim(self) -> int:      # 2K'+2  (paper §IV-A.1)
        return 2 * self.k_local + 2

    @property
    def n_actions(self) -> int:    # K'+1, index 0 = silent
        return self.k_local + 1


class World(NamedTuple):
    active: jax.Array   # [N] bool
    n_active: jax.Array  # [] int32
    subset: jax.Array   # [N, K'] int32 global channel ids


class State(NamedTuple):
    h: jax.Array        # [N, K, 2] float  (re, im) CN(0,1) fading
    pu_free: jax.Array  # [K] bool
    prev_a: jax.Array   # [N] int32 last LOCAL action (0 = silent)
    prev_g: jax.Array   # [N] int32 last GLOBAL channel (-1 = silent)
    prev_ack: jax.Array  # [N] float
    succ: jax.Array     # [N] float  M_n(t): successes so far 
    choice: jax.Array   # [N] int32 last announced channel (baselines' message exchange only; -1 = none)


def sample_world(spec: EnvSpec, key) -> World:
    kn, ks = jax.random.split(key)
    n = jax.random.randint(kn, (), spec.n_min, spec.n_max + 1)
    active = jnp.arange(spec.n_max) < n
    if spec.subset == "all":
        assert spec.k_local == spec.n_channels
        subset = jnp.broadcast_to(jnp.arange(spec.n_channels, dtype=jnp.int32), (spec.n_max, spec.k_local))
    elif spec.subset == "fixed":
        subset = fixed_subsets(spec)
    else:
        noise = jax.random.uniform(ks, (spec.n_max, spec.n_channels))
        subset = jax.lax.top_k(noise, spec.k_local)[1].astype(jnp.int32)
    return World(active, n.astype(jnp.int32), subset)


def fixed_subsets(spec: EnvSpec):
    """Episode-independent allocation : user i's FIRST channel is perm[i mod K] for a fixed random permutation `perm`
    (every channel is covered when n_max >= K and the load is balanced); the other K'-1 channels are drawn at random without
    repeating the first.  Pure function of (spec.subset_seed, n_max, K, K')."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(spec.subset_seed))
    K, Kp, N = spec.n_channels, spec.k_local, spec.n_max
    first = jax.random.permutation(k1, K)[jnp.arange(N) % K]
    noise = jax.random.uniform(k2, (N, K)).at[jnp.arange(N), first].set(-1.0)       # exclude the first channel from the random draw
    rest = jax.lax.top_k(noise, Kp - 1)[1] if Kp > 1 else first[:, :0]
    return jnp.concatenate([first[:, None], rest], axis=1).astype(jnp.int32)


def init_state(spec: EnvSpec, key) -> State:
    kh, kp = jax.random.split(key)
    N, K = spec.n_max, spec.n_channels
    h = jax.random.normal(kh, (N, K, 2)) / jnp.sqrt(2.0) if spec.fading else jnp.zeros((N, K, 2))
    pu_free = jax.random.bernoulli(kp, 0.5, (K,)) if spec.pu else jnp.ones((K,), bool)  # stationary P(free)=0.5 (paper §VI-C)
    return State(h, pu_free, jnp.zeros((N,), jnp.int32), -jnp.ones((N,), jnp.int32),
                 jnp.zeros((N,), jnp.float32), jnp.zeros((N,), jnp.float32), -jnp.ones((N,), jnp.int32))


def rate_mbps(spec: EnvSpec, h):
    """Shannon rate B log2(1 + SNR |h|^2) in Mbps for fading gains h[...,2]; 1.0 per success if no fading (packet units)."""
    if not spec.fading:
        return jnp.ones(h.shape[:-1], jnp.float32)
    g2 = h[..., 0] ** 2 + h[..., 1] ** 2
    return spec.bw_hz / 1e6 * jnp.log2(1.0 + spec.snr_lin * g2)


def observe(spec: EnvSpec, world: World, state: State):
    """x_n(t) = [onehot(a_n(t-1)) over 0..K' | capacities of the K' channels | ACK o_n(t-1)]  (paper §IV-A.1)."""
    a1 = jax.nn.one_hot(state.prev_a, spec.k_local + 1)
    if spec.capacity == "const":
        cap = jnp.ones((spec.n_max, spec.k_local))
    else:
        r = rate_mbps(spec, state.h)                                   # [N,K]
        cap = jnp.take_along_axis(r, world.subset, axis=1) / C_ERGODIC_MBPS
    return jnp.concatenate([a1, cap, state.prev_ack[:, None]], axis=-1).astype(jnp.float32)


def all_rates(spec: EnvSpec, state: State):
    """Instantaneous rate of every user on every channel [N,K] (used by the OCA / DP baselines, genie CSI)."""
    return rate_mbps(spec, state.h)


def step_global(spec: EnvSpec, world: World, state: State, tx, g, key):
    """One slot given transmit flags tx[N] and GLOBAL channels g[N].  Returns (state', info)."""
    K = spec.n_channels
    tx = tx & world.active
    gc = jnp.where(tx, g, 0).astype(jnp.int32)
    count = jnp.zeros((K,), jnp.int32).at[gc].add(tx.astype(jnp.int32))   # SU transmitters per channel
    success = tx & (count[gc] == 1) & state.pu_free[gc]
    r_all = rate_mbps(spec, state.h)
    rate = jnp.where(success, jnp.take_along_axis(r_all, gc[:, None], axis=1)[:, 0], 0.0)
    s = success.astype(jnp.float32)
    kh, kp = jax.random.split(key)
    if spec.fading:
        rho = spec.rho
        w = jax.random.normal(kh, state.h.shape) / jnp.sqrt(2.0)
        h = rho * state.h + jnp.sqrt(1.0 - rho ** 2) * w                # complex AR(1), stationary CN(0,1) 
    else:
        h = state.h
    if spec.pu:
        stay = jax.random.bernoulli(kp, spec.pu_stay, (K,))             # symmetric chain, stationary 0.5 
        pu_free = jnp.where(stay, state.pu_free, ~state.pu_free)
    else:
        pu_free = state.pu_free
    new = State(h, pu_free, state.prev_a, jnp.where(tx, gc, -1), s, state.succ + s, state.choice)
    return new, dict(success=success, tx=tx, g=gc, rate=rate, count=count, pu_free=state.pu_free)


def step_local(spec: EnvSpec, world: World, state: State, a_local, key):
    """One slot for LOCAL actions a_local[N] in {0..K'} (0 = silent) -> global channel via the user's subset."""
    tx = (a_local > 0) & world.active
    g = jnp.take_along_axis(world.subset, jnp.clip(a_local - 1, 0, spec.k_local - 1)[:, None], axis=1)[:, 0]
    new, info = step_global(spec, world, state, tx, g, key)
    return new._replace(prev_a=jnp.where(tx, a_local, 0).astype(jnp.int32)), info


def world_keys(key, m: int):
    """m per-episode world keys, deterministic in (key, m)."""
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(m))


def split_world_key(wkey):
    """world_key -> (world key, init key, step-root key)."""
    k = jax.random.split(wkey, 3)
    return k[0], k[1], k[2]


def max_feasible(spec: EnvSpec, world: World, pu_free):
    """size of the maximum matching of ACTIVE users to PU-free channels they may use (one user per channel) = the largest
    number of simultaneous successes ANY policy that respects the channel subsets could obtain in a slot with this PU state.
    subset == "all": min(n_active, #free).  K' == 2 (random / fixed subsets): a user is an edge between its two channels (a
    user with one PU-blocked channel is a loop on the other, a user with both blocked is dropped); the best assignment of edges
    to distinct endpoints in a connected component with E edges and V vertices is min(E, V), so the matching size is the sum over
    components (components by boolean transitive closure of the K x K adjacency)."""
    K = spec.n_channels
    free = pu_free.astype(bool)
    if spec.subset == "all":
        return jnp.minimum(world.n_active, free.sum()).astype(jnp.int32)
    if spec.k_local != 2:
        raise NotImplementedError("max_feasible: K' == 2 or subset == 'all' only")
    a, b = world.subset[:, 0], world.subset[:, 1]
    la, lb, act = free[a], free[b], world.active
    both = act & la & lb
    only_a, only_b = act & la & ~lb, act & lb & ~la
    A = jnp.zeros((K, K), jnp.float32)
    A = A.at[a, b].add(both.astype(jnp.float32)).at[b, a].add(both.astype(jnp.float32))
    A = (A > 0) | jnp.eye(K, dtype=bool) & free[None, :] & free[:, None]
    R = A.astype(jnp.float32)
    for _ in range(int(np.ceil(np.log2(max(K, 2)))) + 1):
        R = ((R @ R) > 0).astype(jnp.float32)
    lab = jnp.argmax(R > 0, axis=1)                                   # smallest-index vertex of the component (live vertices)
    V = jnp.zeros((K,), jnp.int32).at[lab].add(free.astype(jnp.int32))
    ep = jnp.where(both | only_a, a, b)                               # live endpoint of every counted edge
    cnt = (both | only_a | only_b).astype(jnp.int32)
    E = jnp.zeros((K,), jnp.int32).at[lab[ep]].add(cnt)
    return jnp.minimum(V, E).sum().astype(jnp.int32)
