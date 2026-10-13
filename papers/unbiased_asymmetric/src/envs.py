"""Pure-JAX environments. Categorical POMDPs from the parsed gym-pomdps files (data.py).

Semantics = gym-pomdps@b52e520 (episodic) + asym-rlpo FlatPaddingWrapper + gym TimeLimit(100), checked step by step
against the official code by tests/test_official_env.py:
  reset: s0 ~ start; first observation = the extra reset observation (index n_obs).
  step(s, a): s' ~ T[s, a]; o ~ O[a, s']; r = R[s, a]; done = D[s, a]  (time limit handled by the rollout).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from data import CategoricalPOMDP, load_pomdp


class CatEnv(NamedTuple):
    logT: jnp.ndarray      # [S, A, S'] log-probs
    logO: jnp.ndarray      # [A, S', O] log-probs
    R: jnp.ndarray         # [S, A]
    D: jnp.ndarray         # [S, A] bool
    log_start: jnp.ndarray  # [S]


def _log(p):
    with np.errstate(divide="ignore"):
        return jnp.asarray(np.log(p), jnp.float32)


def make_cat_env(p: CategoricalPOMDP) -> CatEnv:
    return CatEnv(_log(p.T), _log(p.O), jnp.asarray(p.R, jnp.float32), jnp.asarray(p.D), _log(p.start))


def cat_reset(env: CatEnv, key, n_obs: int):
    s = jax.random.categorical(key, env.log_start)
    return s.astype(jnp.int32), jnp.int32(n_obs)


def cat_step(env: CatEnv, key, s, a):
    k1, k2 = jax.random.split(key)
    s2 = jax.random.categorical(k1, env.logT[s, a]).astype(jnp.int32)
    o = jax.random.categorical(k2, env.logO[a, s2]).astype(jnp.int32)
    return s2, o, env.R[s, a], env.D[s, a]


class EnvFns(NamedTuple):
    """Uniform interface used by the trainer: reset(key)->(s,o); step(key,s,a)->(s',o,r,done)."""
    reset: callable
    step: callable
    kind: str
    n_actions: int
    n_obs: int
    n_states: int
    gamma: float


def make_env(name: str) -> EnvFns:
    if name in ("heavenhell_3", "heavenhell_4", "shopping_5", "shopping_6"):
        p = load_pomdp(name)
        env = make_cat_env(p)
        return EnvFns(reset=lambda k: cat_reset(env, k, p.n_obs), step=lambda k, s, a: cat_step(env, k, s, a),
                      kind="categorical", n_actions=p.n_actions, n_obs=p.n_obs, n_states=p.n_states,
                      gamma=p.discount)
    if name == "car_flag":
        return make_carflag()
    if name == "cleaner":
        return make_cleaner()
    raise KeyError(f"unknown env {name}")


# ----------------------------------------------------------------------------- Fig. 4 probes (Heaven-Hell)
def _bfs_path(nxt, src, dst_set, avoid=frozenset()):
    """Shortest action path on a deterministic transition table nxt[s, a] -> s'."""
    from collections import deque
    prev = {src: None}
    q = deque([src])
    while q:
        s = q.popleft()
        if s in dst_set:
            path = []
            while prev[s] is not None:
                ps, a = prev[s]
                path.append(a)
                s = ps
            return path[::-1]
        for a in range(nxt.shape[1]):
            s2 = int(nxt[s, a])
            if s2 not in prev and s2 not in avoid:
                prev[s2] = (s, a)
                q.append(s2)
    raise ValueError("unreachable")


def heavenhell_probes(name: str):
    """The four (history, state) pairs of paper Fig. 4 (ASSUMED construction, our construction):
    for each start (heaven-left map, heaven-right map): (no priest) shortest start->fork path; (priest)
    start->priest->fork. The fork is the last state shared by the shortest paths to the two exits.
    Order: (a) map0 no-priest, (b) map1 no-priest, (c) map0 priest, (d) map1 priest."""
    p = load_pomdp(name)
    live = ~p.D                                           # terminal rows hold the start distribution: skip them
    assert np.all((p.T[live] == 0) | (p.T[live] == 1)) and np.all((p.O == 0) | (p.O == 1))
    nxt = np.where(live, p.T.argmax(-1), np.arange(p.n_states)[:, None])   # [S, A]; terminal -> self
    obs_of = p.O.argmax(-1)                               # [A, S']
    starts = list(np.nonzero(p.start)[0])
    pri_obs = {p.observations.index("left"), p.observations.index("right")}
    seqs = []
    for s0 in starts:
        exits = [s for s in range(p.n_states) if p.D[s].all()]
        reach = lambda path: _states(nxt, s0, path)       # noqa: E731
        ex = [e for e in exits if _reachable(nxt, s0, e)]
        heaven = [e for e in ex if p.R[e].max() > 0][0]
        hell = [e for e in ex if p.R[e].min() < 0][0]
        ph, pl = reach(_bfs_path(nxt, s0, {heaven})), reach(_bfs_path(nxt, s0, {hell}))
        fork = [s for s, s2 in zip(ph, pl) if s == s2][-1]
        priest = [s for s in range(p.n_states) if _reachable(nxt, s0, s)
                  and int(obs_of[0, s]) in pri_obs][0]
        seqs.append((s0, _bfs_path(nxt, s0, {fork}), fork))
        seqs.append((s0, _bfs_path(nxt, s0, {priest}) + _bfs_path(nxt, priest, {fork}), fork))
    order = [seqs[0], seqs[2], seqs[1], seqs[3]]          # (a) (b) (c) (d)
    Lp = max(len(x[1]) for x in order) + 1
    A = np.zeros((4, Lp), np.int32)
    O = np.full((4, Lp), p.n_obs, np.int32)
    S = np.zeros((4, Lp), np.int32)
    idx = np.zeros(4, np.int32)
    for i, (s0, path, fork) in enumerate(order):
        st = _states(nxt, s0, path)
        A[i, :len(path)] = path
        S[i, :len(st)] = st
        for t, a in enumerate(path):   # o_{t+1} = O[a_t, s_{t+1}]
            O[i, t + 1] = obs_of[a, st[t + 1]]
        idx[i] = len(path)
        assert st[-1] == fork
    return dict(actions=A, obs=O, states=S, idx=idx)


def _states(nxt, s0, path):
    st = [int(s0)]
    for a in path:
        st.append(int(nxt[st[-1], a]))
    return st


def _reachable(nxt, s0, dst):
    try:
        _bfs_path(nxt, s0, {dst})
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------------- Car-Flag (App. C.2.1)
# JAX re-implementation of asym-rlpo@1a065b0 envs/extra_hai/car_flag.py (CarEnv + CarEnvWrapper(num_actions=7)),
# checked step by step against the official env in float64 (tests/test_official_carflag.py).
CF = dict(power=0.0015, max_speed=0.07, max_pos=1.1, priest=0.5, priest_delta=0.2, n_actions=7)  # official CarEnv


def carflag_reset(key, dtype=jnp.float32):
    k1, k2 = jax.random.split(key)
    heaven = jnp.where(jax.random.bernoulli(k1), -1.0, 1.0).astype(dtype)     # official: randint(2) == 0 -> +1
    x = jax.random.uniform(k2, (), dtype, -0.2, 0.2)
    state = jnp.stack([x, jnp.zeros((), dtype), heaven])                      # (position, velocity, heaven)
    obs = jnp.stack([x, jnp.zeros((), dtype), jnp.zeros((), dtype)])           # (position, velocity, direction)
    return state, obs


def carflag_step(key, state, a):
    del key                                                                   # deterministic given the action
    x, v, heaven = state[0], state[1], state[2]
    force = jnp.linspace(-1.0, 1.0, CF["n_actions"], dtype=state.dtype)[a]
    v = jnp.clip(v + force * CF["power"], -CF["max_speed"], CF["max_speed"])
    x = jnp.clip(x + v, -CF["max_pos"], CF["max_pos"])
    v = jnp.where((x == -CF["max_pos"]) & (v < 0), 0.0, v)
    done = (x >= 1.0) | (x <= -1.0)                                           # flags at +-1 (heaven / hell)
    r = jnp.where(heaven > 0, jnp.where(x >= 1.0, 1.0, jnp.where(x <= -1.0, -1.0, 0.0)),
                  jnp.where(x <= -1.0, 1.0, jnp.where(x >= 1.0, -1.0, 0.0)))
    in_priest = (x >= CF["priest"] - CF["priest_delta"]) & (x <= CF["priest"] + CF["priest_delta"])
    direction = jnp.where(in_priest, heaven, 0.0)
    s2 = jnp.stack([x, v, heaven]).astype(state.dtype)
    return s2, jnp.stack([x, v, direction]).astype(state.dtype), r.astype(state.dtype), done


def make_carflag() -> EnvFns:
    return EnvFns(reset=carflag_reset, step=carflag_step, kind="vector", n_actions=CF["n_actions"], n_obs=3,
                  n_states=3, gamma=0.99)   # gamma: official --training-discount default (Table 1 lists none)


# ----------------------------------------------------------------------------- Cleaner (App. C.2.2)
# JAX re-implementation of asym-rlpo@1a065b0 envs/extra_lyu/cleaner.py (EnvCleaner(seed=5) + EnvCleaner_Fix +
# SingleAgentWrapper): fixed 13x13 maze (exported once from the official generator as a checksummed fixture; the
# maze is data), 2 agents start at (1, 1), joint action a -> (a // 4, a % 4), moves up/down/left/right unless into a
# wall, agents move and clean sequentially (+1 per newly cleaned cell). The env state is the official 13x13x5 one-hot
# state vector (clean / wall / dirty / agent0 / agent1), which is also the critic's state input.
CLEANER_MAZE_SHA256 = "9484c9e9bd9666f6c9799a084ef8b183edba1a68d5dfd995d8ecc9ea8608e084"
_MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], np.int32)


def _cleaner_maze():
    import hashlib
    import os
    from data import FIXTURES
    path = os.path.join(FIXTURES, "cleaner_maze_seed5.npy")
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != CLEANER_MAZE_SHA256:
        raise RuntimeError("cleaner maze fixture SHA256 mismatch")
    return np.load(path).astype(np.int32)


def _cleaner_encode(occ, pos):
    st = jnp.stack([occ == 0, occ == 1, occ == 2], -1).astype(jnp.float32)
    a0 = jnp.zeros((13, 13), jnp.float32).at[pos[0, 0], pos[0, 1]].set(1.0)
    a1 = jnp.zeros((13, 13), jnp.float32).at[pos[1, 0], pos[1, 1]].set(1.0)
    return jnp.concatenate([st, a0[..., None], a1[..., None]], -1).reshape(-1)


def _cleaner_decode(s):
    g = s.reshape(13, 13, 5)
    occ = jnp.argmax(g[..., :3], -1).astype(jnp.int32)
    p0 = jnp.argmax(g[..., 3].reshape(-1))
    p1 = jnp.argmax(g[..., 4].reshape(-1))
    return occ, jnp.array([[p0 // 13, p0 % 13], [p1 // 13, p1 % 13]], jnp.int32)


def _cleaner_local(occ, me, mate):
    win = jax.lax.dynamic_slice(occ, (me[0] - 1, me[1] - 1), (3, 3))
    obs = jnp.where((win == 0)[..., None], jnp.array([1.0, 1.0, 1.0]),
                    jnp.where((win == 2)[..., None], jnp.array([0.0, 1.0, 0.0]), 0.0))
    d = mate - me
    near = jnp.all(jnp.abs(d) <= 1)
    obs = jnp.where(near, obs.at[1 + d[0], 1 + d[1]].set(jnp.array([0.0, 0.0, 1.0])), obs)
    obs = obs.at[1, 1].set(jnp.array([1.0, 0.0, 0.0]))
    return obs.reshape(-1)


def _cleaner_obs(occ, pos):
    return jnp.concatenate([_cleaner_local(occ, pos[0], pos[1]), _cleaner_local(occ, pos[1], pos[0])])


def make_cleaner() -> EnvFns:
    maze = jnp.asarray(_cleaner_maze())
    start = jnp.array([[1, 1], [1, 1]], jnp.int32)

    def reset(key):
        del key
        return _cleaner_encode(maze, start), _cleaner_obs(maze, start)

    def step(key, s, a):
        del key
        occ, pos = _cleaner_decode(s)
        r = jnp.float32(0.0)
        for i, ai in enumerate((a // 4, a % 4)):
            tgt = pos[i] + jnp.asarray(_MOVES)[ai]
            pos = pos.at[i].set(jnp.where(occ[tgt[0], tgt[1]] != 1, tgt, pos[i]))
            dirty = occ[pos[i, 0], pos[i, 1]] == 2
            occ = occ.at[pos[i, 0], pos[i, 1]].set(jnp.where(dirty, 0, occ[pos[i, 0], pos[i, 1]]))
            r = r + dirty.astype(jnp.float32)
        return _cleaner_encode(occ, pos), _cleaner_obs(occ, pos), r, jnp.bool_(False)   # internal 200 > cap 100

    return EnvFns(reset=reset, step=step, kind="vector", n_actions=16, n_obs=54, n_states=845, gamma=0.99)
