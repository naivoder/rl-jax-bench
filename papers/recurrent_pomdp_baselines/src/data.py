"""Episode replay buffer on device, sub-trajectory / transition sampling, and the PRNG key tree.

The paper has no dataset; all data is generated online by the agent (this module is the replay store).

Buffer (per replication): whole episodes, padded to the env horizon H:
    obs (N, H+1, O), act (N, H, A), rew (N, H, 1), term (N, H, 1), length (N,), count ()
Capacity N = total episodes of the run (the official capacity never binds for these runs: Pendulum 51k << 1e6
transitions, Semi-Circle 1.55M < 4e6), so no overwrite logic is needed.

Capacity: the authors' run configs (Semi-Circle 4e6) are followed; the PRINTED Table 5 says 1e6, which a 1.548M-step
Semi-Circle run would overflow (deviation from the printed paper, sources.yaml).

Sequence sampling == the official SeqReplayBuffer (vanilla; the buffer the public Pendulum / Semi-Circle configs
instantiate, ``buffer_type`` unset; tests/test_official_parity.py) AND the official RAMEfficient_SeqReplayBuffer
(the App. A.1 buffer; tests/test_parity2.py), both at sample_weight_baseline 0: a start index uniform over
all valid starts in the buffer, where an episode of n transitions has max(1, n-L+1) valid starts; zero initial
RNN state, no burn-in; mask = 0 from the first slot past the episode end.  Implemented as episode ~ #valid
starts, then start ~ U{0..#starts-1} (same joint distribution).
Transition sampling (Markov baseline, official SimpleReplayBuffer.random_batch): uniform over stored transitions.

PRNG key tree (README): one top-level ``seed_base``; replication r with seed index s:
    run   = fold_in(PRNGKey(seed_base), s)            # arm is NOT folded in -> arms are seed-paired
    task  = fold_in(run, 0)   task set (goals / wind split)
    env   = fold_in(run, 1)   -> fold_in(episode_idx) -> fold_in(t): resets, task choice, inner resets
    init  = fold_in(run, 2)   network initialisation
    expl  = fold_in(run, 3)   -> fold_in(episode_idx) -> fold_in(t): exploration noise / warm-up actions
    repl  = fold_in(run, 4)   -> fold_in(update_idx): sub-trajectory sampling
    upd   = fold_in(run, 5)   -> fold_in(update_idx): target-smoothing / SAC reparameterisation noise
    eval  = fold_in(run, 6)   -> fold_in(eval_idx) -> fold_in(episode): evaluation resets (+ policy, deterministic)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

F32 = jnp.float32
K_TASK, K_ENV, K_INIT, K_EXPL, K_REPL, K_UPD, K_EVAL = range(7)


def run_key(seed_base: int, seed_idx):
    return jax.random.fold_in(jax.random.PRNGKey(seed_base), seed_idx)


def sub(key, which: int):
    return jax.random.fold_in(key, which)


def init_buffer(n_ep: int, H: int, O: int, A: int):
    return {"obs": jnp.zeros((n_ep, H + 1, O), F32), "act": jnp.zeros((n_ep, H, A), F32),
            "rew": jnp.zeros((n_ep, H, 1), F32), "term": jnp.zeros((n_ep, H, 1), F32),
            "length": jnp.zeros((n_ep,), jnp.int32), "count": jnp.zeros((), jnp.int32)}


def add_episodes(buf, ep):
    """ep: dict with leading axis n (obs (n,H+1,O), act, rew, term, length (n,))."""
    c = buf["count"]
    out = {}
    for k in ("obs", "act", "rew", "term", "length"):
        start = (c,) + (0,) * (buf[k].ndim - 1)
        out[k] = jax.lax.dynamic_update_slice(buf[k], ep[k].astype(buf[k].dtype), start)
    out["count"] = c + ep["length"].shape[0]
    return out


def sample_sequences(buf, key, B: int, L: int):
    """B sub-trajectories of L transitions (L+1 observations), time-major official layout (see algorithm.py)."""
    N = buf["act"].shape[0]
    k1, k2 = jax.random.split(key)
    nstarts = jnp.maximum(1, buf["length"] - L + 1)
    w = jnp.where(jnp.arange(N) < buf["count"], nstarts, 0).astype(F32)
    ep = jax.random.categorical(k1, jnp.log(w), shape=(B,))
    start = jax.random.randint(k2, (B,), 0, nstarts[ep])
    return gather_sequences(buf, ep, start, L)


def gather_sequences(buf, ep, start, L: int):
    """the sub-trajectories of L transitions starting at (episode ep[b], step start[b]); official layout."""
    H = buf["act"].shape[1]
    B = ep.shape[0]
    t = start[:, None] + jnp.arange(L + 1)[None, :]                      # (B, L+1)
    e = ep[:, None]
    obs = buf["obs"][e, jnp.clip(t, 0, H)]                               # (B, L+1, O)
    tt = jnp.clip(t[:, :L], 0, H - 1)
    act, rew, term = buf["act"][e, tt], buf["rew"][e, tt], buf["term"][e, tt]
    mask = (t[:, :L] < buf["length"][ep][:, None]).astype(F32)[..., None]   # (B, L, 1)
    act, rew, term = act * mask, rew * mask, term * mask

    def tm(x):
        return jnp.swapaxes(x, 0, 1)

    z = lambda x: jnp.zeros((1, B, x.shape[-1]), F32)  # noqa: E731
    act_t, rew_t, term_t = tm(act), tm(rew), tm(term)
    return {"obs": tm(obs), "act": jnp.concatenate([z(act_t), act_t]), "rew": jnp.concatenate([z(rew_t), rew_t]),
            "term": jnp.concatenate([z(term_t), term_t]), "mask": tm(mask)}


def sample_transitions(buf, key, B: int):
    N = buf["act"].shape[0]
    k1, k2 = jax.random.split(key)
    w = jnp.where(jnp.arange(N) < buf["count"], buf["length"], 0).astype(F32)
    ep = jax.random.categorical(k1, jnp.log(w), shape=(B,))
    t = jax.random.randint(k2, (B,), 0, buf["length"][ep])
    return {"obs": buf["obs"][ep, t], "obs2": buf["obs"][ep, t + 1], "act": buf["act"][ep, t],
            "rew": buf["rew"][ep, t], "term": buf["term"][ep, t]}
