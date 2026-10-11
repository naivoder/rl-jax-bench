"""Input-driven simulators as pure, jittable, vmappable JAX functions.

Load balancing (paper §3 two-server toy, App. J ten servers)
------------------------------------------------------------
Jobs arrive at integer times ``t_k`` with integer sizes; at each arrival the agent picks a
server; each server is FIFO and work-conserving; reward between consecutive decisions is
``-(number of jobs in system) x (elapsed time)``; the episode ends after the last job has
been assigned *and drained* (the last step's reward integrates the drain -- official code /
Park).

Because servers are FIFO and work-conserving and a job's service time is fixed when it is
assigned, the official event-driven simulator collapses to a closed-form recursion:

    start_k = max(t_k, busy_until[a_k]) finish_k = start_k + floor(size_k / rate[a_k])
    busy_until[a_k] <- finish_k
    r_k = - sum_{j<=k} max(0, min(finish_j, t_{k+1}) - t_k) (t_{n}:= +inf: drain)

i.e. the jobs-in-system integral over [t_k, t_{k+1}) is the summed overlap of each job's
sojourn interval with that window. No per-server queue (and so no queue-capacity overflow)
exists. ``tests/test_env_reference.py`` checks this against an event-driven numpy
simulator with a priority queue (official semantics) on random input/action pairs.

Durations use exact integer arithmetic ``(size * rate_den) // rate_num`` with
``rate = rate_num / rate_den`` (rates 0.15..1.05 are multiples of 0.01): float32 division
would flip ``floor`` at exact multiples (150 / 0.15f = 999.99996).
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .data import LBInputs

BIG_T = jnp.int32(2_000_000_000)


class LBSpec(NamedTuple):
    n_servers: int
    n_jobs: int
    rate_num: tuple  # integers, rate = rate_num / rate_den
    rate_den: int
    obs_mode: str  # "work_queued" (official code), "work" (incl. in-service; Park) or "count" (§3)
    work_scale: float = 5000.0  # official: /job_size_norm_factor/5
    work_cap: float = 20.0
    job_scale: float = 1000.0
    job_cap: float = 10.0
    count_scale: float = 10.0  # ASSUMED: queue length / 10, no cap
    obs_tie: str = "strict"  # "strict" (pre-fix): queued = start > now.  "official" (recheck): also a job
    #                          that starts exactly now but arrived earlier, as in the authors' env, which pops an
    #                          arrival before a same-time completion so that job is still in worker.queue
    obs_job: bool = True  # append the incoming job size to the policy obs (App. J: (j, q_1..q_k)); paper §3's
    # two-server toy observes s_t = (q1, q2) only -> obs_job: false there

    @property
    def rates(self) -> jax.Array:
        return jnp.asarray(self.rate_num, jnp.float32) / float(self.rate_den)

    @property
    def obs_dim(self) -> int:
        return self.n_servers + int(self.obs_job)

    @property
    def critic_dim(self) -> int:
        return self.obs_dim + 1  # obs t/n_jobs


def make_lb_spec(cfg: dict) -> LBSpec:
    rates = np.asarray(cfg["service_rates"], np.float64)
    den = 100
    num = np.round(rates * den).astype(int)
    assert np.allclose(num / den, rates), "service rates must be multiples of 0.01"
    return LBSpec(n_servers=len(rates), n_jobs=int(cfg["jobs_per_episode"]), rate_num=tuple(int(x) for x in num),
                  rate_den=den, obs_mode=str(cfg.get("obs_mode", "work_queued")), obs_job=bool(cfg.get("obs_job", True)),
                  obs_tie=str(cfg.get("obs_tie", "strict")))


class LBState(NamedTuple):
    k: jax.Array  # int32, index of the incoming job
    busy_until: jax.Array  # int32 [S]
    finish: jax.Array  # int32 [n_jobs], 0 for unassigned jobs
    start: jax.Array  # int32 [n_jobs], service start time, 0 for unassigned jobs
    server_of: jax.Array  # int32 [n_jobs], -1 for unassigned


def lb_reset(spec: LBSpec) -> LBState:
    z = jnp.zeros((spec.n_jobs,), jnp.int32)
    return LBState(jnp.int32(0), jnp.zeros((spec.n_servers,), jnp.int32), z, z, -jnp.ones((spec.n_jobs,), jnp.int32))


def lb_durations(spec: LBSpec, size) -> jax.Array:
    """Service time of a job of ``size`` on every server, [S] int32 (exact floor(size/rate))."""
    num = jnp.asarray(spec.rate_num, jnp.int32)
    return (size.astype(jnp.int32) * spec.rate_den) // num


def lb_now(state: LBState, inputs: LBInputs):
    return inputs.arrivals[state.k]


def lb_queue_counts(spec: LBSpec, state: LBState, now) -> jax.Array:
    active = (state.finish > now) & (state.server_of >= 0)
    onehot = jax.nn.one_hot(state.server_of, spec.n_servers, dtype=jnp.int32)
    return jnp.sum(onehot * active[:, None].astype(jnp.int32), axis=0)


def lb_queued_work(spec: LBSpec, state: LBState, inputs: LBInputs, now) -> jax.Array:
    """Summed size of the jobs waiting (not yet started) on each server, [S] float32."""
    if spec.obs_tie == "official":
        waiting = (state.start > now) | ((state.start == now) & (inputs.arrivals < now))
    elif spec.obs_tie == "strict":
        waiting = state.start > now
    else:
        raise ValueError(spec.obs_tie)
    queued = waiting & (state.server_of >= 0)
    onehot = jax.nn.one_hot(state.server_of, spec.n_servers, dtype=jnp.float32)
    return jnp.sum(onehot * (queued * inputs.sizes.astype(jnp.float32))[:, None], axis=0)


def lb_obs(spec: LBSpec, state: LBState, inputs: LBInputs):
    """(policy observation, critic features).  Policy obs = [load_1..load_S, incoming job size]."""
    now = lb_now(state, inputs)
    size = inputs.sizes[state.k].astype(jnp.float32)
    if spec.obs_mode == "work_queued":  # official code: queued work only, in-service job excluded
        load = jnp.minimum(lb_queued_work(spec, state, inputs, now) / spec.work_scale, spec.work_cap)
    elif spec.obs_mode == "work":  # sensitivity: queued + remaining in-service work (Park)
        remaining = jnp.maximum(state.busy_until - now, 0).astype(jnp.float32)
        load = jnp.minimum(remaining * spec.rates / spec.work_scale, spec.work_cap)  #
    elif spec.obs_mode == "count":
        load = lb_queue_counts(spec, state, now).astype(jnp.float32) / spec.count_scale
    else:
        raise ValueError(spec.obs_mode)
    job = jnp.minimum(size / spec.job_scale, spec.job_cap)
    obs = jnp.concatenate([load, job[None]]) if spec.obs_job else load
    tfrac = state.k.astype(jnp.float32) / spec.n_jobs
    return obs, jnp.concatenate([obs, tfrac[None]])


def lb_step(spec: LBSpec, state: LBState, action, inputs: LBInputs):
    """Assign the incoming job to server ``action``; advance to the next arrival (or drain).
    Returns (state', raw reward (float32, unscaled), done)."""
    k = state.k
    now = inputs.arrivals[k]
    dur = lb_durations(spec, inputs.sizes[k])[action]
    start = jnp.maximum(now, state.busy_until[action])
    fin = start + dur
    busy = state.busy_until.at[action].set(fin)
    finish = state.finish.at[k].set(fin)
    start_arr = state.start.at[k].set(start)
    server_of = state.server_of.at[k].set(action.astype(jnp.int32))
    last = k + 1 >= spec.n_jobs
    t_next = jnp.where(last, BIG_T, inputs.arrivals[jnp.minimum(k + 1, spec.n_jobs - 1)])
    overlap = jnp.maximum(jnp.minimum(finish, t_next) - now, 0)  # unassigned jobs have finish=0 -> 0
    reward = -jnp.sum(overlap.astype(jnp.float32))
    return LBState(k + 1, busy, finish, start_arr, server_of), reward, last


class Traj(NamedTuple):
    obs: jax.Array  # [T, obs_dim]
    cobs: jax.Array  # [T, critic_dim]
    actions: jax.Array  # [T] int32
    rewards: jax.Array  # [T] raw
    logp: jax.Array  # [T] log pi(a_t|obs_t) at sampling time


def lb_rollout(spec: LBSpec, policy_logits: Callable, inputs: LBInputs, key, greedy: bool = False) -> Traj:
    """One episode (exactly n_jobs decisions) of a stochastic policy on a fixed input sequence."""
    keys = jax.random.split(key, spec.n_jobs)

    def body(state, k):
        obs, cobs = lb_obs(spec, state, inputs)
        logits = policy_logits(obs)
        logp_all = jax.nn.log_softmax(logits)
        a = jnp.argmax(logits) if greedy else jax.random.categorical(k, logits)
        a = a.astype(jnp.int32)
        state2, r, _ = lb_step(spec, state, a, inputs)
        return state2, (obs, cobs, a, r, logp_all[a])

    _, (obs, cobs, a, r, lp) = jax.lax.scan(body, lb_reset(spec), keys)
    return Traj(obs, cobs, a, r, lp)


# ---------------------------------------------------------------- heuristics
# Recheck: the authors' tie convention (first index) vs ours (faster server).  The old names stay valid.
FIRST_INDEX_RULES = ("spt_official", "lw_official", "jsq_first")
HEURISTIC_ALIASES = {"spt_fastest_tie": "spt", "jsq_fastest_tie": "jsq"}


def lb_heuristic_action(spec: LBSpec, name: str, state: LBState, inputs: LBInputs):
    now = lb_now(state, inputs)
    wait = jnp.maximum(state.busy_until - now, 0).astype(jnp.float32)
    rates = spec.rates
    name = HEURISTIC_ALIASES.get(name, name)
    if name in FIRST_INDEX_RULES:
        # the authors' agents (load_balance/heuristic_agents.py) scan servers from index 0 with a strict '<', so a tie
        # goes to the LOWEST index = the SLOWEST server (rate 0.15); jnp.argmin returns the first minimum, the same rule
        if name == "spt_official":  # ShortestProcessingTimeAgent: queued work (in-service excluded) / rate
            score = lb_queued_work(spec, state, inputs, now) / rates
        elif name == "lw_official":  # LeastWorkAgent: queued work (in-service excluded)
            score = lb_queued_work(spec, state, inputs, now)
        else:  # jsq_first: App. J's join-the-shortest-queue (jobs at the server incl. the one in service), first index
            score = lb_queue_counts(spec, state, now).astype(jnp.float32)
        return jnp.argmin(score).astype(jnp.int32)
    if name == "spt":  # = spt_fastest_tie: queued work (in-service excluded) / rate, ties -> FASTER server (ours, NOT
        #  the authors' tie rule; Recheck:)
        score = lb_queued_work(spec, state, inputs, now) / rates
    elif name == "lrt":  # least remaining time: argmin time until the server is free (incl. in-service work)
        score = wait
    elif name == "eft":  # earliest finish time of the incoming job
        score = wait + lb_durations(spec, inputs.sizes[state.k]).astype(jnp.float32)
    elif name == "jsq":  # = jsq_fastest_tie: join the shortest queue by job count; ties -> faster server (ours)
        score = lb_queue_counts(spec, state, now).astype(jnp.float32) - 1e-3 * rates
    elif name == "fastest":  # always the fastest server (the tie-break below decides); formerly mislabeled "random"
        score = jnp.zeros_like(wait)
    else:
        raise ValueError(name)
    return jnp.argmin(score - 1e-6 * rates).astype(jnp.int32)  # ties -> faster server


def lb_heuristic_rollout(spec: LBSpec, name: str, inputs: LBInputs):
    def body(state, _):
        a = lb_heuristic_action(spec, name, state, inputs)
        state2, r, _ = lb_step(spec, state, a, inputs)
        return state2, r

    _, r = jax.lax.scan(body, lb_reset(spec), None, length=spec.n_jobs)
    return r


def lb_actions_rollout(spec: LBSpec, actions, inputs: LBInputs):
    """Replay a fixed action list (used by the reference-simulator test)."""
    def body(state, a):
        state2, r, d = lb_step(spec, state, a, inputs)
        return state2, (r, d)

    _, (r, d) = jax.lax.scan(body, lb_reset(spec), actions)
    return r, d


# ---------------------------------------------------------------- reference event-driven simulator (numpy)
def reference_event_sim(arrivals, sizes, actions, rate_num, rate_den):
    """Event-driven FIFO simulator with a priority queue, written independently of the closed
    form (official env.step semantics).  Returns per-step rewards and done flags."""
    import heapq
    import itertools

    n = len(arrivals)
    S = len(rate_num)
    queues = [[] for _ in range(S)]
    curr = [None] * S  # (finish_time, job_id)
    cnt = itertools.count()
    tl = []
    for t, j in zip(arrivals, range(n)):
        heapq.heappush(tl, (int(t), next(cnt), ("arr", j)))
    now = 0
    _, _, ev = heapq.heappop(tl)
    now = int(arrivals[0])
    incoming = ev[1]
    rewards, dones = [], []

    def process(s):
        if (curr[s] is None or curr[s][0] <= now) and queues[s]:
            j = queues[s].pop(0)
            dur = (int(sizes[j]) * rate_den) // int(rate_num[s])
            curr[s] = (now + dur, j)
            heapq.heappush(tl, (now + dur, next(cnt), ("fin", s, j)))

    for step in range(n):
        a = int(actions[step])
        queues[a].append(incoming)
        process(a)
        incoming = None
        rew = 0
        while tl:
            t, _, ev = heapq.heappop(tl)
            active = sum(len(q) for q in queues) + sum(1 for s in range(S) if curr[s] is not None)
            rew -= (t - now) * active
            now = t
            if ev[0] == "arr":
                incoming = ev[1]
                break
            s = ev[1]
            if curr[s] is not None and curr[s][1] == ev[2]:
                curr[s] = None
            process(s)
        rewards.append(rew)
        dones.append(not tl and incoming is None)
    return np.asarray(rewards, np.int64), np.asarray(dones)


# ---------------------------------------------------------------- App. A 1-D toy
def toy_step(s, a, z):
    """s' = s + a + z, r = a + z (paper App. A)."""
    return s + a + z, a + z
