"""Poisoned Doors (paper Ex. 1, App. A.5.1) as a pure-JAX environment.

Semantics follow the official env (allenai/advisor @ 6849755, poisoneddoors_plugin/poisoneddoors_tasks.py), as follows:
  * 7 actions in the order (c0, c1, c2, d_locked, d1, d2, d3)                               [official code]
  * state enum obs: 0 choosing door, 1 entering code (start), 2 entering code (cont), 3 done  [App. A.5.10 + official]
  * a digit at the door state is a no-op (reward 0, step consumed)                            [official code]
  * in a code state ANY action that is not the next code digit (doors included) ends with 0    [official code]
  * correct 10th digit: +1; good door +2; other doors -2                                        [paper Ex. 1]
  * max 100 steps (done at the cap)                                                             [official PD_MAX_STEPS]
  * code = CPython random.seed(10); [randint(0, 2) for _ in range(10)] = [2,0,1,1,2,0,0,1,1,1]  [official get_combination]
  * good door uniform over {1,2,3} (official: 1 + seed % 3)                                     [paper Ex. 1 + official]
Teachers: 'privileged' = official query_expert (3 + good door, then the correct digit);
          'public'     = : choose d_locked, then the code with the LAST digit wrong (return 0).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

N_ACTIONS = 7          # official code (3 digits + 4 doors); paper App. A.8 head 128x7
N_OBS = 4              # state enum; App. A.8 Embedding(4 -> 128)
MAX_STEPS = 100        # official code PD_MAX_STEPS
CODE = (2, 0, 1, 1, 2, 0, 0, 1, 1, 1)  # official get_combination (CPython MT, seed 10)
CODE_LEN = 10          # App. A.5.1
LOCKED = 3             # action index of d_locked (official order)


class State(NamedTuple):
    phase: jax.Array      # int32 in {0,1,2,3}
    chosen: jax.Array     # int32: -1 none, 0 locked door, 1..3 other doors
    idx: jax.Array        # int32: digits entered correctly so far
    max_idx: jax.Array    # int32
    good: jax.Array       # int32 in {1,2,3}
    t: jax.Array          # int32 steps taken
    ret: jax.Array        # float32 episode return so far


def reset(key, good=None) -> State:
    if good is None:
        good = 1 + jax.random.randint(key, (), 0, 3)
    z = jnp.zeros((), jnp.int32)
    return State(z, z - 1, z, z, jnp.asarray(good, jnp.int32), z, jnp.zeros((), jnp.float32))


def obs(s: State):
    return s.phase


def step(s: State, a):
    code = jnp.asarray(CODE, jnp.int32)
    a = jnp.asarray(a, jnp.int32)
    code_branch = (a < 3) | (s.phase != 0)
    noop = code_branch & (s.chosen < 0)
    entering = code_branch & (s.chosen >= 0)
    correct = code[jnp.clip(s.idx, 0, CODE_LEN - 1)] == a
    idx_e = jnp.where(correct, s.idx + 1, 0)
    opened = entering & correct & (idx_e >= CODE_LEN)
    phase_e = jnp.where(correct, 2, 3)
    phase_e = jnp.where(idx_e >= CODE_LEN, 3, phase_e)
    pick_locked = (~code_branch) & (a == LOCKED)
    pick_door = (~code_branch) & (a > LOCKED)
    door_r = jnp.where(s.good == a - LOCKED, 2.0, -2.0)

    phase = jnp.where(noop, s.phase, jnp.where(entering, phase_e, jnp.where(pick_locked, 1, 3)))
    chosen = jnp.where(pick_locked, 0, jnp.where(pick_door, a - LOCKED, s.chosen))
    idx = jnp.where(entering, idx_e, jnp.where(pick_locked, 0, s.idx))
    max_idx = jnp.maximum(s.max_idx, idx)
    r = jnp.where(opened, 1.0, 0.0) + jnp.where(pick_door, door_r, 0.0)
    t = s.t + 1
    ns = State(phase.astype(jnp.int32), chosen.astype(jnp.int32), idx.astype(jnp.int32), max_idx.astype(jnp.int32),
               s.good, t, s.ret + r)
    done = (phase == 3) | (t >= MAX_STEPS)
    return ns, r.astype(jnp.float32), done


def teacher_privileged(s: State):
    code = jnp.asarray(CODE, jnp.int32)
    return jnp.where(s.phase == 0, LOCKED + s.good, code[jnp.clip(s.idx, 0, CODE_LEN - 1)]).astype(jnp.int32)


def teacher_public(s: State, n_wrong=1):
    """Public teacher (extension, not in the paper): d_locked, then the code with the LAST n_wrong digits replaced by (digit+1) % 3 (n_wrong = 1 is the
    default; 2 and 3 move the teacher further from optimal). Reads only (phase, #digits). Return 0."""
    code = jnp.asarray(CODE, jnp.int32)
    d = code[jnp.clip(s.idx, 0, CODE_LEN - 1)]
    d = jnp.where(s.idx >= CODE_LEN - n_wrong, (d + 1) % 3, d)
    return jnp.where(s.phase == 0, LOCKED, d).astype(jnp.int32)


def episode_info(s: State):
    """Per-episode metrics (official PoisonedDoorsTask.metrics) evaluated on the terminal state."""
    return {
        "ret": s.ret,
        "len": s.t.astype(jnp.float32),
        "chose_locked": (s.chosen == 0).astype(jnp.float32),
        "chose_good": (s.chosen == s.good).astype(jnp.float32),
        "opened_lock": (s.max_idx >= CODE_LEN).astype(jnp.float32),
        "chose_none": (s.chosen < 0).astype(jnp.float32),
    }


def eval_task(k: int):
    """Official test sampler: seed 2**31 - 1 + k, good door = 1 + seed % 3 (Python ints, no overflow)."""
    return 1 + ((2 ** 31 - 1 + k) % 3)


def eval_doors(n: int):
    """Good door of evaluation tasks 0..n-1 as a host-side int64 table (becomes a jit constant). In an earlier version the
    expression above was evaluated on a TRACED int32 k, which wraps for k >= 1 (2**31 - 1 + k), so 999 of 1000 tasks
    had a different good door than the official sampler (same per-door counts)."""
    import numpy as np
    return np.array([eval_task(k) for k in range(n)], np.int64).astype(np.int32)


def eval_good(k, n_max: int = 4096):
    """Traced lookup of eval_task(k) for k < n_max."""
    return jnp.asarray(eval_doors(n_max))[k]
