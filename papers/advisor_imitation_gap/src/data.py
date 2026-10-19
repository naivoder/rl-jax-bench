"""Environment registry. The paper's tasks are simulated, so 'data' is the on-line experience of the pure-JAX envs
in src/envs/; there is no dataset. Every env exposes the same pure-function interface, with a per-run parameter dict
``ep`` (traced arrays, so runs with different view radii / teachers vmap in one program):

    reset(key, ep)            -> state
    step(state, action, ep)   -> state, reward, done
    obs(state, ep)            -> network input (PD: int; LH: active-column list [13]; MG: image [7,7,3])
    teacher(state, ep)        -> int label (privileged or public teacher, chosen by ep)
    info(state)               -> dict of per-episode metrics (on the terminal state)
    eval_reset(k, key, ep)    -> state of the k-th fixed evaluation task
    next_episode(s, fresh, ep)-> (state after an episode ends, used_fresh) (MiniGrid repeats failed layouts)
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .envs import lighthouse as LH
from .envs import pd as PD


class EnvSpec(NamedTuple):
    name: str
    n_actions: int
    max_steps: int
    reset: Callable
    step: Callable
    obs: Callable
    teacher: Callable
    info: Callable
    eval_reset: Callable
    next_episode: Callable = None   # (state_at_done, fresh_state, ep) -> (next state, consumed_fresh)
    pool_size: int = 0              # > 0: fresh states come from a per-update pool of this many per env


# ------------------------------------------------------------------ Poisoned Doors
def _pd_teacher(s, ep):
    n_wrong = ep["n_wrong"] if "n_wrong" in ep else 1      # public teacher's wrong final digits (extension)
    return jnp.where(ep["public"], PD.teacher_public(s, n_wrong), PD.teacher_privileged(s))


PD_SPEC = EnvSpec(
    name="pd", n_actions=PD.N_ACTIONS, max_steps=PD.MAX_STEPS,
    reset=lambda key, ep: PD.reset(key),
    step=lambda s, a, ep: PD.step(s, a),
    obs=lambda s, ep: PD.obs(s),
    teacher=_pd_teacher,
    info=PD.episode_info,
    # official test sampler (good door of task k = 1 + (2^31 - 1 + k) % 3), as a host-built table: the traced int32
    # expression used in an earlier version overflowed for k >= 1
    eval_reset=lambda k, key, ep: PD.reset(key, good=PD.eval_good(k)),
)


# ------------------------------------------------------------------ 2D Lighthouse
_LH_ACTIVE = None


def lh_active_table():
    global _LH_ACTIVE
    if _LH_ACTIVE is None:
        _LH_ACTIVE = LH.design_active_columns().astype(np.int32)   # numpy: becomes a jit constant (no tracer leak)
    return jnp.asarray(_LH_ACTIVE)


_LH_TEST_GOALS = None


def lh_test_goal(k):
    global _LH_TEST_GOALS
    if _LH_TEST_GOALS is None:
        import json
        import os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "lh_official_test_goals.json")
        _LH_TEST_GOALS = np.asarray(json.load(open(p))["goals"], np.int32)
    tab = jnp.asarray(_LH_TEST_GOALS)
    return jnp.where(k < tab.shape[0], tab[jnp.minimum(k, tab.shape[0] - 1)], k % 4)


def _lh_obs(s, ep):
    return lh_active_table()[LH.tuple_index(LH.corner_obs(s, ep["i"]))]


LH_SPEC = EnvSpec(
    name="lh", n_actions=LH.N_ACTIONS, max_steps=LH.MAX_STEPS,
    reset=lambda key, ep: LH.reset(key),
    step=lambda s, a, ep: LH.step(s, a),
    obs=_lh_obs,
    teacher=lambda s, ep: LH.teacher(s, ep["j"]),
    info=LH.episode_info,
    # official test tasks: the k-th of the 200 test episodes uses the goal that the authors'
    # seeded test sampler draws for seed 2^31-1+k (src/configs/lh_official_test_goals.json, computed in the official
    # venv). Counts 50/51/57/42 per corner, NOT balanced. An earlier version ASSUMED balanced goals k % 4.
    eval_reset=lambda k, key, ep: LH.reset(key, goal=lh_test_goal(k)),
)


def get_spec(name: str) -> EnvSpec:
    if name == "pd":
        return PD_SPEC
    if name == "lh":
        return LH_SPEC
    if name == "mg":
        from .envs import crossing as MG
        return MG.SPEC
    raise KeyError(name)
