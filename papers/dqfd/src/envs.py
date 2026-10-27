"""MinAtar (gymnax 1.0.0) thin wrapper: jittable, bool observations, explicit terminated/truncated.

Reduced-scale port: no frame stack, minimal action set, episode cap = gymnax default
(1000 steps, treated as TRUNCATION: bootstrapped; a natural game over is a TERMINATION).
"""
from __future__ import annotations

import gymnax
import jax
import jax.numpy as jnp

GAMES = {
    "Breakout": "Breakout-MinAtar",
    "SpaceInvaders": "SpaceInvaders-MinAtar",
    "Asterix": "Asterix-MinAtar",
    "Seaquest": "Seaquest-MinAtar",
    "Freeway": "Freeway-MinAtar",  # spare
}


def _fix_space_invaders(env):
    """Two deviations of gymnax 1.0.0 SpaceInvaders from the ORIGINAL MinAtar (kenjyoung/MinAtar 4241f246
    `space_invaders.py`), found by the differential test scripts/record_official_env.py and repaired here (our code is the one that must match the reference env):
      (1) reset: gymnax fills alien_map[0:4, 2:9] (7 columns, 28 aliens); the original (and gymnax's own wave respawn)
          use [0:4, 2:8] (6 columns, 24 aliens);
      (2) step_agent: the enemy-bullet hit test reads the cannon position BEFORE this step's move
          (`e_bullet_map[9, state.pos]`); the original tests the position AFTER the move (`self.pos` updated first).
    Idempotent; patches the gymnax module function that `step_env` looks up at call time."""
    from gymnax.environments.minatar import space_invaders as S
    if not getattr(S.step_agent, "_dqfd_fixed", False):
        orig = S.step_agent

        def step_agent_fixed(action, state, params):
            s2 = orig(action, state, params)
            # `orig` set terminal |= e_bullet_map[9, OLD pos]; redo the test with the NEW pos, keep earlier terminals
            return s2.replace(terminal=state.terminal | s2.e_bullet_map[9, s2.pos].astype(bool))

        step_agent_fixed._dqfd_fixed = True
        S.step_agent = step_agent_fixed
    orig_reset = env.reset_env

    def reset_fixed(key, params):
        _, st = orig_reset(key, params)
        st = st.replace(alien_map=jnp.zeros((10, 10)).at[0:4, 2:8].set(True))
        return env.get_obs(st), st

    env.reset_env = reset_fixed
    return env


def asterix_spawn_repaired(key, state):
    """Original MinAtar (kenjyoung/MinAtar 4241f246 `asterix.py::_spawn_entity`) spawn, written over gymnax's state layout.

    gymnax 1.0.0 `spawn_entity` differs in two ways: `while_sample_slots` increments its index before
    the first check, so permutation position 0 is never examined; and when it finds no free slot, `step_env` still writes a
    row with free = 0 into slot `order[7]`, DELETING a live enemy or gold.  The original picks a slot uniformly over the
    free rows and does nothing when none is free.  Same key split and lr / gold draws as gymnax.
    Returns (entity_row, slot, any_free); the caller must leave the entities unchanged when any_free is False."""
    key_lr, key_gold, key_slot = jax.random.split(key, 3)
    lr = jax.random.choice(key_lr, jnp.array([1, 0]))
    is_gold = jax.random.choice(key_gold, jnp.array([1, 0]), p=jnp.array([1 / 3, 2 / 3]))
    freem = state.entities[:, 4] == 0
    any_free = freem.any()
    p = jnp.where(any_free, freem / jnp.maximum(freem.sum(), 1), jnp.ones(8) / 8)
    slot = jax.random.choice(key_slot, 8, p=p)
    ent = jnp.array([(1 - lr) * 9, slot + 1, lr, is_gold, 1])
    return ent, slot, any_free


def _asterix_pre_spawn(key, state):
    """Do this step's spawn with the repaired rule BEFORE gymnax `step_env` runs, then hand it a state whose spawn timer is
    already reset (spawn_speed >= 1), so gymnax's own (defective) spawn branch is not taken.  Identical to gymnax's step
    whenever a free slot exists and gymnax would have found the same one; the timer path is unchanged."""
    ent, slot, any_free = asterix_spawn_repaired(key, state)
    now = state.spawn_timer == 0
    entities = jnp.where(now & any_free, state.entities.at[slot].set(ent), state.entities)
    spawn_timer = jnp.where(now, state.spawn_speed, state.spawn_timer)
    return state.replace(entities=entities, spawn_timer=spawn_timer)


class MinAtar:
    """env.reset(key) -> (obs_bool, state); env.step(key, state, a) -> (obs2, state2, r_raw, term, trunc)."""

    FIXES = {"SpaceInvaders": "gymnax SI reset 7->6 alien columns; enemy-bullet hit test at the post-move cannon position",
             "Asterix (flag asterix_spawn_repair, default OFF = the env of the main runs)":
                 "spawn uniform over all free rows, nothing when none is free (gymnax skips perm position 0 and deletes slot order[7])"}

    def __init__(self, game: str, max_steps: int | None = None, upstream_fixes: bool = True, sticky_action_prob: float = 0.0,
                 asterix_spawn_repair: bool = False):
        # asterix_spawn_repair: default False keeps every shipped number bit for bit; it is
        # per instance (no module-global patch), recorded in the demo JSON and followed by src/train.py.
        # sticky_action_prob (default 0 = the env of the main runs, bit for bit): with this probability the previous
        # EFFECTIVE action is repeated, as in original MinAtar (default 0.1; gymnax 1.0.0 has no such parameter).  The state
        # then carries the last effective action: state = (gymnax_state, last_action).  The paper's 30 no-op starts (p.4) are not
        # time-scalable to MinAtar (a Breakout ball is lost after ~7 no-op steps), so sticky actions are the randomisation used.
        self.sticky = float(sticky_action_prob)
        self.game = game
        self.env, params = gymnax.make(GAMES[game])
        if upstream_fixes and game == "SpaceInvaders":
            self.env = _fix_space_invaders(self.env)
        if max_steps is not None:
            params = params.replace(max_steps_in_episode=max_steps)
        self.params = params
        self.obs_shape = tuple(self.env.obs_shape)
        self.obs_dim = int(jnp.prod(jnp.array(self.obs_shape)))
        self.n_actions = int(self.env.num_actions)
        self.max_steps = int(params.max_steps_in_episode)
        self.asterix_spawn_repair = bool(asterix_spawn_repair)
        if self.asterix_spawn_repair and game != "Asterix":
            raise ValueError("asterix_spawn_repair applies to Asterix only")
        self._check_truncation_semantics()

    def _check_truncation_semantics(self):
        """Guard: the installed gymnax's `step_env` done must be the NATURAL game over only (time cap
        excluded), else `trunc` below is dead and every cap becomes a termination (stock gymnax 1.0.0 MinAtar ORs the cap
        into is_terminal).  A live reset state stepped at time = max_steps - 1 must give term False and trunc True."""
        _, st = self.env.reset_env(jax.random.PRNGKey(0), self.params)
        st = st.replace(time=self.max_steps - 1)
        if self.sticky > 0:
            st = (st, jnp.zeros((), jnp.int32))
        _, _, _, term, trunc = self.step(jax.random.PRNGKey(1), st, 0)
        if bool(term) or not bool(trunc):
            raise RuntimeError(f"gymnax {self.game}: a step at the time cap gave term={bool(term)} trunc={bool(trunc)}; "
                               "expected term False / trunc True (the installed gymnax folds the cap into termination)")

    def reset(self, key):
        obs, state = self.env.reset_env(key, self.params)
        if self.sticky > 0:
            state = (state, jnp.zeros((), jnp.int32))
        return obs > 0.5, state

    def step(self, key, state, action):
        # step_env does NOT auto-reset, so the pre-reset next observation (needed for truncation bootstrap) survives.
        if self.sticky > 0:
            state, last_a = state
            stick = jax.random.uniform(jax.random.fold_in(key, 7)) < self.sticky
            action = jnp.where(stick, last_a, action).astype(jnp.int32)
        if self.asterix_spawn_repair:
            state = _asterix_pre_spawn(key, state)
        obs, state2, r, term, _ = self.env.step_env(key, state, action, self.params)
        term = term.astype(bool)
        trunc = self.env.is_truncated(state2, self.params) & ~term
        if self.sticky > 0:
            state2 = (state2, action)
        return obs > 0.5, state2, r.astype(jnp.float32), term, trunc


def reward_transform(r):
    """rho(r) = sign(r) log(1 + |r|)   (paper §Experimental Setup)."""
    return jnp.sign(r) * jnp.log1p(jnp.abs(r))


def pack_obs(obs_bool):
    """(..., H, W, C) bool -> (..., ceil(HWC/8)) uint8 (replay storage is bit-packed)."""
    flat = obs_bool.reshape(obs_bool.shape[:-3] + (-1,)).astype(jnp.uint8)
    return jnp.packbits(flat, axis=-1)


def unpack_obs(packed, obs_shape):
    """Inverse of pack_obs -> float32 (..., H, W, C)."""
    d = obs_shape[0] * obs_shape[1] * obs_shape[2]
    flat = jnp.unpackbits(packed, axis=-1)[..., :d]
    return flat.reshape(packed.shape[:-1] + tuple(obs_shape)).astype(jnp.float32)
