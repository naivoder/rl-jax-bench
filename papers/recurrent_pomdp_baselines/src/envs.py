"""Pure-JAX environments used by the reproduction (all float32, all fixed-horizon except CartPole).

Every env exposes the same functional interface (pure functions, vmappable, jit-able):

    tasks = env.make_tasks(key)              -> {"train": (n_train, k), "eval": (n_eval, k)}   (k may be 0)
    state, obs = env.reset(key, task)
    state, obs, reward, done, term = env.step(key, state, task, action)   # action in the AGENT's space
    a_store = env.warmup_action(key)         # what the official code stores during random warm-up

Agent action space: continuous actions live in [-1, 1]^d and are clipped then affinely mapped to the env's
range by the env (official ``POMDPWrapper`` / ``VariBadWrapper``).  Discrete actions are one-hot vectors.

Faithful quirks of the official code (README, README):
* Pendulum: gym 0.18 ``Pendulum-v0`` order (theta advanced with the UNclipped speed, then clip) by default;
  ``version="v1"`` gives gym 0.21 ``Pendulum-v1`` (clip first).  Time-limit end is non-terminal.
* Random warm-up actions are drawn from the UNDERLYING env's action space (Pendulum U[-2,2], point robot
  U[-0.1,0.1]) and stored raw, although the wrapper then clips to [-1,1] and rescales
  (learner.collect_rollouts: ``self.train_env.action_space.sample()``; the wrappers do not override
  ``action_space``).
* Semi-Circle (VariBAD wrapper, 2 episodes per task): the observation carries the inner-episode done flag; at the
  inner boundary the agent sees the terminal position with d=1, the env resets internally, the next action is
  applied from the fresh start; the buffer's terminal flag is ``is_goal_state()`` evaluated AFTER that internal
  reset (learner calls it after ``env.step``).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32


def _affine(a, lb, ub):
    """official wrappers: clip to [-1,1], map to [lb,ub], clip again."""
    a = jnp.clip(a, -1.0, 1.0)
    a = lb + (a + 1.0) * 0.5 * (ub - lb)
    return jnp.clip(a, lb, ub)


# --------------------------------------------------------------------------------------------- Pendulum
@dataclass(frozen=True)
class Pendulum:
    """gym Pendulum (g=10, m=l=1, dt=0.05, |u|<=2, |thdot|<=8, H=200) behind the official POMDPWrapper.

    obs_dims: [0,1] = -P (cos, sin); [2] = -V (thdot); [0,1,2] = full (-F).  version: 'v0' (gym 0.18) or 'v1'."""
    obs_dims: tuple = (2,)
    version: str = "v0"
    H: int = 200
    name: str = "Pendulum-V"
    discrete: bool = False
    act_dim: int = 1
    n_train_tasks: int = 0
    n_eval_tasks: int = 0
    episodes_per_task: int = 1

    @property
    def obs_dim(self):
        return len(self.obs_dims)

    def make_tasks(self, key):
        return {"train": jnp.zeros((1, 1), F32), "eval": jnp.zeros((1, 1), F32)}  # unused (no task); non-empty for orbax

    def _obs(self, th, thdot):
        full = jnp.stack([jnp.cos(th), jnp.sin(th), thdot])
        return full[jnp.array(self.obs_dims)].astype(F32)

    def reset(self, key, task):
        hi = jnp.array([np.pi, 1.0], F32)
        s = jax.random.uniform(key, (2,), F32, -hi, hi)
        state = {"th": s[0], "thdot": s[1], "t": jnp.zeros((), jnp.int32)}
        return state, self._obs(s[0], s[1])

    def physics(self, th, thdot, u):
        """one gym step from (th, thdot) with torque u (already in [-2,2]); returns th', thdot', reward."""
        g, m, l, dt = 10.0, 1.0, 1.0, 0.05
        u = jnp.clip(u, -2.0, 2.0)
        thn = ((th + np.pi) % (2 * np.pi)) - np.pi
        cost = thn ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2
        if self.version == "v0":  # gym 0.18: -3g/(2l) sin(th+pi); theta uses the unclipped speed
            newthdot = thdot + (-3 * g / (2 * l) * jnp.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
            newth = th + newthdot * dt
            newthdot = jnp.clip(newthdot, -8.0, 8.0)
        else:  # gym 0.21 v1: 3g/(2l) sin(th); clip first
            newthdot = thdot + (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l ** 2) * u) * dt
            newthdot = jnp.clip(newthdot, -8.0, 8.0)
            newth = th + newthdot * dt
        return newth.astype(F32), newthdot.astype(F32), (-cost).astype(F32)

    def step(self, key, state, task, action):
        u = _affine(action[0], -2.0, 2.0)
        th, thdot, r = self.physics(state["th"], state["thdot"], u)
        t = state["t"] + 1
        done = t >= self.H
        return {"th": th, "thdot": thdot, "t": t}, self._obs(th, thdot), r, done, jnp.zeros((), F32)

    def warmup_action(self, key):
        return jax.random.uniform(key, (1,), F32, -2.0, 2.0)


# --------------------------------------------------------------------------------------------- Semi-Circle
@dataclass(frozen=True)
class SemiCircle:
    """PointRobotSparse-v0 + VariBadWrapper(episodes_per_task=2): goal on the unit upper semicircle, reward 1 inside
    radius 0.2, H=60 per inner episode, start ~ U([-1.5,1.5]x[-0.5,1.5]).  oracle=True appends the goal to obs."""
    n_tasks: int = 100
    n_train_tasks: int = 80
    n_eval_tasks: int = 20
    goal_radius: float = 0.2
    inner_H: int = 60
    episodes_per_task: int = 2
    oracle: bool = False
    name: str = "SemiCircle"
    discrete: bool = False
    act_dim: int = 2

    @property
    def H(self):
        return self.inner_H * self.episodes_per_task

    @property
    def obs_dim(self):
        return 2 + (2 if self.oracle else 0) + (1 if self.episodes_per_task > 1 else 0)

    def make_tasks(self, key):
        """100 goals at angles ~U[0,pi] (point_robot.SparsePointEnv), shuffled; first 80 train, last 20 eval
        (learner.init_env: permutation of task indices)."""
        k1, k2 = jax.random.split(key)
        ang = jax.random.uniform(k1, (self.n_tasks,), F32, 0.0, np.pi)
        goals = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
        perm = jax.random.permutation(k2, self.n_tasks)
        goals = goals[perm]
        return {"train": goals[: self.n_train_tasks], "eval": goals[-self.n_eval_tasks:]}

    def _start(self, key):
        return jnp.stack([jax.random.uniform(key, (), F32, -1.5, 1.5),
                          jax.random.uniform(jax.random.fold_in(key, 1), (), F32, -0.5, 1.5)])

    def _obs(self, pos, task, done_mdp):
        parts = [pos]
        if self.oracle:
            parts.append(task)
        if self.episodes_per_task > 1:
            parts.append(done_mdp.astype(F32)[None])
        return jnp.concatenate(parts).astype(F32)

    def reset(self, key, task):
        pos = self._start(key)
        state = {"pos": pos, "t": jnp.zeros((), jnp.int32), "ep": jnp.zeros((), jnp.int32)}
        return state, self._obs(pos, task, jnp.zeros((), bool))

    def step(self, key, state, task, action):
        move = _affine(action, -0.1, 0.1)
        pos = state["pos"] + move
        dist = jnp.sqrt(jnp.sum((pos - task) ** 2))
        reward = jnp.where(-dist >= -self.goal_radius, 1.0, 0.0).astype(F32)
        t = state["t"] + 1
        done_mdp = t >= self.inner_H
        obs = self._obs(pos, task, done_mdp)
        ep = state["ep"] + done_mdp.astype(jnp.int32)
        done_bamdp = ep >= self.episodes_per_task
        do_reset = done_mdp & (~done_bamdp)
        pos_after = jnp.where(do_reset, self._start(key), pos)  # internal reset_mdp
        t_after = jnp.where(do_reset, 0, t)
        term = (jnp.sqrt(jnp.sum((pos_after - task) ** 2)) <= self.goal_radius).astype(F32)  # is_goal_state()
        return {"pos": pos_after, "t": t_after, "ep": ep}, obs, reward, done_bamdp, term

    def warmup_action(self, key):
        return jax.random.uniform(key, (2,), F32, -0.1, 0.1)


# --------------------------------------------------------------------------------------------- Wind
WIND_NP_SEED = 1337  # wind.py: np.random.seed(1337) before drawing the winds


def official_winds(n_tasks: int = 60):
    """The exact wind list of the official WindEnv (numpy legacy RNG seeded 1337, two uniforms per task) and the
    learner's train/eval split: WindEnv.__init__ re-seeds numpy's GLOBAL RNG with 1337, so the learner's
    ``np.random.permutation`` right after construction is the same for every run seed (verified against the official
    code in tests/test_envs.py)."""
    rs = np.random.RandomState(WIND_NP_SEED)
    winds = np.array([[rs.uniform(-0.08, 0.08), rs.uniform(-0.08, 0.08)] for _ in range(n_tasks)], np.float32)
    return winds, rs.permutation(n_tasks)


@dataclass(frozen=True)
class Wind:
    """Wind-v0: start (0,0), goal (0,1), radius 0.03, s' = s + a + w, H=75, one episode per task (no done flag)."""
    n_tasks: int = 60
    n_train_tasks: int = 40
    n_eval_tasks: int = 20
    goal_radius: float = 0.03
    H: int = 75
    episodes_per_task: int = 1
    oracle: bool = False
    name: str = "Wind"
    discrete: bool = False
    act_dim: int = 2

    @property
    def obs_dim(self):
        return 2 + (2 if self.oracle else 0)

    def make_tasks(self, key):
        w, perm = official_winds(self.n_tasks)
        winds = jnp.asarray(w)[jnp.asarray(perm)]          # official fixed split (seed-independent)
        return {"train": winds[: self.n_train_tasks], "eval": winds[-self.n_eval_tasks:]}

    def _obs(self, pos, task):
        return (jnp.concatenate([pos, task]) if self.oracle else pos).astype(F32)

    def reset(self, key, task):
        pos = jnp.zeros((2,), F32)
        return {"pos": pos, "t": jnp.zeros((), jnp.int32)}, self._obs(pos, task)

    def step(self, key, state, task, action):
        move = _affine(action, -0.1, 0.1)
        pos = state["pos"] + move + task
        goal = jnp.array([0.0, 1.0], F32)
        ing = jnp.sqrt(jnp.sum((pos - goal) ** 2)) <= self.goal_radius
        t = state["t"] + 1
        return {"pos": pos, "t": t}, self._obs(pos, task), ing.astype(F32), t >= self.H, ing.astype(F32)

    def warmup_action(self, key):
        return jax.random.uniform(key, (2,), F32, -0.1, 0.1)


# --------------------------------------------------------------------------------------------- Cue memory (toy stage)
@dataclass(frozen=True)
class CueMemory:
    """the README toy.  A cue c in {0,1} is shown at t=0 only (one-hot), every obs carries f_t = 1[t = H-1].
    Reward only at the decision step t = H-1: 1[a = c] (discrete) or (2c-1)*a (continuous, a in [-1,1]).
    The last step is terminal (term = 1)."""
    H: int = 5
    discrete: bool = True
    name: str = "CueMemory"
    n_train_tasks: int = 0
    n_eval_tasks: int = 0
    episodes_per_task: int = 1

    @property
    def obs_dim(self):
        return 3

    @property
    def act_dim(self):
        return 2 if self.discrete else 1

    def make_tasks(self, key):
        return {"train": jnp.zeros((1, 1), F32), "eval": jnp.zeros((1, 1), F32)}  # unused (no task); non-empty for orbax

    def _obs(self, c, t):
        cue = jnp.where(t == 0, jax.nn.one_hot(c, 2), jnp.zeros((2,)))
        return jnp.concatenate([cue, (t == self.H - 1).astype(F32)[None]]).astype(F32)

    def reset(self, key, task):
        c = jax.random.bernoulli(key).astype(jnp.int32)
        t = jnp.zeros((), jnp.int32)
        return {"c": c, "t": t}, self._obs(c, t)

    def step(self, key, state, task, action):
        c, t = state["c"], state["t"]
        last = t == self.H - 1
        if self.discrete:
            r = (jnp.argmax(action) == c).astype(F32)
        else:
            r = (2.0 * c - 1.0) * jnp.clip(action[0], -1.0, 1.0)
        r = jnp.where(last, r, 0.0).astype(F32)
        t1 = t + 1
        return {"c": c, "t": t1}, self._obs(c, t1), r, t1 >= self.H, last.astype(F32)

    def warmup_action(self, key):
        if self.discrete:
            return jax.nn.one_hot(jax.random.randint(key, (), 0, 2), 2).astype(F32)
        return jax.random.uniform(key, (1,), F32, -1.0, 1.0)


def make_env(cfg: dict):
    name = cfg["env"]
    if name in ("Pendulum-V", "Pendulum-P", "Pendulum-F"):
        dims = {"Pendulum-V": (2,), "Pendulum-P": (0, 1), "Pendulum-F": (0, 1, 2)}[name]
        return Pendulum(obs_dims=dims, version=cfg.get("pendulum_version", "v0"), name=name)
    if name == "SemiCircle":
        return SemiCircle(oracle=bool(cfg.get("oracle", False)))
    if name == "Wind":
        return Wind(oracle=bool(cfg.get("oracle", False)))
    if name == "CueMemory":
        return CueMemory(H=int(cfg.get("cue_H", 5)), discrete=cfg["algo"] == "sacd")
    raise ValueError(name)
