"""small-scale: gymnax vs gymnasium 0.29.1 dynamics differential test.

(1) Teacher-forced one-step test: 1000 transitions per env from a random-policy auto-reset rollout in gymnax; for each,
    the pre-step state, the action, and gymnax's next obs / reward / terminated. The oracle sets gymnasium's
    env.unwrapped.state to the same state, steps the same action and records the same quantities.
(2) Open-loop test (descriptive): 10 matched initial states + one fixed action sequence, both envs run freely until
    the first done (max 500 steps); max |obs difference| is reported (chaotic float32-vs-float64 drift is expected).
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import src.train  # noqa: E402, (GPU bootstrap + matmul precision)

import gymnax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(REPO, "runs", "small", "difftest")
FIELDS = {"CartPole-v1": ["x", "x_dot", "theta", "theta_dot"],
          "Acrobot-v1": ["joint_angle1", "joint_angle2", "velocity_1", "velocity_2"],
          "MountainCar-v0": ["position", "velocity"],
          "Pendulum-v1": ["theta", "theta_dot"]}


def action_of(env_id, key, n):
    if env_id == "Pendulum-v1":
        return jax.random.uniform(key, (n, 1), minval=-2.0, maxval=2.0)
    na = {"CartPole-v1": 2, "Acrobot-v1": 3, "MountainCar-v0": 3}[env_id]
    return jax.random.randint(key, (n,), 0, na)


def vec(s, env_id):
    return jnp.stack([jnp.asarray(getattr(s, f)).reshape(()) for f in FIELDS[env_id]])


def one_env(env_id, n=1000, seed=0):
    env, p = gymnax.make(env_id)
    k = jax.random.PRNGKey(seed)
    ka, kr, ks = jax.random.split(k, 3)
    acts = action_of(env_id, ka, n)
    o, s = env.reset(kr, p)

    def body(c, x):
        s, = c
        a, t = x
        o_st, s_st, r, term, _ = env.step_env(jax.random.fold_in(ks, t), s, a, p)
        _, s2, _, _, _, _ = env.step(jax.random.fold_in(ks, t), s, a, p)
        return (s2,), (vec(s, env_id), o_st, r, term)

    _, (states, nobs, rew, term) = jax.lax.scan(body, (s,), (acts, jnp.arange(n)))
    out = {f"{env_id}/tf/state": states, f"{env_id}/tf/action": acts, f"{env_id}/tf/obs": nobs,
           f"{env_id}/tf/reward": rew, f"{env_id}/tf/terminated": term}
    # (2) open loop
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), 10)
    seq = action_of(env_id, jax.random.PRNGKey(seed + 2), 500)
    inits, trajs, alive_all = [], [], []
    for i in range(10):
        o, s = env.reset(keys[i], p)
        inits.append(vec(s, env_id))
        obs_l, alive, alive_l = [], True, []
        for t in range(500):
            o_st, s, r, term, _ = env.step_env(keys[i], s, seq[t], p)
            obs_l.append(np.asarray(o_st)); alive_l.append(alive)
            alive = alive and not bool(term)
        trajs.append(np.stack(obs_l)); alive_all.append(np.array(alive_l))
    out.update({f"{env_id}/ol/init": np.stack(inits), f"{env_id}/ol/actions": seq, f"{env_id}/ol/obs": np.stack(trajs),
                f"{env_id}/ol/alive": np.stack(alive_all)})
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for e in FIELDS:
        out.update({k: np.asarray(v) for k, v in one_env(e).items()})
    np.savez(os.path.join(OUT, "ours_envs.npz"), **out)
    print("[INFO] wrote ours_envs.npz", len(out))
