"""Reference curve for the DDPG learner (no authors' code exists; CleanRL is the labelled
second-best oracle). Our DDPG pieces (alg.ddpg_update, alg.Replay/replay_add/replay_sample and train.py's
exploration rule: uniform warm-up, then clip(mu + std * scale * N(0,1))) are run on gymnax Pendulum-v1 with
CleanRL ddpg_continuous_action_jax.py @ fe8d8a0 hyperparameters (lr 3e-4, gamma 0.99, tau 0.005, batch 256,
exploration 0.1 x action scale, learning_starts 25000, policy_frequency 1 to match ddpg_update's one-critic +
one-actor step per call). The reference is CleanRL's own script run in its own venv
($OFFICIAL_ROOT/.venv_cleanrl, built by scripts/official/setup.sh) on gymnasium Pendulum-v1, same settings.

Two network arms, one vmapped program:
  cleanrl_nets : CleanRL's QNetwork / Actor (read from their file with ast at run time; nothing copied into src/)
  paper_nets   : OUR model.Actor (256-128, sigmoid x scale) / model.Critic (100-100) shifted to [-2, 2]
Metric: training episodic return (with exploration noise, as CleanRL logs it), mean of the last 10 episodes per seed.

Usage: python scripts/oracle_pendulum_ddpg.py --seeds 1,2,3 --steps 60000 --out runs/oracle/pendulum
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (gpu bootstrap before jax)
from src.common import info, metric  # noqa: E402

import flax  # noqa: E402
import flax.linen as nn  # noqa: E402
import gymnax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from src import algorithm as alg  # noqa: E402
from src.model import Actor as OurActor, Critic as OurCritic  # noqa: E402

CLEANRL = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "cleanrl", "cleanrl", "ddpg_continuous_action_jax.py")


def cleanrl_nets():
    src = open(CLEANRL).read()
    ns = dict(flax=flax, nn=nn, jax=jax, jnp=jnp, np=np)
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name in ("QNetwork", "Actor"):
            exec(ast.get_source_segment(src, node), ns)
    return ns["Actor"], ns["QNetwork"]


class ShiftActor(nn.Module):
    """Our Actor maps to [0, 2*max]; shift to [-max, max] (the paper actor's sigmoid x T0 head, T0 := 2 max)."""
    hi: float = 2.0

    @nn.compact
    def __call__(self, s):
        return OurActor(act_dim=1, hidden=(256, 128), T0=2 * self.hi)(s) - self.hi


class ShiftCritic(nn.Module):
    hi: float = 2.0

    @nn.compact
    def __call__(self, s, a):
        return OurCritic(hidden=(100, 100), T0=2 * self.hi)(s, a + self.hi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--learning_starts", type=int, default=25000)
    ap.add_argument("--out", default="runs/oracle/pendulum")
    ap.add_argument("--hp", default="cleanrl", choices=["cleanrl", "paper"],
                    help="cleanrl: Args defaults @ fe8d8a0; paper: Table V (lr 2e-4, gamma 0.9, tau 0.01, batch 128)")
    a = ap.parse_args(argv)
    common.check_device()
    lr, gamma, tau, B, expl = 3e-4, 0.99, 0.005, 256, 0.1        # CleanRL Args defaults @ fe8d8a0
    if a.hp == "paper":
        lr, gamma, tau, B = 2e-4, 0.9, 0.01, 128                   # Table V (exploration 0.1 = A-22)
    env = gymnax.environments.classic_control.Pendulum()
    ep = env.default_params
    hi = float(ep.max_torque)
    CA, CQ = cleanrl_nets()
    arms = {"cleanrl_nets": (CA(action_dim=1, action_scale=jnp.array([hi], jnp.float32),
                                action_bias=jnp.array([0.0], jnp.float32)), CQ()),
            "paper_nets": (ShiftActor(hi), ShiftCritic(hi))}
    seeds = [int(s) for s in a.seeds.split(",")]
    opt = optax.adam(lr)
    results = {}
    for name, (actor, critic) in arms.items():
        def run(seed):
            k = jax.random.PRNGKey(seed)
            k_init, k_env, k_expl, k_rb = jax.random.split(k, 4)
            st = alg.ddpg_init(k_init, 3, 1, actor, critic, opt, opt)
            buf = alg.replay_init(a.steps, 3, 1)
            obs, es = env.reset_env(jax.random.fold_in(k_env, 0), ep)

            def step(c, i):
                st, buf, obs, es, ret, n_ep = c
                kx = jax.random.fold_in(k_expl, i)
                mu = actor.apply(st.actor, obs[None].astype(jnp.float32))[0]
                a_expl = jnp.clip(mu + expl * hi * jax.random.normal(kx, (1,)), -hi, hi)
                a_rand = jax.random.uniform(jax.random.fold_in(kx, 1), (1,), minval=-hi, maxval=hi)
                act = jnp.where(i < a.learning_starts, a_rand, a_expl)
                obs2, es2, r, _, _ = env.step_env(jax.random.fold_in(k_env, 2 * i + 1), es, act, ep)
                done = env.is_truncated(es2, ep)                     # 200-step limit (Pendulum never terminates)
                buf = alg.replay_add(buf, obs.astype(jnp.float32), act.astype(jnp.float32), r.astype(jnp.float32),
                                     obs2.astype(jnp.float32))      # truncation only: real next obs, disc 1

                def upd(st):
                    batch = alg.replay_sample(buf, jax.random.fold_in(k_rb, i), B)
                    return alg.ddpg_update(st, batch, actor.apply, critic.apply, opt, opt, gamma, tau,
                                           a_clip=(-hi, hi))[0]
                st = jax.lax.cond(i > a.learning_starts, upd, lambda s: s, st)
                ret2 = ret + r
                o_r, es_r = env.reset_env(jax.random.fold_in(k_env, 2 * i + 2), ep)
                obs_n = jnp.where(done, o_r, obs2)
                es_n = jax.tree.map(lambda x, y: jnp.where(done, x, y), es_r, es2)
                return (st, buf, obs_n, es_n, jnp.where(done, 0.0, ret2), n_ep + done), (done, ret2)

            c0 = (st, buf, obs, es, jnp.zeros(()), jnp.zeros((), jnp.int32))
            _, (dones, rets) = jax.lax.scan(step, c0, jnp.arange(a.steps))
            return dones, rets

        t0 = time.time()
        dones, rets = jax.jit(jax.vmap(run))(jnp.array(seeds))
        dones, rets = np.asarray(dones), np.asarray(rets)
        info("arm done", arm=name, wall_s=round(time.time() - t0, 1))
        per = {}
        for j, s in enumerate(seeds):
            ep_ret = rets[j][dones[j]]
            per[s] = dict(episode_returns=ep_ret.tolist(), last10=float(ep_ret[-10:].mean()))
            metric("pendulum", arm=name, seed=s, last10=per[s]["last10"], n_ep=int(dones[j].sum()))
        results[name] = per
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"ours_pendulum_{a.hp}.json"), "w") as f:
        json.dump(dict(settings=dict(lr=lr, gamma=gamma, tau=tau, batch=B, expl=expl, steps=a.steps,
                                     learning_starts=a.learning_starts, policy_frequency=1, env="gymnax Pendulum-v1"),
                       results=results), f)


if __name__ == "__main__":
    main()
