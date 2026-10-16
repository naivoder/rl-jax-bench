"""Labelled SECOND-BEST reference curve: Brax's own PPO (brax.training.agents.ppo, same organisation)
on the same brax-mjx PROXY env, at a Table-2-like setting: 256 envs x 8-step unrolls (2048 steps/iteration), 10
epochs, minibatch 64 transitions (8 unrolls x 8), gamma 0.99, GAE 0.95, PPO eps 0.2, lr 3e-4 (constant), grad
clip 0.5, obs normalisation on, no entropy, tanh-Normal policy (64,64), value MLP (64,64). Differences from the
paper's agent that cannot be configured in brax: swish hidden activation (brax default), brax's value-loss weight 0.5,
per-minibatch advantage normalisation ON (brax default), brax's AutoResetWrapper (fixed initial state per env),
evaluation with brax's own evaluator (128 stochastic episodes). Usage: python scripts/brax_ppo_reference.py <env> <seed>
"""
import sys
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import functools, json, os, time, warnings
import jax
from rl_jax_bench.jaxlog import info, metric, check_device
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from brax import envs
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks

check_device()
if "device_put_replicated" not in jax.__dict__:
    # brax 0.14.2 calls jax.device_put_replicated, removed in jax 0.11 (deprecation stub raises): drop-in shim
    # per the jax migration guide (stack one copy per device, then device_put).
    import jax.numpy as jnp
    jax.device_put_replicated = lambda x, devices: jax.device_put(
        jax.tree.map(lambda a: jnp.stack([jnp.asarray(a)] * len(devices)), x), devices[0])
name, seed = sys.argv[1], int(sys.argv[2])
env = envs.get_environment(name, backend='mjx')
curve = []
t0 = time.time()


def progress(step, m):
    curve.append((int(step), float(m['eval/episode_reward'])))
    metric("brax_ppo", env=name, seed=seed, step=int(step), eval_return=round(float(m['eval/episode_reward']), 1),
           wall=round(time.time() - t0, 1))


net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(64, 64),
                        value_hidden_layer_sizes=(64, 64))
ppo.train(environment=env, num_timesteps=1_000_000, num_envs=256, episode_length=1000, action_repeat=1,
          learning_rate=3e-4, entropy_cost=0.0, discounting=0.99, unroll_length=8, batch_size=8, num_minibatches=32,
          num_updates_per_batch=10, normalize_observations=True, reward_scaling=1.0, clipping_epsilon=0.2,
          gae_lambda=0.95, max_grad_norm=0.5, network_factory=net, seed=seed, num_evals=11, num_eval_envs=128,
          deterministic_eval=False, progress_fn=progress)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'brax_ppo_ref', name, f'seed_{seed}')
os.makedirs(out, exist_ok=True)
json.dump({"env": name, "seed": seed, "curve": curve, "score_mean_pts_after_0": sum(r for s, r in curve[1:]) / max(1, len(curve) - 1),
           "wall_s": time.time() - t0, "brax": __import__('brax').__version__}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
info("done", out=out)
