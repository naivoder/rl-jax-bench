"""Analytic model FLOPs per environment step for each (env, method): counts the dense multiply-accumulates of the
networks as built by model.init_params (GRU w_ih/w_hh, MLP kernels; embeddings are lookups, 0 FLOPs), then
FLOPs/step = 2 * [actor_MAC * (1 rollout + 3 train fwd+bwd) + critic_MAC * (3 online fwd+bwd + 1 target fwd
+ 1 online fwd inside the actor loss)]. react-k re-runs its GRU over the k-window at every position (x k).
Ignores padding to length buckets (the real cost is higher by the padding fraction) -> runs/flops_per_step.json
    JAX_PLATFORMS=cpu python scripts/flops.py"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax  # noqa: E402

from envs import make_env  # noqa: E402
from model import METHODS, NetSpec, init_params  # noqa: E402


def macs(tree, k):
    tot = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        n = jax.tree_util.keystr(path)
        if n.endswith("['kernel']"):
            tot += leaf.shape[0] * leaf.shape[1]
        elif n.endswith("['w_ih']") or n.endswith("['w_hh']"):
            tot += leaf.shape[0] * leaf.shape[1] * k
    return tot


out = {}
for env in ("heavenhell_3", "heavenhell_4", "shopping_5", "shopping_6", "car_flag", "cleaner"):
    e = make_env(env)
    sp = NetSpec(e.kind, e.n_actions, e.n_obs, e.n_states)
    for m in METHODS:
        pa, pc = init_params(sp, m, jax.random.PRNGKey(0))
        k = METHODS[m][1] if isinstance(METHODS[m], tuple) and METHODS[m][1] else 1
        a, c = macs(pa, k), macs(pc, k)
        out[f"{env}/{m}"] = dict(actor_mac=a, critic_mac=c, flops_per_env_step=2 * (4 * a + 5 * c),
                                 n_params=int(sum(x.size for x in jax.tree.leaves((pa, pc)))))
        print(env, m, out[f"{env}/{m}"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "flops_per_step.json"), "w"), indent=1)
