"""Export our Breakout guide (flax msgpack) to plain numpy arrays + JAX reference logits on demo states (main venv)."""
import sys, os
from rl_jax_bench import gpu_bootstrap; os.environ.setdefault('JAX_PLATFORMS', 'cpu'); gpu_bootstrap.setup()
PAPER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, PAPER)
import numpy as np, yaml, jax, jax.numpy as jnp
from src.envs import MinAtar, unpack_obs
from src.guide import load_guide, make_guide_fn
cfg = yaml.safe_load(open(os.path.join(PAPER, 'src', 'configs', 'default.yaml')))
env = MinAtar('Breakout')
net, params = load_guide(env, cfg)
flat = {}
for path, leaf in jax.tree_util.tree_leaves_with_path(params):
    flat['/'.join(str(getattr(p, 'key', p)) for p in path)] = np.asarray(leaf)
print({k: v.shape for k, v in flat.items()})
d = np.load(os.path.join(PAPER, 'data', 'demos', 'Breakout_subopt.npz'))
obs = np.asarray(unpack_obs(jnp.asarray(d['obs'][:4096]), env.obs_shape)).astype(bool)
logits, _ = net.apply(params, jnp.asarray(obs, jnp.float32))
np.savez('guide_breakout_np.npz', **flat)
np.savez('guide_ref_logits.npz', obs=obs, logits=np.asarray(logits), demo_act=d['act'][:4096])
print('obs', obs.shape, 'action_set', env.env.action_set if hasattr(env.env,'action_set') else None, 'n_actions', env.n_actions, 'max_steps', env.max_steps)
