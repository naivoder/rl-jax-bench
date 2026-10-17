"""Our side of the frozen-acting differential: lane_step(online=False) over the eval CPI of the diff-A stream, starting
from our final diff-A DDRQN/DQN params (carry otherwise fresh)."""
import sys, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the paper folder
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.join(PAPER, 'src'))
import data as D, train as TR
from model import arm_config
S = np.load("diffA_seed0.npz"); ntr, ntot = int(S["ntr"]), int(S["ntot"])
cfg = TR.load_cfg(os.path.join(PAPER, 'src', 'configs', 'full.yaml')); cfg["eval_cpis"] = 70
lanes, _ = TR.build_lanes(cfg, ["mc_p04"], ["eq3"], [0]); lane = jax.tree.map(lambda x: x[0], lanes)
out = {}
for arm in ("dqn", "ddrqn"):
    q = arm_config(arm, cfg)
    c0 = TR.init_carry(lane, q)
    p = jax.tree.map(lambda x: x, c0.q.params)
    flat, tdef = jax.tree_util.tree_flatten_with_path(p)
    leaves = [jnp.asarray(S[f"final_{arm}__" + "/".join(str(getattr(k, 'key', k)) for k in path)]) for path, _ in flat]
    p = jax.tree_util.tree_unflatten(tdef, leaves)
    c0 = c0._replace(q=c0.q._replace(params=p))
    def body(c, t):
        c2, o, fp, th = TR.lane_step(c, t, D.T_for_cpi(lane.params, t // D.CPI, ntr // 1000), lane, q, ntr, False, 0)
        return c2, (o, c2.prev_a)
    _, (o, a) = jax.jit(lambda c: jax.lax.scan(body, c, jnp.arange(ntr, ntot)))(c0)
    out[f"{arm}_act"] = np.asarray(a); out[f"{arm}_rew"] = np.asarray(o[:, 0]); out[f"{arm}_learned"] = np.asarray(o[:, 7]).sum()
np.savez("frozen_ours.npz", **out); print({k: v for k, v in out.items() if "learned" in k})
