"""XLA cost analysis (FLOPs) of one lane-step of every arm (act + store + one gradient update), on CPU.
Compute-parity accounting (XLA FLOPs per lane-step for each arm). Usage: JAX_PLATFORMS=cpu python scripts/flops.py"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import algorithm as alg  # noqa: E402
import train as TR  # noqa: E402
from model import ARMS, arm_config, is_window  # noqa: E402

CFG = TR.load_cfg(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "configs", "full.yaml"))
lanes, _ = TR.build_lanes(CFG, ["mc_p04"], ["eq3"], [0])
lane = jax.tree.map(lambda x: x[0], lanes)
out = {}
for arm in ARMS:
    q = arm_config(arm, CFG)
    w = is_window(arm)
    c = TR.init_carry(lane, q, w)
    fn = TR.lane_step_window if w else TR.lane_step
    batch = alg.buffer_sample(c.buf, jax.random.PRNGKey(0), q.batch_size)
    upd = (lambda s, b: TR.window_update(s, b, q)) if w else (lambda s, b: alg.update(s, b, q))
    fl_upd = jax.jit(upd).lower(c.q, batch).compile().cost_analysis()
    fl_upd = (fl_upd[0] if isinstance(fl_upd, list) else fl_upd)["flops"]
    net = alg.make_network(q)
    obs = jnp.zeros((1, q.obs_dim))
    if w:
        act = lambda p: alg.unroll(net, q, p, jnp.zeros((1, q.trace_len, q.obs_dim)), alg.initial_carry(q, (1,)))[0]  # noqa: E731
    else:
        act = lambda p: alg.act_q(net, q, p, obs, alg.initial_carry(q, (1,)))[0]  # noqa: E731
    fl_act = jax.jit(act).lower(c.q.params).compile().cost_analysis()
    fl_act = (fl_act[0] if isinstance(fl_act, list) else fl_act)["flops"]
    n_par = sum(x.size for x in jax.tree.leaves(c.q.params))
    steps = (CFG["train_cpis"] + CFG["eval_cpis"]) * 1000
    out[arm] = dict(params=int(n_par), flops_update=float(fl_upd), flops_act=float(fl_act),
                    flops_per_seed_run=float((fl_upd + fl_act) * steps), env_steps=steps, grad_steps=steps)
    print(arm, out[arm])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "flops.json"), "w"), indent=1)
