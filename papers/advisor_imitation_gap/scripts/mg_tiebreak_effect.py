"""How often the earlier MiniGrid teacher (ties -> lowest action index) and the official one
(left/right tie facing direction 0 -> right) give a DIFFERENT label along the trajectories that matter: the final
full-scale MG policies (restored from runs/full_mg/ckpt, sampled actions, the 1000 fixed eval layouts' first n_ep) and the
teacher's own trajectories. Split by dark (before the toggle) and lit states. Writes runs/report/mg_tiebreak.json.
Usage: mg_tiebreak_effect.py [n_ep=200]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import bootstrap, check_device, info  # noqa: E402

bootstrap()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import model as M  # noqa: E402
from src import train as TR  # noqa: E402
from src.envs import crossing as MG  # noqa: E402


def teacher_pair(s):
    d = s.dist
    x, y, k = s.pos[0], s.pos[1], s.dir
    vec = jnp.asarray(MG.DIR_TO_VEC)[k]
    cand = jnp.stack([d[x, y, (k - 1) % 4], d[x, y, (k + 1) % 4], d[x + vec[0], y + vec[1], k]])
    legacy = jnp.argmin(cand)
    off = jnp.where((k == 0) & (cand[0] == cand[1]) & (cand[1] < cand[2]), 1, legacy)
    tie = jnp.sum(cand == cand.min()) > 1
    return legacy, off, tie


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    plat = check_device()
    S = json.load(open("runs/full_mg/summary.json"))
    cfg = TR.TrainCfg(env="mg", total_steps=1000000)
    net = M.MGNet(n_actions=MG.N_ACTIONS)
    spec = MG.SPEC
    specs = [dict(arm=r["arm"], seed=r["seed"], lr=r["lr"], alpha=r["alpha"], beta=0.0, split=r["split"], ep=r["ep"],
                  opt_len=0.0) for r in S["runs"]]
    runs = TR.build_runs(cfg, specs)
    st0 = jax.jit(jax.vmap(lambda r: TR.init_run(cfg, spec, net, r)))(runs)
    st, _ = TR._maybe_restore_at(os.path.join(os.path.abspath("runs"), "full_mg", "ckpt"), st0, cfg.n_updates)
    keep = [i for i, r in enumerate(S["runs"]) if r["arm"] in ("advisor", "bc", "ppo")]
    params = jax.tree_util.tree_map(lambda x: x[jnp.asarray(keep)], st["params"])

    def roll(p, key, use_teacher):
        env = jax.vmap(lambda i: spec.eval_reset(i, None, {}))(jnp.arange(n_ep))
        carry = net.initial_carry(n_ep)

        def body(c, t):
            env, carry, done, acc = c
            o = jax.vmap(MG.obs)(env)
            lg, _, _, carry = net.apply(p, o, carry, jnp.full(n_ep, t == 0))
            leg, off, tie = jax.vmap(teacher_pair)(env)
            a = jnp.where(use_teacher, off, jax.random.categorical(jax.random.fold_in(key, t), lg))
            live = ~done
            dark = ~env.reveal
            diff = leg != off
            acc = acc + jnp.stack([jnp.sum(live & dark), jnp.sum(live & dark & diff), jnp.sum(live & dark & tie),
                                   jnp.sum(live & ~dark), jnp.sum(live & ~dark & diff), jnp.sum(live & ~dark & tie)])
            env_n, _, d = jax.vmap(MG.step)(env, a)
            env_n = TR._tree_where_b(done, env, env_n)
            return (env_n, carry, done | d, acc), None
        (_, _, _, acc), _ = jax.lax.scan(body, (env, carry, jnp.zeros(n_ep, bool), jnp.zeros(6)), jnp.arange(MG.MAX_STEPS))
        return acc

    f = jax.jit(jax.vmap(roll, in_axes=(0, None, None)))
    out = {}
    for name, ut in (("policy", False), ("teacher", True)):
        acc = np.asarray(f(params, jax.random.PRNGKey(123), ut))
        for j, i in enumerate(keep):
            r = S["runs"][i]
            a = acc[j]
            key = f"{name}/{r['arm']}/seed{r['seed']}" + (f"/draw{r['draw']}" if r.get("draw") is not None else "")
            out[key] = dict(dark_steps=float(a[0]), dark_label_diff=float(a[1] / max(a[0], 1)), dark_tie=float(a[2] / max(a[0], 1)),
                            lit_steps=float(a[3]), lit_label_diff=float(a[4] / max(a[3], 1)), lit_tie=float(a[5] / max(a[3], 1)))
            info(key, **{k: round(v, 4) for k, v in out[key].items()})
            if ut:
                break    # the teacher's trajectory does not depend on the policy: one copy
    json.dump(dict(platform=plat, n_ep=n_ep, rows=out), open("runs/report/mg_tiebreak.json", "w"), indent=1)


if __name__ == "__main__":
    main()
