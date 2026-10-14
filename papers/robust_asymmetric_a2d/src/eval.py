"""Evaluation (official logging()): every 5 iterations, roll out each policy for >= 2000 interactions of
COMPLETE episodes (one env, auto-reset; every episode that starts before step 2000 is counted; each lasts <= 100
steps so a 2100-step scan suffices) and report the mean undiscounted episode return, for the stochastic and the
argmax policy. Also: the clean divergence F_clean = mean KL[pi_theta(.|s) || pi_psi(.|b)] over the trainee's own
stochastic eval visits, hazard-hit rate, mean episode length, button-press rate (TD).
Eval uses its own PRNG stream and never touches training state.

`python -m src.eval <runs/...>` aggregates per-seed metrics into a summary (mean +- s.e., median/IQR, paired tests).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import data as D

EVAL_STEPS = 2000          # official --eval-batch-size (complete episodes)


def make_evaluate(spec, obs_l, state_f, pol_e, pol_l, lanes, need_kl: bool):
    """lanes: static tuple of (actor, deterministic) with actor in {'learner', 'expert'}.
    Returns evaluate(params, key) -> dict of per-lane arrays."""
    T = EVAL_STEPS + D.MAX_STEPS
    use_e = jnp.array([a == 'expert' for a, _ in lanes])
    det = jnp.array([d for _, d in lanes])
    has_e = any(a == 'expert' for a, _ in lanes) or need_kl
    has_l = any(a == 'learner' for a, _ in lanes)

    def one_lane(params, key, ue, dt):
        k_reset, k_act, k_noise = (jax.random.fold_in(key, i) for i in range(3))
        st0 = D.reset(spec, jax.random.fold_in(k_reset, 0))

        def body(c, t):
            st, start, ep_ret, sums = c
            lat = D.latent(spec, st)
            s = state_f(lat)
            lp_e = pol_e.apply(params['expert_pol'], s) if has_e else None
            lp_l = pol_l.apply(params['learner_pol'], obs_l(lat, jax.random.fold_in(k_noise, t))) if has_l else None
            if has_e and has_l:
                lp = jnp.where(ue, lp_e, lp_l)
            else:
                lp = lp_e if has_e and not has_l else lp_l
            a = jnp.where(dt, jnp.argmax(lp), jax.random.categorical(jax.random.fold_in(k_act, t), lp))
            st2, r, done = D.step(spec, st, a)
            counted = start < EVAL_STEPS
            kl = jnp.sum(jnp.exp(lp_e) * (lp_e - lp_l)) if (need_kl and has_l) else 0.0
            haz = done & (r < -50.0)
            press = (~st.illum) & st2.illum if spec.button is not None else False
            ep_ret = ep_ret + r
            sums = {'ret': sums['ret'] + jnp.where(done & counted, ep_ret, 0.0),
                    'n_ep': sums['n_ep'] + (done & counted),
                    'len': sums['len'] + jnp.where(counted, 1.0, 0.0),
                    'haz': sums['haz'] + (haz & counted),
                    'press': sums['press'] + (press & counted),
                    'kl': sums['kl'] + jnp.where(counted, kl, 0.0)}
            st2 = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y),
                                         D.reset(spec, jax.random.fold_in(k_reset, t + 1)), st2)
            start = jnp.where(done, t + 1, start)
            ep_ret = jnp.where(done, 0.0, ep_ret)
            return (st2, start, ep_ret, sums), None
        z = jnp.zeros(())
        sums0 = {'ret': z, 'n_ep': jnp.int32(0), 'len': z, 'haz': jnp.int32(0), 'press': jnp.int32(0), 'kl': z}
        (_, _, _, s), _ = jax.lax.scan(body, (st0, jnp.int32(0), z, sums0), jnp.arange(T, dtype=jnp.int32))
        n = jnp.maximum(s['n_ep'], 1)
        return {'ret': s['ret'] / n, 'n_ep': s['n_ep'], 'ep_len': s['len'] / n, 'haz_rate': s['haz'] / n,
                'press_rate': s['press'] / n, 'kl_clean': s['kl'] / jnp.maximum(s['len'], 1.0)}

    def evaluate(params, key):
        keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(len(lanes)))
        return jax.vmap(one_lane, in_axes=(None, 0, 0, 0))(params, keys, use_e, det)
    return evaluate


# ------------------------------------------------------------------------------------------------ aggregation CLI
def summarise(root: str) -> dict:
    """Collect runs/<exp>/<env>/<arm>/seed_*/metrics.json under `root` into {env: {arm: per-seed finals}}."""
    import glob
    import json
    import os
    out = {}
    for f in sorted(glob.glob(os.path.join(root, '*', '*', 'seed_*', 'metrics.json'))):
        m = json.load(open(f))
        env, arm = m['config']['env'], m['arm']
        out.setdefault(env, {}).setdefault(arm, {})[m['seed']] = m['final']
    return out


if __name__ == '__main__':
    import json
    import sys
    print(json.dumps(summarise(sys.argv[1]), indent=1))
