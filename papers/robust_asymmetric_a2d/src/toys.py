"""Toy correctness checks (i)-(v). Writes runs/toys/toys.json.

(i)   environment == official env (bit-exact golden rollouts, 5 envs) and every algorithm piece == official code
      -> tests/test_official_parity.py (17 differential tests), run here as a subprocess and recorded.
(ii)  exact optima of OUR env (value iteration / belief-MDP DP / exact AIL fixed point) == hand-derived values + paper text.
(iii) exact AIL fixed point vs OUR AIL training code (compact trainee, sampled, NN) for a fixed (distilled max-ent)
      expert on FL and TD: final stochastic return and clean KL within max(3 s.e., 5 % |exact|) / 25 % relative.
(iv)  TD1 closed-form expert gradient at initialization (uniform trainee, uniform expert) through OUR
      importance-weight code: Q-target (+0.5, +0.5, -0.5, -0.5) and Monte-Carlo target (+15.5, -14.5, -0.5, -0.5)
      in (left, right, up, down) order, to 1e-6.
(v)   exact tabular A2D: Q(b,a) critic reaches the POMDP optimum on TD and TD1; the MC target does not on TD1
      (stays at the implicit-policy value -42 with KL = log 2).
"""
from __future__ import annotations

import sys
from rl_jax_bench import gpu_bootstrap  # noqa: E402
gpu_bootstrap.setup(deterministic=False)

import json, os, subprocess, time  # noqa: E401,E402
import numpy as np                  # noqa: E402
import jax                          # noqa: E402
import jax.numpy as jnp             # noqa: E402
import optax                        # noqa: E402

from rl_jax_bench import jaxlog                      # noqa: E402
from . import data as D, exact as E, model as M, algorithm as AL, train as T  # noqa: E402

REPO = T.REPO
OUT = os.path.join(REPO, 'runs', 'toys')
PAPER = {  # hand-derived exact values / paper text
    'FL': dict(mdp=10.6667, pomdp=4.0, ail_sto=-36.141, ail_greedy=-26.667),
    'TD': dict(mdp=6.0, pomdp=2.0, ail_sto=-66.0),
    'TD1': dict(mdp=18.0, pomdp=16.0, ail_sto=-42.0),
    'TD2': dict(mdp=16.0, pomdp=14.0, ail_sto=-48.0),
    'TD3': dict(mdp=14.0, pomdp=12.0, ail_sto=-58.0),
}


def check_oracle_tests():
    t = time.time()
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_official_parity.py'], cwd=REPO,
                       capture_output=True, text=True, env=dict(os.environ, JAX_PLATFORMS='cpu'))
    last = r.stdout.strip().splitlines()[-1]
    return {'pass': r.returncode == 0, 'summary': last, 'secs': round(time.time() - t, 1)}


def check_table():
    out = {}
    for name, ref in PAPER.items():
        tab = E.Tab(name)
        mu, md, pi = tab.mdp_optimum()
        po = tab.pomdp_optimum()
        fp = tab.ail_fixed_point(pi, gamma_w=D.GAMMA)
        got = dict(mdp=mu, pomdp=po[0], ail_sto=fp['ret_sto'], ail_greedy=fp['ret_greedy'], ail_kl=fp['kl'],
                   mdp_disc=md, pomdp_disc=po[1])
        ok = all(abs(got[k] - v) < 2e-3 for k, v in ref.items())
        out[name] = {'got': got, 'ref': ref, 'pass': bool(ok)}
    return out


def check_td1_gradient():
    """Hand-derived closed form via exact Q values of our env and our importance_weight in the grad path."""
    tab = E.Tab('TD1')
    spec = tab.spec
    pi_l = np.full((tab.B, 4), 0.25)
    Qs, Qb, _ = tab.belief_q(pi_l, gamma=1.0 - 1e-12)
    p0 = spec.start[1] * spec.n + spec.start[0]
    h_left = 1                                   # goal on the low-x side ("L")
    s = p0 * spec.K * 2 + h_left * 2
    b = tab.obs[s]
    lpl = jnp.log(jnp.full((4,), 0.25, jnp.float64))

    def grad_for(X):
        def L(z):
            lpe = jax.nn.log_softmax(z)
            w = AL.importance_weight(lpe, lpl, 0.0)          # pi_theta(a|s) / pi_psi(a|b) at beta = 0
            return jnp.sum(jnp.exp(lpl) * w * X)
        return np.asarray(jax.grad(L)(jnp.zeros(4, jnp.float64)))
    order = [2, 0, 3, 1]                          # (left, right, up, down) in our action indices
    g_q = grad_for(jnp.asarray(Qb[b]))[order]
    g_mc = grad_for(jnp.asarray(Qs[s]))[order]
    ref_q, ref_mc = np.array([0.5, 0.5, -0.5, -0.5]), np.array([15.5, -14.5, -0.5, -0.5])
    V_b = float(np.sum(pi_l[b] * Qb[b]))
    return {'V_start': V_b, 'Q_belief_start(l,r,u,d)': Qb[b][order].tolist(),
            'Q_state_goalL(l,r,u,d)': Qs[s][order].tolist(), 'grad_q': g_q.tolist(), 'grad_mc': g_mc.tolist(),
            'bias': (g_mc - g_q).tolist(),
            'pass': bool(np.allclose(g_q, ref_q, atol=1e-6) and np.allclose(g_mc, ref_mc, atol=1e-6)
                         and abs(V_b + 44.0) < 1e-6)}


def check_exact_a2d(iters=600):
    out = {}
    for name in ['TD1', 'TD']:
        tab = E.Tab(name)
        for tgt in ['q', 'mc']:
            r = tab.exact_a2d(tgt, iters=iters)
            out[f'{name}/{tgt}'] = {'ret': r['ret'], 'kl': r['kl'], 'hist': r['hist']}
    ok = (abs(out['TD1/q']['ret'] - 16) < 0.05 and abs(out['TD/q']['ret'] - 2) < 0.05
          and abs(out['TD1/mc']['ret'] + 42) < 0.5 and abs(out['TD1/mc']['kl'] - np.log(2)) < 0.02)
    out['pass'] = bool(ok)
    return out


def distil_expert(tab: E.Tab, target_pi, steps=4000):
    spec = tab.spec
    lat = D.Latent(jnp.asarray(tab.pos), jnp.asarray(tab.h), jnp.asarray(tab.il.astype(bool)))
    X = D.state_vec(spec, lat)
    reach = tab.occupancy(tab.behaviour(np.full((tab.B, 4), 0.25)))[0] > 0
    w = jnp.asarray(reach.astype(np.float32))
    tgt = jnp.asarray(target_pi, jnp.float32)
    pol = M.make_policy('flat')
    p = pol.init(jax.random.PRNGKey(123), X[0])
    opt = optax.adam(3e-3)
    st = opt.init(p)

    def loss(p):
        lp = pol.apply(p, X)
        kl = jnp.sum(jnp.where(tgt > 0, tgt * (jnp.log(jnp.maximum(tgt, 1e-30)) - lp), 0.0), -1)
        return jnp.sum(w * kl) / jnp.sum(w)

    @jax.jit
    def stepf(p, st):
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l
    for _ in range(steps):
        p, st, l = stepf(p, st)
    probs = np.asarray(jnp.exp(pol.apply(p, X)), np.float64)
    probs /= probs.sum(1, keepdims=True)
    return p, probs, float(l)


def check_ail_vs_exact(seeds=(0, 1, 2, 3, 4), iters=100):
    out = {}
    for name in ['FL', 'TD']:
        tab = E.Tab(name)
        pi_star = E.maxent_expert(tab)
        p, probs, dl = distil_expert(tab, pi_star)
        fp = tab.ail_fixed_point(probs, iters=80, gamma_w=1.0)       # undiscounted visitation = what AIL samples
        fp_disc = tab.ail_fixed_point(probs, iters=80, gamma_w=D.GAMMA)
        path = os.path.join(OUT, f'expert_{name}.npz')
        stacked = jax.tree_util.tree_map(lambda x: jnp.stack([x] * len(seeds)), p)
        T.save_expert(path, stacked, list(seeds))
        cfg = T.Config(**dict(T.ARMS['ail_compact'], env=name, iters=iters, expert_ckpt=path))
        curves = T.run('toys', 'ail_compact_distilled', cfg, list(seeds), out_root=OUT)
        rets = np.array([np.mean([np.asarray(c['learner_sto/ret'])[i] for c in curves[-5:]]) for i in range(len(seeds))])
        kls = np.array([np.mean([np.asarray(c['learner_sto/kl_clean'])[i] for c in curves[-5:]])
                        for i in range(len(seeds))])
        m, se = rets.mean(), rets.std(ddof=1) / np.sqrt(len(rets))
        tol = max(3 * se, 0.05 * abs(fp['ret_sto']))
        ok_r = abs(m - fp['ret_sto']) <= tol
        ok_k = abs(kls.mean() - fp['kl']) <= 0.25 * fp['kl']
        out[name] = {'distil_kl': dl, 'exact_fp_undisc': {'ret_sto': fp['ret_sto'], 'ret_greedy': fp['ret_greedy'],
                                                          'kl': fp['kl']},
                     'exact_fp_disc': {'ret_sto': fp_disc['ret_sto'], 'kl': fp_disc['kl']},
                     'ours_ret_last5': rets.tolist(), 'ours_ret_mean': m, 'ours_ret_se': se, 'tol': tol,
                     'ours_kl_clean': kls.tolist(), 'ours_kl_mean': float(kls.mean()),
                     'pass_ret': bool(ok_r), 'pass_kl': bool(ok_k), 'pass': bool(ok_r and ok_k)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    jaxlog.check_device()
    res = {}
    t = time.time()
    res['i_oracle_tests'] = check_oracle_tests(); jaxlog.info('oracle', **res['i_oracle_tests'])
    res['ii_exact_table'] = check_table(); jaxlog.info('table', ok=all(v['pass'] for v in res['ii_exact_table'].values()))
    res['iv_td1_gradient'] = check_td1_gradient(); jaxlog.info('td1 grad', ok=res['iv_td1_gradient']['pass'])
    res['v_exact_a2d'] = check_exact_a2d(); jaxlog.info('exact a2d', ok=res['v_exact_a2d']['pass'])
    res['iii_ail_vs_exact'] = check_ail_vs_exact(); jaxlog.info('ail vs exact',
                                                                **{k: v['pass'] for k, v in res['iii_ail_vs_exact'].items()})
    res['PASS'] = bool(res['i_oracle_tests']['pass'] and all(v['pass'] for v in res['ii_exact_table'].values())
                       and res['iv_td1_gradient']['pass'] and res['v_exact_a2d']['pass']
                       and all(v['pass'] for v in res['iii_ail_vs_exact'].values()))
    res['secs'] = round(time.time() - t, 1)
    jaxlog.write_json(os.path.join(OUT, 'toys.json'), res)
    jaxlog.metric('TOYS', PASS=res['PASS'], secs=res['secs'])


if __name__ == '__main__':
    main()
