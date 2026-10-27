"""Toy checks: closed forms derived from the paper's losses, checked against the ACTUAL implementation (CPU, no GPU).
Writes a log of every number to runs/toys/toy_numbers.json at the end of the module."""
import json
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src import algorithm as A, replay as R
from src.envs import reward_transform

LOG = {}


def test_5a_margin_value_and_subgradient():
    q = jnp.array([1.0, 0.5, 0.3]); a = jnp.array(1)
    je = A.margin_loss(q, a, 0.8)
    g = jax.grad(lambda x: A.margin_loss(x, a, 0.8))(q)
    LOG["5a"] = dict(je=float(je), grad=g.tolist())
    assert abs(float(je) - 1.3) < 1e-6 and np.allclose(g, [1, -1, 0])
    q2 = jnp.array([0.0, 1.0, 0.1])
    je2 = A.margin_loss(q2, a, 0.8); g2 = jax.grad(lambda x: A.margin_loss(x, a, 0.8))(q2)
    assert abs(float(je2)) < 1e-6 and np.allclose(g2, 0)  # margin satisfied: loss 0 and zero gradient (no tie at the max)


def test_5b_one_step_targets():
    qon = jnp.array([[1.0, 2.0, 1.5]]); qtg = jnp.array([[1.2, 0.7, 3.0]])
    y_double = A.td_targets(qon, qtg, jnp.array([0.5]), jnp.array([0.99]))
    y_max = A.td_targets(qon, qtg, jnp.array([0.5]), jnp.array([0.99]), nmax=jnp.array(1.0))
    y_term = A.td_targets(qon, qtg, jnp.array([0.5]), jnp.array([0.0]))
    LOG["5b"] = dict(double=float(y_double[0]), max=float(y_max[0]), terminal=float(y_term[0]))
    assert abs(float(y_double[0]) - 1.193) < 1e-6 and abs(float(y_max[0]) - 3.47) < 1e-6 and abs(float(y_term[0]) - 0.5) < 1e-6


def _chain_buffer(episodes, n_step, gamma, cap=4):
    """Demo buffer from [(rewards, end_kind)] with end_kind 'term' | 'trunc'; obs bytes encode the global index."""
    obs, nobs, act, rew, term, end = [], [], [], [], [], []
    k = 0
    for rs, kind in episodes:
        for j, r in enumerate(rs):
            obs.append([k]); nobs.append([k + 1]); act.append(0); rew.append(r)
            last = j == len(rs) - 1
            term.append(last and kind == "term"); end.append(last); k += 1
        k += 100  # gap so next-obs ids do not collide
    arr = lambda x, dt: jnp.asarray(np.array(x), dt)
    demo = dict(obs=arr(obs, jnp.uint8), nobs=arr(nobs, jnp.uint8), act=arr(act, jnp.int32), rew=arr(rew, jnp.float32),
                term=arr(term, bool), end=arr(end, bool))
    L = R.make_layout(len(act), cap, 1, n_step, gamma, 0.4, bs=4)
    hp = dict(demo_on=jnp.array(1.0), eps_d=jnp.array(1.0), eps_a=jnp.array(0.001), rbs=jnp.array(0.0))
    return L, R.init_buffer(L, demo, hp)


def test_5c_nstep_return():
    g = 0.9
    # (i) game over after r_{t+2}; (ii) continuing; (iii) game over after r_{t+1}; (iv) truncation after r_{t+1}
    L, buf = _chain_buffer([([1., 0., 2.], "term"), ([1., 0., 2., 5.], "trunc"), ([1., 0.], "term"), ([1., 0.], "trunc")], 3, g)
    b = R.gather_batch(L, buf, jnp.array([0, 3, 7, 9]))
    Q = 1.0
    G = np.asarray(b["Gn"]) + np.asarray(b["disc_n"]) * Q
    LOG["5c"] = dict(G=G.tolist(), m=np.asarray(b["m"]).tolist())
    assert np.allclose(G, [2.62, 3.349, 1.0, 1.81], atol=1e-5), G
    assert np.asarray(b["m"]).tolist() == [3, 3, 2, 2]
    # paper constants: ten rewards of 1, bootstrap 0: (1 - 0.99^10) / 0.01
    L2, buf2 = _chain_buffer([([1.0] * 12, "term")], 10, 0.99)
    b2 = R.gather_batch(L2, buf2, jnp.array([0]))
    assert abs(float(b2["Gn"][0]) - 9.561792) < 1e-5 and abs(float(b2["disc_n"][0]) - 0.99 ** 10) < 1e-6
    # the n-step window never leaks across an episode boundary
    b3 = R.gather_batch(L, buf, jnp.array([2]))
    assert float(b3["Gn"][0]) == 2.0 and float(b3["disc_n"][0]) == 0.0


def test_5d_per_probabilities_weights_and_montecarlo():
    L = R.make_layout(2, 2, 1, 3, 0.99, 0.4, bs=4)
    hp = dict(demo_on=jnp.array(1.0), eps_d=jnp.array(1.0), eps_a=jnp.array(0.001), rbs=jnp.array(0.0))
    demo = dict(obs=jnp.zeros((2, 1), jnp.uint8), nobs=jnp.zeros((2, 1), jnp.uint8), act=jnp.zeros(2, jnp.int32),
                rew=jnp.zeros(2), term=jnp.zeros(2, bool), end=jnp.zeros(2, bool))
    buf = R.init_buffer(L, demo, hp)
    buf["is_demo"] = buf["is_demo"].at[2:4].set(False)
    buf = R.update_priorities(L, buf, jnp.arange(4), jnp.array([0.5, 0.0, 1.0, 2.0]), jnp.ones(4, bool), hp)
    pa = np.asarray(buf["pa"])[:4]
    P = pa / pa.sum()
    assert np.allclose(pa, [1.176079, 1.0, 1.0004, 1.319772], atol=1e-5)
    assert np.allclose(P, [0.261569, 0.222408, 0.222496, 0.293527], atol=1e-5)
    w = np.asarray(R.is_weights(jnp.asarray(P), 4.0, 0.6, jnp.ones(4, bool)))
    raw = (4 * P) ** -0.6
    assert np.allclose(raw, [0.973223, 1.072690, 1.072433, 0.908186], atol=1e-5)
    assert np.allclose(w, raw / raw.max(), atol=1e-6) and np.allclose(w, [0.907273, 1.0, 0.99976, 0.846644], atol=1e-5)
    demo_share = P[:2].sum(); up = demo_share / 0.5
    assert abs(demo_share - 0.483976) < 1e-5 and abs(up - 0.967953) < 1e-5
    # Monte Carlo (1e6 draws) through the real two-level sampler
    idx, _, _, valid = jax.jit(lambda k: R.sample(L, buf["pa"], k, 1_000_000))(jax.random.PRNGKey(0))
    freq = np.bincount(np.asarray(idx), minlength=L.npad)[:4] / 1e6
    LOG["5d"] = dict(P=P.tolist(), freq=freq.tolist(), max_abs_err=float(np.abs(freq - P).max()), up_sample=float(up))
    assert np.abs(freq - P).max() < 3e-3
    # derived bound: U -> 1/eps_a^alpha = 15.85 as N_a/N_d -> inf with |delta| -> 0
    assert abs(1 / 0.001 ** 0.4 - 15.8489) < 1e-3


def _tabular_run(lam_e, steps=60000, lr=0.05, tau=50):
    """Pre-train a tabular Q (3 states x 2 actions) on the 3-transition demo chain with the REAL loss code."""
    g, n = 0.9, 2
    L, buf = _chain_buffer([([0., 0., 1.], "term")], n, g)
    batch = R.gather_batch(L, buf, jnp.arange(3))
    hp = dict(lam_td=1.0, lam_n=1.0, lam_e=lam_e, lam_ce=0.0, margin=0.8, nmax=0.0)
    sid = lambda o: o[:, 0].astype(jnp.int32) % 100  # state id = global index mod 100 (s0,s1,s2 -> 0,1,2; terminal next ids unused)
    sid = lambda o: jnp.clip(o[:, 0].astype(jnp.int32), 0, 2)
    s, s1, sn = sid(batch["obs"]), sid(batch["nobs1"]), sid(batch["nobsn"])
    q0 = jnp.zeros((3, 2)).at[1, 1].set(5.0).at[2, 1].set(3.0)

    def loss(q, qt):
        t = A.dqfd_losses(q[s], jax.lax.stop_gradient(q[s1]), qt[s1], jax.lax.stop_gradient(q[sn]), qt[sn], batch, hp, g)
        return jnp.mean(A.total_per_sample(t, hp))

    def body(c, i):
        q, qt = c
        q = q - lr * jax.grad(loss)(q, qt)
        qt = jnp.where((i + 1) % tau == 0, q, qt)
        return (q, qt), None

    (q, _), _ = jax.lax.scan(body, (q0, q0), jnp.arange(steps))
    return np.asarray(q)


def test_5e_tabular_fixed_points():
    q = _tabular_run(1.0)
    LOG["5e_margin"] = q.tolist()
    assert np.abs(q[:, 0] - [0.81, 0.9, 1.0]).max() < 1e-3, q
    assert np.all(q[:, 1] <= q[:, 0] - 0.8 + 0.05 * 1.0 + 1e-3), q  # hinge only pushes down (closed form: <= V - 0.8 + lr*lam2)
    assert np.all(np.argmax(q, 1) == 0)  # greedy policy = demonstrator's
    q0 = _tabular_run(0.0)
    LOG["5e_nomargin"] = q0.tolist()
    assert np.abs(q0[:, 0] - [3.465, 1.8, 1.0]).max() < 1e-3, q0
    assert q0[1, 1] == 5.0 and q0[2, 1] == 3.0          # unseen entries untouched
    assert np.argmax(q0[1]) == 1 and np.argmax(q0[2]) == 1  # greedy picks the never-seen action
    assert q0[0, 0] / 0.81 > 4.0                        # overestimated by up to 4.3x


def test_5g_reward_transform():
    r = jnp.array([-25000., -1, 0, 1, 10, 25000])
    out = np.asarray(reward_transform(r))
    assert np.allclose(out, [-10.126671, -0.693147, 0, 0.693147, 2.397895, 10.126671], atol=1e-5)


def test_ce_vs_margin_gradient_property():
    """CE > margin loss and CE's gradient never vanishes while the hinge's does once the gap is 0.8."""
    q = jnp.array([2.0, 0.5, 0.5])
    assert float(A.ce_loss(q, jnp.array(0))) > 0 and float(A.margin_loss(q, jnp.array(0), 0.8)) == 0.0
    assert float(jnp.abs(jax.grad(lambda x: A.ce_loss(x, jnp.array(0)))(q)).sum()) > 0


def teardown_module(module):
    import os
    os.makedirs("runs/toys", exist_ok=True)
    json.dump(LOG, open("runs/toys/toy_numbers.json", "w"), indent=1)
