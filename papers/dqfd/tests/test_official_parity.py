"""Differential tests against the reference implementations.  There is NO official DQfD code;
the oracles are, in decreasing authority:

 1. the ORIGINAL numpy MinAtar (kenjyoung/MinAtar 4241f246) for the environments -- the code our gymnax-based env derives from;
 2. opendilab/DI-engine d0b21d06 `dqfd_nstep_td_error` (PyTorch) for the DQfD loss, plus a PyTorch re-implementation of our
    network and optimiser step (scripts/official/ref_torch_dqfd.py) -- second-best (third-party), labelled as such;
 3. felix-kerkhoff/DQfD 2dc59602 `deep_q_networks.py:224-225` for the J_E value (transcribed; TF not installed).

All fixtures in tests/fixtures/ were RECORDED from those codes (scripts/official/oracle_di_engine.py under the torch venv,
scripts/record_official_env.py with the original MinAtar source); no third-party code is copied into this repository.
Tolerances: float32 pipeline -> 1e-5 abs/rel on single-batch quantities, 1e-4 after 25 Adam steps; environments exact.
Declared non-equalities (not bugs): DI-engine bootstraps a TRUNCATED window with gamma^n and uses one `nstep` for both terms
(we use gamma^m and separate windows); Seaquest's TERMINAL observation differs in the two gauge channels (gymnax refills
oxygen / decrements divers at the terminating surfacing step; the original does not) -- never bootstrapped from.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src import algorithm as A
from src.model import DuelingQNet, l2_half_sumsq

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    return json.load(open(os.path.join(FIX, name)))


# ------------------------------------------------------------------------------------------------ DI-engine loss
@pytest.mark.parametrize("ci", range(6))
def test_di_engine_loss_grad_parity(ci):
    d = _load("di_engine_batches.json")
    c = d["cases"][ci]
    g, T = c["gamma"], c["T"]
    rew = np.asarray(c["rew"], np.float32)
    batch = dict(act=jnp.asarray(c["act"], jnp.int32), r1=jnp.asarray(rew[0]), term1=jnp.asarray(c["done1"]) > 0,
                 Gn=jnp.asarray((rew * (g ** np.arange(T))[:, None]).sum(0), jnp.float32),
                 disc_n=jnp.asarray((g ** T) * (1 - np.asarray(c["done"], np.float32)), jnp.float32),
                 is_demo=jnp.asarray(c["is_demo"]) > 0)
    hp = dict(lam_td=c["lam1"], lam_n=c["lamn"], lam_e=c["lame"], lam_ce=0.0, margin=c["margin"], nmax=0.0)
    a = lambda k: jnp.asarray(c[k], jnp.float32)
    w = a("w")

    def loss_fn(q):
        t = A.dqfd_losses(q, a("q1_on"), a("q1_tg"), a("qn_on"), a("qn_tg"), batch, hp, g)
        return jnp.mean(w * A.total_per_sample(t, hp)), t

    (loss, t), grad = jax.value_and_grad(loss_fn, has_aux=True)(a("q"))
    per = hp["lam_n"] * jnp.abs(t["jn"]) + hp["lam_td"] * jnp.abs(t["j1"]) + hp["lam_e"] * jnp.abs(t["je"])
    ref = np.asarray(c["per_sample"])
    assert abs(float(loss) - c["loss"]) < 1e-5 * max(1.0, abs(c["loss"]))
    assert (np.abs(np.asarray(per) - ref) / np.maximum(1.0, np.abs(ref))).max() < 1e-5
    assert np.abs(np.asarray(grad) - np.asarray(c["grad_q"])).max() < 1e-5
    # the three unweighted means (DI-engine returns them as `stats`: n-step TD, 1-step TD, J_E)
    got = [float(t["jn"].mean()), float(t["j1"].mean()), float(t["je"].mean())]
    assert np.abs(np.asarray(got) - np.asarray(c["stats"])).max() < 1e-5 * max(1.0, max(map(abs, c["stats"])))


def test_felix_kerkhoff_je_value():
    """felix-kerkhoff/DQfD deep_q_networks.py:224-225: J_E = |Q(a_E) - max_a (Q + l)|, l = 0 at a_E, 0.8 elsewhere (transcribed)."""
    rng = np.random.default_rng(1)
    q = rng.normal(size=(128, 6)).astype(np.float32); act = rng.integers(0, 6, 128)
    l = np.where(np.arange(6)[None] == act[:, None], 0.0, 0.8)
    felix = np.abs(q[np.arange(128), act] - (q + l).max(1))
    ours = np.asarray(A.margin_loss(jnp.asarray(q), jnp.asarray(act), 0.8))
    assert np.abs(felix - ours).max() < 1e-6 and (ours >= 0).all()


# ------------------------------------------------------------------------------------------------ network + optimiser
def canonical_weights(seed, A_=6, C=4, ch=16, hid=64):
    """Same function as scripts/official/oracle_di_engine.py:canonical_weights (flax layout)."""
    rng = np.random.default_rng(seed)
    n = lambda *s: (rng.normal(size=s) * (1.0 / np.sqrt(np.prod(s[:-1])))).astype(np.float32)
    b = lambda k: (0.1 * rng.normal(size=(k,))).astype(np.float32)
    return dict(conv_k=n(3, 3, C, ch), conv_b=b(ch), vh_k=n(ch * 100, hid), vh_b=b(hid), vo_k=n(hid, 1), vo_b=b(1),
                ah_k=n(ch * 100, hid), ah_b=b(hid), ao_k=n(hid, A_), ao_b=b(A_))


def flax_params(w):
    j = jnp.asarray
    return {"params": {"conv": {"kernel": j(w["conv_k"]), "bias": j(w["conv_b"])},
                       "v_hidden": {"kernel": j(w["vh_k"]), "bias": j(w["vh_b"])}, "v_out": {"kernel": j(w["vo_k"]), "bias": j(w["vo_b"])},
                       "a_hidden": {"kernel": j(w["ah_k"]), "bias": j(w["ah_b"])}, "a_out": {"kernel": j(w["ao_k"]), "bias": j(w["ao_b"])}}}


def test_network_forward_parity():
    d = _load("di_engine_net.json")
    x = (np.random.default_rng(12).random((16, 10, 10, 4)) < 0.2).astype(np.float32)
    q = np.asarray(DuelingQNet(6).apply(flax_params(canonical_weights(11)), jnp.asarray(x)))
    ref = np.asarray(d["q"], np.float32)
    assert np.abs(q - ref).max() < 1e-5 * max(1.0, np.abs(ref).max())


def test_training_trajectory_parity():
    """25 Adam steps (clip 10, lr 1e-4, eps 1.5e-4, explicit 1e-5 * 0.5 sum theta^2, target sync after step 10) on identical
    batches: torch + DI-engine loss  vs  flax/optax + src/algorithm.py.  Loss trace and final Q must agree to 1e-4."""
    d = _load("di_engine_train.json")
    G, T, Bt, steps = 0.99, 10, 32, 25
    net = DuelingQNet(6)
    params, tparams = flax_params(canonical_weights(21)), flax_params(canonical_weights(22))
    tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(1e-4, eps=1.5e-4))
    opt = tx.init(params)
    hp = dict(lam_td=1.0, lam_n=1.0, lam_e=1.0, lam_ce=0.0, margin=0.8, nmax=0.0)
    r2 = np.random.default_rng(23)
    losses = []
    for it in range(steps):
        s = (r2.random((Bt, 10, 10, 4)) < 0.2).astype(np.float32); s1 = (r2.random((Bt, 10, 10, 4)) < 0.2).astype(np.float32)
        sn = (r2.random((Bt, 10, 10, 4)) < 0.2).astype(np.float32)
        act = r2.integers(0, 6, Bt)
        rew = r2.normal(size=(T, Bt)).astype(np.float32)
        term_pos = r2.integers(0, 2 * T, Bt); done = (term_pos < T).astype(np.float32)
        for j in range(Bt):
            if term_pos[j] < T:
                rew[term_pos[j] + 1:, j] = 0.0
        done1 = (term_pos == 0).astype(np.float32); is_demo = r2.integers(0, 2, Bt).astype(np.float32); wt = r2.uniform(0.2, 1.0, Bt).astype(np.float32)
        batch = dict(act=jnp.asarray(act, jnp.int32), r1=jnp.asarray(rew[0]), term1=jnp.asarray(done1) > 0,
                     Gn=jnp.asarray((rew * (G ** np.arange(T))[:, None]).sum(0)), disc_n=jnp.asarray((G ** T) * (1 - done), jnp.float32),
                     is_demo=jnp.asarray(is_demo) > 0)
        x_next = jnp.asarray(np.concatenate([s1, sn]))
        q_tg = net.apply(tparams, x_next); q_on = net.apply(params, x_next)

        def loss_fn(p):
            q_s = net.apply(p, jnp.asarray(s))
            t = A.dqfd_losses(q_s, q_on[:Bt], q_tg[:Bt], q_on[Bt:], q_tg[Bt:], batch, hp, G)
            return jnp.mean(jnp.asarray(wt) * A.total_per_sample(t, hp)) + 1e-5 * l2_half_sumsq(p)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        upd, opt = tx.update(grads, opt, params)
        params = optax.apply_updates(params, upd)
        losses.append(float(loss))
        if it + 1 == d["sync_at"]:
            tparams = params
    ref = np.asarray(d["losses"])
    assert np.abs(np.asarray(losses) - ref).max() / np.abs(ref).max() < 1e-4, (losses[:3], ref[:3])
    x = (np.random.default_rng(12).random((16, 10, 10, 4)) < 0.2).astype(np.float32)
    qf = np.asarray(net.apply(params, jnp.asarray(x)))
    qr = np.asarray(d["q_final"], np.float32)
    assert np.abs(qf - qr).max() < 1e-4 * max(1.0, np.abs(qr).max())


# ------------------------------------------------------------------------------------------------ environments vs original MinAtar
def _replay_gymnax(game):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from record_official_env import keys_for
    from src.envs import MinAtar
    z = np.load(os.path.join(FIX, "env_parity.npz"))
    m = MinAtar(game, max_steps=10 ** 6)
    env, params = m.env, m.params
    step_key, reset_key = keys_for(game)
    step = jax.jit(env.step_env); reset = jax.jit(env.reset_env)
    acts, o_obs, o_r, o_term = z[f"{game}_act"], z[f"{game}_obs"], z[f"{game}_r"], z[f"{game}_term"]
    T = len(acts)
    nres = 0
    _, st = reset(reset_key(nres), params); nres += 1
    bad_obs = bad_obs_terminal_gauge = bad_r = bad_t = 0
    other_terminal = 0
    for t in range(T):
        obs, st, r, done, _ = step(step_key(t), st, int(acts[t]), params)
        ob = np.unpackbits(o_obs[t])[: obs.size].reshape(obs.shape).astype(bool)
        g = np.asarray(obs) > 0.5
        if not np.array_equal(g, ob):
            if o_term[t] and game == "Seaquest" and all(np.array_equal(g[..., c], ob[..., c]) for c in range(g.shape[-1]) if c not in (7, 8)):
                bad_obs_terminal_gauge += 1
            else:
                bad_obs += 1
        bad_r += int(float(r) != float(o_r[t])); bad_t += int(bool(done) != bool(o_term[t]))
        if o_term[t]:
            _, st = reset(reset_key(nres), params); nres += 1
    return dict(T=T, episodes=int(o_term.sum()), reward=float(o_r.sum()), bad_obs=bad_obs, bad_r=bad_r, bad_t=bad_t, gauge_only=bad_obs_terminal_gauge)


@pytest.mark.parametrize("game", ["Breakout", "SpaceInvaders", "Asterix", "Seaquest"])
def test_env_matches_original_minatar(game):
    res = _replay_gymnax(game)
    assert res["bad_obs"] == 0 and res["bad_r"] == 0 and res["bad_t"] == 0, res
    assert res["episodes"] > 30 and res["T"] >= 3000


def test_unrepaired_gymnax_space_invaders_deviates():
    """The two repairs in src/envs.py are necessary: the shipped gymnax 1.0.0 port diverges from the original (documented bug)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from src.envs import MinAtar
    from gymnax.environments.minatar import space_invaders as S
    m = MinAtar("SpaceInvaders", upstream_fixes=False)
    obs, st = m.env.reset_env(jax.random.PRNGKey(0), m.params)
    assert int(np.asarray(st.alien_map).sum()) == 28          # shipped reset: 4 rows x 7 columns (original: 24)
    m2 = MinAtar("SpaceInvaders")                             # repaired
    obs2, st2 = m2.reset(jax.random.PRNGKey(0))
    assert int(np.asarray(st2.alien_map).sum()) == 24


# ------------------------------------------------------------------------------------------------ replay: JAX vs the independent numpy replay of the reference run
def _ref_module():
    """Import scripts/official/ref_torch_dqfd.py WITHOUT torch (stubbed) to reuse its numpy Replay class."""
    import importlib.util
    import sys
    import types
    stub = types.ModuleType("torch"); stub.set_num_threads = lambda n: None
    nn = types.ModuleType("torch.nn"); nn.Module = object
    fn = types.ModuleType("torch.nn.functional"); stub.nn = nn; nn.functional = fn
    saved = {k: sys.modules.get(k) for k in ("torch", "torch.nn", "torch.nn.functional")}
    sys.modules.update({"torch": stub, "torch.nn": nn, "torch.nn.functional": fn})
    try:
        spec = importlib.util.spec_from_file_location("ref_torch_dqfd_stub", os.path.join(os.path.dirname(__file__), "..", "scripts", "official", "ref_torch_dqfd.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return m


def test_replay_nstep_windows_match_reference_numpy_replay():
    """Our n-step gather (src/replay.py:gather_batch) vs the reference run's independent numpy window code on a buffer with
    demo episodes, a wrapped agent ring and episode ends inside windows (all ends are terminations here: the reference run
    treats a time cap as termination, DI-engine has no truncation input)."""
    from src import replay as RP
    ref = _ref_module()
    rng = np.random.default_rng(5)
    nd, cap, n, g, obs_bytes = 40, 64, 10, 0.99, 52   # obs bytes: 13 bytes = 104 bits would do; 52 B = 416 bits >= 400
    # episodes of random length 3..14 laid over the demo region and over the ring
    def ends(m):
        e = np.zeros(m, bool); i = 0
        while i < m:
            i += rng.integers(3, 15); e[min(i - 1, m - 1)] = True
        return e
    demo = dict(obs=np.zeros((nd, 400), np.uint8), nobs=np.zeros((nd, 400), np.uint8), act=rng.integers(0, 3, nd), rew=rng.normal(size=nd).astype(np.float32),
                end=ends(nd))
    demo["term"] = demo["end"].copy()
    for i in range(nd):                                       # tag nobs[i] with its index in the first 16 bits
        demo["nobs"][i, :16] = [(i >> b) & 1 for b in range(16)]
    R = ref.Replay(demo, cap, n, g, 0.4, 1.0, 0.001, True, 400)
    L = RP.make_layout(nd, cap, obs_bytes, n_step=n, gamma=g, alpha=0.4, bs=16)
    pack = lambda bits: np.packbits(np.pad(bits, (0, obs_bytes * 8 - 400)))
    buf = RP.init_buffer(L, dict(obs=jnp.asarray(np.stack([pack(o) for o in demo["obs"]])), nobs=jnp.asarray(np.stack([pack(o) for o in demo["nobs"]])),
                                 act=jnp.asarray(demo["act"], jnp.int32), rew=jnp.asarray(demo["rew"]), term=jnp.asarray(demo["term"]), end=jnp.asarray(demo["end"])),
                         dict(demo_on=1.0, eps_d=1.0, eps_a=0.001, rbs=0.0))
    e_ring = ends(cap + 30)
    for k in range(cap + 30):                                  # wrap the ring
        tag = np.zeros(400, np.uint8); tag[:16] = [((1000 + k) >> b) & 1 for b in range(16)]
        r = float(rng.normal()); a = int(rng.integers(0, 3)); term = bool(e_ring[k])
        R.add(np.zeros(400, np.uint8), tag, a, np.float32(r), term, term)
        buf = RP.add_transition(L, buf, jnp.asarray(pack(np.zeros(400, np.uint8))), jnp.asarray(pack(tag)), a, jnp.float32(r), term, term)
    idx = np.concatenate([np.arange(nd), nd + np.arange(cap)])
    rew, done_n, sn, m = R.windows(idx)
    out = RP.gather_batch(L, buf, jnp.asarray(idx))
    Gn_ref = (rew * (g ** np.arange(n))[:, None]).sum(0)
    disc_ref = (1 - done_n) * g ** n
    np.testing.assert_allclose(np.asarray(out["Gn"]), Gn_ref, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out["disc_n"]), disc_ref, rtol=1e-6)
    assert (np.asarray(out["m"]) == m).all()
    tags_ours = np.unpackbits(np.asarray(out["nobsn"]), axis=-1)[:, :16]
    assert (tags_ours == sn[:, :16]).all()
