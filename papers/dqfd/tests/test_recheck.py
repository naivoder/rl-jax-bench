"""Environment and replay re-check tests:

* the flag-gated Asterix spawn repair makes a COMPETENT policy (the shipped PPO expert) behave exactly like the
  original numpy MinAtar past the defect region (>= 927 steps of one episode, many full-slot spawns), while the shipped
  (unrepaired) env deviates on the same stream; and the repair is OFF by default.
* policy-driven long-episode parity for Breakout, SpaceInvaders and Seaquest (PPO demonstrators).
* brute-force n-step targets on a WRAPPED agent ring with terminations AND truncations, plus the pending set.
* the installed gymnax keeps the time cap out of termination (guard in MinAtar.__init__).
The policy-driven tests need the original MinAtar source (sh scripts/official/setup.sh) and the shipped PPO demonstrators in
data/demos; they skip without them.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import replay as RP
from src.envs import MinAtar

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _need(game, who="expert"):
    from record_official_env import SRC
    DEMO_DIR = os.path.join(REPO, "data", "demos")
    fn = f"ppo_{game}.msgpack" if who != "subopt" else f"ppo_{game}_subopt.msgpack"
    if not os.path.isdir(SRC) or not os.path.exists(os.path.join(DEMO_DIR, fn)):
        pytest.skip(f"original MinAtar source not found at {SRC} (run: sh scripts/official/setup.sh) or PPO demonstrator {fn} missing from data/demos")


# ------------------------------------------------------------------------------------------------ Asterix spawn repair
def test_asterix_repair_default_off_and_rejected_elsewhere():
    assert MinAtar("Asterix").asterix_spawn_repair is False
    with pytest.raises(ValueError):
        MinAtar("Breakout", asterix_spawn_repair=True)


def test_asterix_repair_full_slots_keeps_entities():
    """With all 8 rows occupied the repaired spawn leaves every entity in place (gymnax deletes one)."""
    m0, m1 = MinAtar("Asterix", max_steps=10 ** 6), MinAtar("Asterix", max_steps=10 ** 6, asterix_spawn_repair=True)
    _, st = m0.reset(jax.random.PRNGKey(0))
    ents = jnp.array([[3, i + 1, i % 2, 0, 1] for i in range(8)])
    st = st.replace(entities=ents, spawn_timer=0, move_timer=5)   # no entity moves this step
    kept0 = kept1 = 0
    for s in range(20):
        _, s0, *_ = m0.step(jax.random.PRNGKey(s), st, 0)
        _, s1, *_ = m1.step(jax.random.PRNGKey(s), st, 0)
        kept0 += int(np.asarray(s0.entities)[:, 4].sum() == 8)
        kept1 += int(np.asarray(s1.entities)[:, 4].sum() == 8)
        assert int(s1.spawn_timer) == int(s0.spawn_timer)        # timer path unchanged
    assert kept1 == 20 and kept0 == 0


def test_asterix_repaired_matches_original_under_competent_policy():
    _need("Asterix")
    from policy_parity import run
    res = run("Asterix", "expert", T=1600, seed=1, repair=True)
    assert res["bad_obs"] == 0 and res["bad_r"] == 0 and res["bad_t"] == 0, res
    assert res["ep_len_max"] >= 927 and res["max_ramp_index"] >= 9 and res["asterix_full_spawns"] >= 1, res


def test_asterix_unrepaired_deviates_under_competent_policy():
    _need("Asterix")
    from policy_parity import run
    res = run("Asterix", "expert", T=1600, seed=1, repair=False, stop_at_bad=True)
    assert res["first_bad"] is not None and res["first_bad"]["ep_step"] >= 900, res


# ------------------------------------------------------------------------------------------------ policy-driven parity
@pytest.mark.parametrize("game,who,T,seed", [("Breakout", "expert", 2000, 1), ("Breakout", "tracker", 10000, 4),
                                             ("SpaceInvaders", "expert", 3000, 1), ("Seaquest", "expert", 3000, 1)])
def test_policy_driven_parity(game, who, T, seed):
    _need(game)
    from policy_parity import run
    res = run(game, who, T=T, seed=seed)
    assert res["bad_obs"] == 0 and res["bad_r"] == 0 and res["bad_t"] == 0, res
    if who == "tracker":
        assert res["brick_respawns"] >= 1, res                     # late-game Breakout mechanic exercised
    assert res["ep_len_max"] > 100, res                            # far longer than the random-action fixtures


# ------------------------------------------------------------------------------------------------ ring n-step brute force
def _tag(i):
    return np.frombuffer(np.int32(i).tobytes(), np.uint8)


def test_ring_nstep_truncation_and_pending_set():
    rng = np.random.default_rng(11)
    nd, cap, n, g = 23, 37, 10, 0.99
    L = RP.make_layout(nd, cap, 4, n_step=n, gamma=g, alpha=0.4, bs=8)
    dend = np.zeros(nd, bool); dterm = np.zeros(nd, bool)
    dend[[6, 15, nd - 1]] = True; dterm[[6, nd - 1]] = True             # one demo episode ends by truncation (15)
    drew = rng.normal(size=nd).astype(np.float32)
    demo = dict(obs=jnp.asarray(np.stack([_tag(i) for i in range(nd)])), nobs=jnp.asarray(np.stack([_tag(10_000 + i) for i in range(nd)])),
                act=jnp.zeros(nd, jnp.int32), rew=jnp.asarray(drew), term=jnp.asarray(dterm), end=jnp.asarray(dend))
    buf = RP.init_buffer(L, demo, dict(demo_on=1.0, eps_d=1.0, eps_a=0.001, rbs=0.0))
    T, seq = 260, []
    for t in range(T):
        u = rng.random(); term = u < 0.05; trunc = (not term) and u < 0.09
        r = float(rng.normal()); seq.append((r, term, term or trunc))
        buf = RP.add_transition(L, buf, jnp.asarray(_tag(t)), jnp.asarray(_tag(20_000 + t)), 0, jnp.float32(r), term, term or trunc)
    assert sum(1 for _, te, en in seq if en and not te) >= 5 and sum(1 for _, te, _ in seq if te) >= 5

    def brute(t):
        G, m = 0.0, 0
        for k in range(n):
            if t + k >= T:
                return None
            r, term, end = seq[t + k]
            G += g ** k * r; m += 1
            if end:
                return G, (0.0 if term else g ** m), 20_000 + t + k
        return G, g ** n, 20_000 + t + n - 1

    pa = np.asarray(buf["pa"])
    present = list(range(T - cap, T))
    idx = np.array([nd + (t % cap) for t in present])
    out = RP.gather_batch(L, buf, jnp.asarray(idx))
    Gn, disc, nob = np.asarray(out["Gn"]), np.asarray(out["disc_n"]), np.asarray(out["nobsn"])
    n_active = 0
    for j, t in enumerate(present):
        b = brute(t)
        assert (pa[idx[j]] > 0) == (b is not None), t                  # pending set == incomplete windows
        if b is not None:
            n_active += 1
            assert abs(Gn[j] - b[0]) < 1e-5 and abs(disc[j] - b[1]) < 1e-6, (t, Gn[j], disc[j], b)
            assert int(np.frombuffer(nob[j].tobytes(), np.int32)[0]) == b[2]
    assert n_active == cap - (n - 1)
    outd = RP.gather_batch(L, buf, jnp.arange(nd))
    for i in range(nd):
        G, d, last = 0.0, None, None
        for k in range(n):
            G += g ** k * drew[i + k]
            if dend[i + k]:
                d = 0.0 if dterm[i + k] else g ** (k + 1); last = i + k; break
        if d is None:
            d, last = g ** n, i + n - 1
        assert abs(float(outd["Gn"][i]) - G) < 1e-5 and abs(float(outd["disc_n"][i]) - d) < 1e-6
        assert int(np.frombuffer(np.asarray(outd["nobsn"])[i].tobytes(), np.int32)[0]) == 10_000 + last


# ------------------------------------------------------------------------------------------------ time cap is truncation
@pytest.mark.parametrize("game", ["Breakout", "SpaceInvaders", "Asterix", "Seaquest"])
def test_time_cap_is_truncation_not_termination(game):
    m = MinAtar(game, max_steps=50)                                    # the guard in __init__ already ran
    _, st = m.reset(jax.random.PRNGKey(3))
    st = st.replace(time=49)
    _, st2, _, term, trunc = m.step(jax.random.PRNGKey(4), st, 0)
    assert not bool(term) and bool(trunc)
    _, st = m.reset(jax.random.PRNGKey(3))
    _, _, _, term, trunc = m.step(jax.random.PRNGKey(4), st, 0)        # far from the cap: neither
    assert not bool(term) and not bool(trunc)


def test_time_cap_guard_raises_on_cap_as_termination(monkeypatch):
    from gymnax.environments.minatar import breakout as B
    orig = B.MinBreakout.step_env

    def step_env_cap_terminates(self, key, state, action, params):
        o, s, r, d, i = orig(self, key, state, action, params)
        return o, s, r, d | (s.time >= params.max_steps_in_episode), i     # stock-1.0.0-style done

    monkeypatch.setattr(B.MinBreakout, "step_env", step_env_cap_terminates)
    with pytest.raises(RuntimeError):
        MinAtar("Breakout", max_steps=50)
