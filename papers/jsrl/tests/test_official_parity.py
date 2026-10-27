"""Parity with the only runnable JSRL code that exists: steventango/jumpstart-rl e5f690fe (third-party, SB3; the authors released NO code).
The fixtures tests/fixtures/stevtango_parity.npz were produced by scripts/official/record_stevtango_fixtures.py, which EXECUTES their classes
(JSRLAfterEvalCallback, JSRLPolicy.predict / update_horizon, their stage-list expression).  Their code is not copied here.
Tolerances: every comparison below is EXACT (integer stage traces / action masks); evaluation values are float32 in both.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.algorithm import jsrl_act  # noqa: E402
from src.curriculum import CurRef, cur_init, cur_step, stage_list  # noqa: E402

FX = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "stevtango_parity.npz"))


def _scan_factory(M, n_stages):
    @jax.jit
    def run(seq, tol):
        def body(cs, R):
            cs, _ = cur_step(cs, R, tol, n_stages, 0.0, True)       # reset_on_adv = 0: ST's rolling window, never reset
            return cs, cs["stage"]
        _, st = jax.lax.scan(body, cur_init(M), seq)
        return st
    return run


def test_fixture_has_advances_and_every_config():
    tr = FX["traces"]
    assert tr.shape[0] == len(FX["cfgs"]) and (np.diff(tr, axis=2) > 0).sum() > 1000
    assert (np.diff(tr, axis=2) <= 1).all(), "ST advances at most one stage per evaluation"


def test_beta_state_machine_equals_their_real_callback():
    """Our JAX state machine (rolling variant) == their REAL JSRLAfterEvalCallback + update_horizon, stage by stage, on 40 sequences x 40 configs
    (M in 1,2,3,5; tol in 0..0.5; 5 and 11 stages; negative rewards, exact ties, dyadic boundaries).  Exact."""
    cfgs, seqs, traces = FX["cfgs"], FX["seqs"], FX["traces"]
    cache = {}
    n_cmp = n_tie = 0
    for ci, (M, tol, nh) in enumerate(cfgs):
        M, nh = int(M), int(nh)
        run = cache.setdefault((M, nh), _scan_factory(M, nh - 1))
        for si in range(len(seqs)):
            ours = np.asarray(run(jnp.asarray(seqs[si]), jnp.float32(tol)))
            n_cmp += 1
            bad = np.flatnonzero(ours != traces[ci, si])
            amb = np.flatnonzero(FX["ambig"][ci, si])
            if len(bad):
                # the ONLY admissible divergence: at/after an evaluation whose MA sits on the threshold within float32 rounding (their float64
                # threshold vs our float32 one; see test_threshold_tie_is_numpy_version_dependent_in_theirs)
                assert len(amb) and bad[0] >= amb[0], (M, tol, nh, FX["names"][si], bad[:5], amb[:5])
                n_tie += 1
    assert n_cmp == 1600 and n_tie <= 3 and n_tie < n_cmp * 0.01


def test_beta_python_reference_equals_their_real_callback():
    """Our python reference CurRef (rolling variant) == their real callback; guards the reference used by tests/test_lock.py."""
    cfgs, seqs, traces = FX["cfgs"], FX["seqs"], FX["traces"]
    for ci, (M, tol, nh) in enumerate(cfgs):
        for si in range(0, len(seqs), 3):
            r = CurRef(int(M), float(tol), int(nh) - 1, reset_on_adv=False)
            out = []
            for R in seqs[si]:
                r.step(R); out.append(r.stage)
            amb = np.flatnonzero(FX["ambig"][ci, si])
            end = amb[0] if len(amb) else len(out)                      # compare up to the first float32-vs-float64 threshold tie
            assert out[:end] == traces[ci, si][:end].tolist(), (M, tol, nh, FX["names"][si])


def test_stage_list_equals_theirs_when_n_divides_H1():
    """ST: np.arange(H1, -1, -H1 // n).  Ours: n + 1 values, always ending at 0.  Identical exactly when n | H1; elsewhere recorded, not asserted equal."""
    n_eq = n_div = 0
    for H1, n, ln, lst in zip(FX["sl_h"], FX["sl_n"], FX["sl_len"], FX["sl_list"]):
        ours = stage_list(int(H1), int(n))
        assert len(ours) == n + 1 and ours[-1] == 0 and ours[0] == H1 and all(a >= b for a, b in zip(ours, ours[1:]))
        if H1 % n == 0:
            n_div += 1
            assert ln == n + 1 and ours == lst[:ln].tolist(), (H1, n, ours, lst[:ln])
            n_eq += 1
    assert n_div == n_eq and n_div > 400


def test_stage_list_theirs_is_wrong_off_divisor():
    """Characterises THEIR defect (not ours).  Their `-H1 // n` is (-H1) // n = -ceil(H1/n), so the step is ceil(H1/n).  When n does not divide H1 the
    arange never has n + 1 entries (3,286 of 3,286 such pairs on H1 <= 200, n <= 20) and in 2,780 of them it stops at a POSITIVE horizon, i.e. their curriculum
    would never reach h = 0 (pi_e alone).  Documented in the README ('Checked against')."""
    h, n, ln, lst = FX["sl_h"], FX["sl_n"], FX["sl_len"], FX["sl_list"]
    off = h % n != 0
    last = np.array([lst[i][ln[i] - 1] for i in range(len(h))])
    assert off.sum() == 3286 and (ln[off] != n[off] + 1).all()
    assert (off & (last != 0)).sum() == 2780
    # our own configuration (H1 = 81, n = 10): theirs has 10 stages 81, 72, ..., 0 (step 9), ours the paper's n + 1 = 11 values
    i = int(np.flatnonzero((h == 81) & (n == 10))[0])
    assert ln[i] == 10 and stage_list(81, 10) != lst[i][:ln[i]].tolist() and len(stage_list(81, 10)) == 11


def test_switching_rule_offset_by_one_vs_theirs():
    """Their rule is `timesteps <= horizon` (guide acts at t = 0..h); ours (paper Alg. 1 line 4) is t < h.  Recorded from their real predict():
    with training=1 their guide acts on h+1 steps; ours on h.  Even horizon 0 lets their guide act at t = 0."""
    hs, ts, sw = FX["sw_h"], FX["sw_t"], FX["sw"]
    for hi, h in enumerate(hs):
        theirs = (sw[hi, 1, 1] == 1)                                 # training, jsrl_evaluation=1 -> combined policy
        assert theirs.sum() == min(h + 1, len(ts))
        ours = np.asarray(jsrl_act(jnp.asarray(ts), jnp.int32(h), jnp.int32(1), jnp.int32(2))) == 1
        assert ours.sum() == min(h, len(ts))
        assert (ours[:] == (ts < h)).all() and (theirs == (ts <= h)).all()
        # ours == theirs shifted: ours with h+1 is theirs with h
        ours_h1 = np.asarray(jsrl_act(jnp.asarray(ts), jnp.int32(h + 1), jnp.int32(1), jnp.int32(2))) == 1
        assert (ours_h1 == theirs).all()


def test_their_pi_e_alone_evaluation_is_not_pure_at_h0():
    """In their 'not training and not jsrl_evaluation' branch horizon is forced to 0, yet `timesteps <= 0` still hands step 0 to the guide."""
    sw = FX["sw"]
    assert sw[0, 0, 0, 0] == 1 and (sw[0, 0, 0, 1:] == 2).all()
    # and the flag really switches the horizon off: horizon 60, training=0, jsrl_evaluation=0 -> same mask as horizon 0
    assert (sw[4, 0, 0] == sw[0, 0, 0]).all()
    # with training=0 and jsrl_evaluation=1 the combined policy runs
    assert (sw[4, 0, 1] == 1).all()


def test_their_random_strategy_support_is_every_integer_below_H1():
    """Their random draw is np.random.choice(max_horizon) = U{0..H1-1}: not the stage set, H1 itself never drawn."""
    d = FX["rnd"]
    assert d.min() == 0 and d.max() == 59 and len(np.unique(d)) == 60


def test_threshold_tie_is_numpy_version_dependent_in_theirs():
    """Honest accounting of the 3 / 1600 (config, sequence) pairs where exact traces differ: M = 3, tol = 0.25, MA = 2.0 exactly vs threshold
    (8/3)(1 - 1/4) = 2 exactly.  Their `tolerance * np.abs(ma)` is python-float x np.float32; under numpy 1.x that promotes to float64 (threshold
    2.0000000596 -> NO advance), under numpy 2 / NEP 50 it stays float32 (threshold rounds to 2.0 -> advance, as ours and as exact arithmetic).
    Ours follows exact arithmetic.  The fixture was recorded with numpy 1.26.4 (the reference venv pins numpy<2)."""
    amb_pairs = np.argwhere(FX["ambig"].any(axis=2))
    assert 1 <= len(amb_pairs) <= 3
    assert {FX["names"][si] for _, si in amb_pairs} <= {"integer_ties", "dyadic_ties"}


def test_their_user_eval_runs_the_combined_policy_on_training_counters():
    """issue 10a / issue 10b CONFIRMED by an instrumented run of their real JSRLAlgorithm(TD3) with a user EvalCallback (scripts/official/ref_r10_probe.py).
    (a) SB3's policy.training is False even while training, so their `horizon = 0` branch is keyed on `jsrl_evaluation`, which their eval callback sets True every step:
        the user EvalCallback evaluates the COMBINED policy at the current horizon (60 here), not pi_e alone.
    (b) the step counter fed to predict() in that evaluation is the TRAINING env's `_timesteps` (the first evaluation call saw t = 49, not 0), so the guide acts only on
        the tail of the counter window rather than on each evaluation episode's first h steps (4.7 % of evaluation calls used the guide).
    Our evaluation (src/eval.py) grades pi_e alone with its own per-episode counter, so neither defect can reach our numbers."""
    import json
    r = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "stevtango_r10_probe.json")))
    assert r["user_eval_training_flag_values"] == [False] and r["user_eval_jsrl_evaluation_values"] == [True]
    assert r["horizon_during_eval"] == [60]
    assert r["user_eval_first_30_timesteps"][0] > 0                      # not an episode-local counter
    assert 0.0 < r["user_eval_guide_used_fraction"] < 0.15


def test_their_dqn_wrapper_drops_epsilon_greedy():
    """Recorded by scripts/official/ref_dqn_eps_probe.py: with exploration_rate = 1.0 stock SB3 DQN executes a non-greedy action 55 % of the time (2 actions, ~50 %
    expected); THEIR JSRLAlgorithm(DQN) 0.25 %: its predict() overrides DQN.predict and calls policy.predict directly, so no epsilon-greedy.  Therefore their
    code is NOT usable unmodified as a DQN-family oracle (our learner is DQN-family: the earlier DQfD reproduction's PDD); issue 12 stands, learner parity stays with the earlier DQfD reproduction."""
    import json
    r = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "stevtango_dqn_eps_probe.json")))
    assert r["stock_DQN"]["frac_action_differs_from_greedy_at_eps1"] > 0.4
    assert r["jsrl_DQN"]["frac_action_differs_from_greedy_at_eps1"] < 0.02
