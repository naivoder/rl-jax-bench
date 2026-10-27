"""Lock checks: the combination lock, exactly computable, plus the mechanism checks Lock-e / Lock-f.

Lock-a  closed form vs EXACT enumeration (fractions, real lock rollouts) of the explore-then-commit oracle: abs error < 1e-12
Lock-b  Monte Carlo (>= 1e6 trajectories) of the same algorithm within 4 SE of the closed form on every row
Lock-c  from-scratch first-reward episode count: mean A^H within 4 SE (Theorem 4.1)
Lock-d  coverage variant (guide takes a* w.p. p): closed form vs exact enumeration and Monte Carlo
Lock-e  the switching rule used by the GPU code: the guide acts at exactly t < h, steps are counted and stored (src/algorithm.py on the lock)
Lock-f  stage list rounding, and the beta state machine (JAX) vs a python reference and vs a line-by-line transcription of steventango's
      JSRLAfterEvalCallback (rolling-window variant).
Run: python -m pytest tests/test_lock.py -q   (CPU; ~2-4 min).  Numbers are written to runs/toys/lock.json by scripts/toys_report.py.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import lock as K  # noqa: E402
from src.curriculum import CurRef, cur_init, cur_step, stage_list  # noqa: E402

ROWS = [(10, 2, 10), (10, 2, 5), (8, 2, 4), (10, 3, 10), (20, 2, 10)]   # lock rows (H, A, m)


# ------------------------------------------------------------------------------------------------ Lock-a
@pytest.mark.parametrize("H,A,m", [(1, 2, 1), (2, 2, 2), (3, 2, 3), (3, 3, 2), (4, 2, 3), (4, 2, 4), (5, 2, 2), (6, 2, 2), (3, 3, 3), (6, 3, 1)])
def test_g1a_closed_form_equals_exact_enumeration(H, A, m):
    ex = K.exact_etc(H, A, m)
    assert abs(float(ex) - K.p_opt_closed(H, A, m)) < 1e-12, (float(ex), K.p_opt_closed(H, A, m))


def test_g1a_bound_Hq():
    """V* - E[V^pi] = 1 - (1-q)^H <= H q  (Theorem A.3 with C = 1, fbar = q).  An implementation check of the ETC argmax oracle (NOT Algorithm 2 with
    the Corollary-A.4 oracle): the C-scaling and the T^-1/3 / H^(7/3) rates are measured by scripts/alg2.py."""
    for H, A, m in ROWS:
        q = K.q_stage(A, m)
        assert 1 - K.p_opt_closed(H, A, m) <= H * q + 1e-15


# ------------------------------------------------------------------------------------------------ Lock-d
@pytest.mark.parametrize("H,A,m,p", [(3, 2, 3, "0.7"), (4, 2, 2, "0.5"), (3, 3, 2, "0.9")])
def test_g1d_coverage_exact(H, A, m, p):
    from fractions import Fraction
    ex = K.exact_etc(H, A, m, Fraction(p))
    assert abs(float(ex) - K.p_opt_cov_closed(H, A, m, float(Fraction(p)))) < 1e-12


def test_g1d_coverage_value_and_C():
    assert abs(K.p_opt_cov_closed(6, 2, 20, 0.7) - 0.866357) < 5e-7          # closed form, value derived by hand
    assert abs(0.7 ** (-(6 - 1)) - 5.95) < 0.01                               # C = p^{-(H-1)}


# ------------------------------------------------------------------------------------------------ Lock-b / Lock-d MC
def _mc(H, A, m, N, seed, p=1.0, chunk=200_000):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N // chunk):
        hits += int(K.mc_etc(H, A, m, chunk, rng, p).sum())
    pe = hits / N
    return pe, float(np.sqrt(pe * (1 - pe) / N))


@pytest.mark.parametrize("row", range(len(ROWS)))
def test_g1b_monte_carlo(row):
    H, A, m = ROWS[row]
    exact = K.p_opt_closed(H, A, m)
    pe, se = _mc(H, A, m, 1_000_000, 20261027 + row)
    assert abs(pe - exact) < 4 * se, (ROWS[row], pe, exact, se)


def test_g1d_monte_carlo():
    exact = K.p_opt_cov_closed(6, 2, 20, 0.7)
    pe, se = _mc(6, 2, 20, 1_000_000, 20261027 + 90, p=0.7)
    assert abs(pe - exact) < 4 * se, (pe, exact, se)


# ------------------------------------------------------------------------------------------------ Lock-c
@pytest.mark.parametrize("H,A,n", [(6, 2, 100_000), (8, 2, 100_000), (5, 3, 100_000)])
def test_g1c_scratch_first_reward(H, A, n):
    rng = np.random.default_rng(20261027 + 50 + H)
    ep = K.scratch_first_reward(H, A, n, rng)
    mean, se = ep.mean(), ep.std(ddof=1) / np.sqrt(n)
    assert abs(mean - A ** H) < 4 * se, (mean, A ** H, se)


def test_g1c_scratch_vs_jsrl_scale():
    """The contrast of the theorem: A^H episodes for 0-initialised exploration vs H*A-ish for JSRL's stage-wise search."""
    H, A = 10, 2
    assert A ** H == 1024 and H * A == 20
    T = 100
    assert 1 - (1 - A ** -H) ** T < 0.1 < K.p_opt_closed(H, A, 10)             # same T = 100 episodes: <= 0.093 vs 0.995


# ------------------------------------------------------------------------------------------------ Lock-f stage list
@pytest.mark.parametrize("h1", [80.546, 177.0, 640.0, 2837.0, 178.97, 2601.08, 583.37])
def test_g1f_stage_list(h1):
    n = 10
    s = stage_list(h1, n)
    assert len(s) == n + 1 and s[0] == int(np.floor(h1 + 0.5)) and s[-1] == 0
    assert all(a >= b for a, b in zip(s, s[1:]))
    steps = np.diff(s)
    assert np.all(np.abs(-steps - h1 / n) <= 1.0 + 1e-9)                       # equal decrements up to rounding


def test_g1f_stage_list_known():
    assert stage_list(80.546, 10) == [81, 73, 65, 57, 49, 41, 32, 24, 16, 8, 0]   # Breakout (guide_stats.py)
    assert stage_list(100, 5) == [100, 80, 60, 40, 20, 0]


# ------------------------------------------------------------------------------------------------ Lock-f beta state machine
class StevRef:
    """Line-by-line transcription of steventango/jumpstart-rl e5f690fe src/jsrl/jsrl.py JSRLAfterEvalCallback._on_step (curriculum
    branch) + JSRLPolicy.update_horizon, with the SB3 plumbing removed.  `R` = parent.last_mean_reward.  Rolling window, never reset."""

    def __init__(self, window_size, tolerance, n_horizons):
        self.window_size, self.tolerance = window_size, tolerance
        self.horizon_step, self.n_horizons = 0, n_horizons
        self.best_moving_mean_reward = -np.inf
        self.tolerated_moving_mean_reward = -np.inf
        self.mean_rewards = np.full(window_size, -np.inf, dtype=np.float32)
        self.advanced = False

    def on_step(self, R):
        self.advanced = False
        self.mean_rewards = np.roll(self.mean_rewards, 1)
        self.mean_rewards[0] = R
        moving_mean_reward = np.mean(self.mean_rewards)
        horizon_is_zero = self.horizon_step >= self.n_horizons - 1          # horizons[horizon_step] <= 0 (last entry is 0)
        if self.mean_rewards[-1] == -np.inf or horizon_is_zero:
            return
        elif self.best_moving_mean_reward == -np.inf:
            self.best_moving_mean_reward = moving_mean_reward
        elif moving_mean_reward >= self.tolerated_moving_mean_reward:
            self.horizon_step = min(self.horizon_step + 1, self.n_horizons - 1)
            self.advanced = True
        if moving_mean_reward >= self.best_moving_mean_reward:
            self.tolerated_moving_mean_reward = moving_mean_reward - self.tolerance * np.abs(moving_mean_reward)
            self.best_moving_mean_reward = max(self.best_moving_mean_reward, moving_mean_reward)


def _run_jax(seq, M, tol, ns, reset):
    cs = cur_init(M)
    stages = []
    for R in seq:
        cs, adv = cur_step(cs, np.float32(R), tol, ns, 1.0 if reset else 0.0, True)
        stages.append(int(cs["stage"]))
    return stages


def _run_ref(seq, M, tol, ns, reset):
    r = CurRef(M, tol, ns, reset)
    out = []
    for R in seq:
        r.step(R); out.append(r.stage)
    return out


def _seqs(rng, T=140):
    """hand-built and random evaluation sequences: stall, monotone up, noisy plateau, up-then-collapse, constant, random walk."""
    t = np.arange(T)
    return {
        "stall_down": 10.0 - 0.05 * t,
        "monotone_up": 1.0 + 0.1 * t,
        "constant": np.full(T, 5.0),
        "noisy_plateau": 6.0 + rng.normal(0, 0.4, T),
        "up_then_collapse": np.where(t < 60, 1 + 0.1 * t, 2.0 + 0 * t),
        "random_walk": np.cumsum(rng.normal(0, 0.3, T)) + 8.0,
        "zeros": np.zeros(T),
    }


@pytest.mark.parametrize("M", [1, 3, 5])
@pytest.mark.parametrize("tol", [0.0, 0.05, 0.15])
def test_g1f_beta_jax_equals_python_reference(M, tol):
    rng = np.random.default_rng(7 + M)
    for name, seq in _seqs(rng).items():
        seq = seq.astype(np.float32)
        for reset in (True, False):
            assert _run_jax(seq, M, tol, 10, reset) == _run_ref(seq, M, tol, 10, reset), (name, M, tol, reset)


@pytest.mark.parametrize("M", [1, 3, 5])
@pytest.mark.parametrize("tol", [0.0, 0.05, 0.15])
def test_g1f_beta_equals_steventango_rolling(M, tol):
    """Rolling-window variant (reset_on_adv = 0) of our state machine == steventango's callback, stage by stage."""
    rng = np.random.default_rng(11 + M)
    for name, seq in _seqs(rng).items():
        seq = seq.astype(np.float32)
        st = StevRef(M, tol, 11)
        ref = []
        for R in seq:
            st.on_step(np.float32(R)); ref.append(st.horizon_step)
        assert _run_jax(seq, M, tol, 10, False) == ref, (name, M, tol)


def test_g1f_beta_semantics():
    """Hand-checked: M = 3, tol = 0.1, stage advances only after the first MA sets the best, one advance per evaluation, window resets."""
    seq = [10, 10, 10,  10,  10, 10, 10,  9.2, 9.2, 9.2,  9.2]
    st = _run_ref(seq, 3, 0.1, 10, True)
    # evals 1-3: first MA (10) sets the best; eval 4: window of 3 (10,10,10)>=9 -> advance to stage 1 and reset;
    # evals 5-7 fill the new window (n<3 at 5,6 -> no advance); eval 7: MA 10 >= 9 -> advance to 2; evals 8-10 fill; eval 10: MA 9.2 >= 9 -> advance to 3
    assert st == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3], st
    # a stall: combined return drops below (1 - tol) * best forever -> never advances past stage 1
    seq2 = [10, 10, 10, 10] + [8.0] * 30
    assert max(_run_ref(seq2, 3, 0.1, 10, True)) == 1
    # tol = 0 requires re-attaining the best
    assert max(_run_ref([10, 10, 10, 10] + [9.99] * 30, 3, 0.0, 10, True)) == 1


def test_g1f_never_past_last_stage():
    s = _run_jax(np.full(400, 5.0, np.float32), 5, 0.05, 10, True)
    assert max(s) == 10 and s[-1] == 10


# ------------------------------------------------------------------------------------------------ Lock-e: the switching rule on the lock
def _lock_setup(H=8, A=8, steps=400, mode=1, h_stage0=5, seed=0):
    import jax, jax.numpy as jnp, yaml
    from src import algorithm as AL, replay as R
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src/configs/default.yaml")))
    env = K.LockEnv(H, A)
    L = R.make_layout(0, 1024, 25, cfg["dqfd"]["n_step"], cfg["env"]["gamma"], cfg["per"]["alpha"], 256)
    fns = AL.Fns(env, L, cfg, steps, env.guide_fn, chunk=steps)
    hl = [h_stage0] + [0] * 10
    hp = {k: (jnp.asarray(v, jnp.float32)) for k, v in AL.arm_hp("PDD", cfg).items()}
    hp = dict(hp, jsrl_mode=jnp.float32(mode), h_list=jnp.asarray([float(x) for x in hl[:11]], jnp.float32),
              learn_start=jnp.float32(10 ** 9),   # no learning: pi_e is the random-init net + eps (eps = 1 at t = 0 -> uniform actions)
              eps_decay=jnp.float32(10 ** 9))
    key = jax.random.PRNGKey(seed)
    params = fns.init_params(key)
    buf = R.init_buffer(L, None, hp)
    skey = jax.random.fold_in(key, 1005)
    c = fns.init_interact(params, params, fns.tx.init(params), buf, jax.random.fold_in(key, 2), skey, hp)
    c = fns.interact_chunk(c, hp, jax.random.fold_in(key, 3), steps, jnp.array(True))
    return env, fns, hp, c, skey


@pytest.mark.parametrize("mode", [1, 2])
def test_g1e_switching_rule_on_the_lock(mode):
    import jax, jax.numpy as jnp
    from src import algorithm as AL
    from src.envs import unpack_obs
    H, A, steps = 8, 8, 400
    env, fns, hp, c, skey = _lock_setup(H, A, steps, mode)
    assert int(c["t"]) == steps and int(c["buf"]["ptr"]) == steps          # every env step counted AND stored, guide steps included
    obs = unpack_obs(c["buf"]["obs"][:steps], env.obs_shape) > 0.5
    layer, red = env.decode(obs)
    layer = np.asarray(layer); act = np.asarray(c["buf"]["act"][:steps])
    assert (layer == np.arange(steps) % H).all()                            # episodes are H steps, within-episode index = layer
    ep = np.arange(steps) // H
    hl = np.asarray(hp["h_list"])
    h_ep = np.array([int(AL.episode_h(dict(hp, jsrl_mode=jnp.float32(mode)), jnp.int32(0), skey, int(e))) for e in range(steps // H)])
    if mode == 1:
        assert (h_ep == int(hl[0])).all()
    else:
        assert set(h_ep.tolist()) <= set(int(x) for x in hl) and len(set(h_ep.tolist())) > 1
    guide_act = np.asarray(jax.vmap(env.guide_fn, in_axes=(0, None))(jnp.asarray(obs), None))
    use_g = layer < h_ep[ep]
    # (1) wherever t < h the executed action IS the guide's action, exactly
    assert (act[use_g] == guide_act[use_g]).all()
    # (2) wherever t >= h the action is pi_e's (uniform at eps = 1): it is NOT the guide's action more often than chance would allow
    agree = (act[~use_g] == guide_act[~use_g]).mean()
    assert agree < 2.0 / A + 0.1, agree
    # (3) bookkeeping: the guide-step counter equals the number of t < h steps
    assert int(c["guide_steps"]) == int(use_g.sum())


def test_g1e_jsrl_act_unit():
    import jax.numpy as jnp
    from src.algorithm import jsrl_act
    for h in (0, 1, 5):
        out = [int(jsrl_act(jnp.int32(t), jnp.int32(h), jnp.int32(7), jnp.int32(3))) for t in range(8)]
        assert out == [7] * h + [3] * (8 - h)


def test_g1e_combined_evaluation_h_equals_H_is_the_guide():
    """evaluate(): with h_thr = H the 'combined' episodes are the pure guide, which fails the last step of the lock -> return exactly 0;
    and the pi_e-alone episodes are not contaminated by it."""
    import jax, jax.numpy as jnp
    env, fns, hp, c, skey = _lock_setup(8, 2, 64, 1)
    ev = fns.evaluate(c["params"], jnp.int32(8), jax.random.PRNGKey(5))
    assert float(ev["thr"]) == 0.0
    ev0 = fns.evaluate(c["params"], jnp.int32(0), jax.random.PRNGKey(5))
    assert float(ev0["alone"]) == float(ev["alone"])                       # same keys, same pi_e-alone episodes regardless of h_thr


# ------------------------------------------------------------------------------------------------ Algorithm 2 literal
def test_alg2_eps_greedy_vectorised_vs_python_reference():
    """The vectorised eps-greedy Algorithm 2 (`alg2_eps_greedy`) against an independent per-learner python implementation (explicit rollouts
    with sampled round indices for the suffix and for the final V): mean V over many learners agrees within 4 SE (H = 3, A = 2, T = 60, p = 1 and 0.8)."""
    for p in (1.0, 0.8):
        rng = np.random.default_rng(11)
        sub, lsum, q = K.alg2_eps_greedy(3, 2, 60, 6000, rng, p=p)
        v_vec = 1.0 - sub
        ref = np.array([K.alg2_reference(3, 2, 60, np.random.default_rng(1000 + i), p=p) for i in range(250)])
        se = np.sqrt(v_vec.var() / len(v_vec) + ref.var(ddof=1) / len(ref))
        assert abs(v_vec.mean() - ref.mean()) < 4 * se + 1e-3, (p, v_vec.mean(), ref.mean(), se)
        assert (lsum >= sub - 1e-12).all() and (q >= 0).all() and (q <= 1).all()       # the Theorem-A.3 sum is an upper bound on subopt


def test_alg2_more_rounds_help_and_deterministic_for_a_seed():
    a = K.alg2_eps_greedy(4, 2, 80, 2000, np.random.default_rng(5))[0].mean()
    b = K.alg2_eps_greedy(4, 2, 800, 2000, np.random.default_rng(5))[0].mean()
    assert b < a and a == K.alg2_eps_greedy(4, 2, 80, 2000, np.random.default_rng(5))[0].mean()
