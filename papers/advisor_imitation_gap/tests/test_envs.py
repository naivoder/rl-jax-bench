"""Env-internal equivalences that do not need official fixtures."""
import jax
import jax.numpy as jnp
import numpy as np


def test_vectorised_process_vis_equals_sequential():
    from src.envs import crossing as MG
    r = np.random.RandomState(0)
    for p in (0.1, 0.3, 0.5, 0.7):
        op = jnp.asarray(r.rand(2000, 7, 7) < p)
        a = jax.jit(jax.vmap(MG._process_vis))(op)
        b = jax.jit(jax.vmap(MG._process_vis_scalar))(op)
        assert not bool(jnp.any(a != b))


def test_mg_teacher_reaches_goal():
    from src.envs import crossing as MG
    st = jax.vmap(MG.reset)(jax.random.split(jax.random.PRNGKey(5), 32))

    def run(s):
        def body(c, _):
            s, done = c
            s2, _, d = MG.step(s, MG.teacher(s))
            s = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), s, s2)
            return (s, done | d), None
        (s, done), _ = jax.lax.scan(body, (s, jnp.zeros((), bool)), None, length=200)
        return s.success, s.t
    succ, t = jax.jit(jax.vmap(run))(st)
    assert bool(jnp.all(succ))
    assert np.array_equal(np.asarray(t), np.asarray(st.dist[:, 1, 1, 0]))


def test_lh_eval_goals_are_the_official_test_goals():
    """Eval task k of 2D-LH uses the official test sampler's goal for seed 2^31-1+k."""
    import json, os
    import jax
    import numpy as np
    from src.data import LH_SPEC
    from src.envs import lighthouse as LH
    d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "lh_official_test_goals.json")))
    assert d["world_corners"] == LH.CORNERS.tolist()
    goals = [int(LH_SPEC.eval_reset(k, jax.random.PRNGKey(0), {}).goal) for k in range(200)]
    assert goals == d["goals"]
    assert np.bincount(goals, minlength=4).tolist() == [50, 51, 57, 42]


def test_pd_eval_doors_are_the_official_test_sampler():
    """Eval task k of PD has the good door 1 + seed % 3 of the official test
    sampler with seed 2^31 - 1 + k (Python ints). The traced int32 expression used before overflowed for k >= 1."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.data import PD_SPEC
    from src.envs import pd as PD
    official = [1 + ((2 ** 31 - 1 + k) % 3) for k in range(1000)]
    doors = jax.jit(jax.vmap(lambda k: PD_SPEC.eval_reset(k, jax.random.PRNGKey(0), {"public": False}).good))(
        jnp.arange(1000))
    assert np.asarray(doors).tolist() == official
    assert official[:6] == [2, 3, 1, 2, 3, 1]
    assert [PD.eval_task(k) for k in range(1000)] == official


def test_pd_public_teacher_n_wrong():
    """The public teacher picks d_locked and gets the last n_wrong digits wrong."""
    import jax.numpy as jnp
    from src.envs import pd as PD
    for n_wrong in (1, 2, 3):
        s = PD.reset(None, good=2)
        acts = []
        for _ in range(12):
            a = int(PD.teacher_public(s, n_wrong))
            acts.append(a)
            s, r, d = PD.step(s, a)
            if bool(d):
                break
        assert acts[0] == PD.LOCKED
        digits = acts[1:]
        k = PD.CODE_LEN - n_wrong                       # first wrong digit index; the episode ends there with 0
        assert digits[:k] == list(PD.CODE[:k]) and digits[k] == (PD.CODE[k] + 1) % 3
        assert float(s.ret) == 0.0 and len(digits) == k + 1


def test_pd_cap_is_a_time_limit_not_a_terminal_phase():
    """No-op digits at the door state reach the 100-step cap with done = True but phase != 3: the cap is a time limit,
    not a terminal state (a value target with terminal = phase == 3 bootstraps there)."""
    import jax
    from src.envs import pd as PD
    s = PD.reset(jax.random.PRNGKey(0))
    for t in range(PD.MAX_STEPS):
        s, r, d = PD.step(s, 0)   # a digit at the door state is a no-op (actions 0-2 are digits)
        assert float(r) == 0.0
        assert bool(d) == (t == PD.MAX_STEPS - 1)
    assert int(s.phase) == 0
