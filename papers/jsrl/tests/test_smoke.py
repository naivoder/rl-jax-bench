"""Smoke test (< 30 s, CPU): shapes, one training step, one interaction chunk with the guide, one evaluation point, one curriculum step.
Uses the combination-lock env (src/lock.py) so no game files or guide weights are needed."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np, yaml
from src import algorithm as AL, replay as R
from src.curriculum import cur_init, cur_step, stage_list
from src.lock import LockEnv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(REPO, "src/configs/default.yaml")))


def _setup(steps=200, chunk=100):
    env = LockEnv(6, 3)
    L = R.make_layout(0, 512, 25, CFG["dqfd"]["n_step"], CFG["env"]["gamma"], CFG["per"]["alpha"], 256)
    fns = AL.Fns(env, L, CFG, steps, env.guide_fn, chunk)
    names = ["JSRL", "JSRL-Random", "PDD", "DQfD", "BCinit-FT", "IMIT"]
    hp = AL.stack_hp(names, 2, CFG, 6.0)
    return env, L, fns, names, hp


def test_arm_hp_shapes_and_modes():
    env, L, fns, names, hp = _setup()
    assert hp["h_list"].shape == (12, 11) and hp["lr"].shape == (12,)
    assert list(np.asarray(hp["jsrl_mode"])) == [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.asarray(hp["h_list"][0]).tolist() == stage_list(6.0, 10)
    assert np.asarray(hp["h_list"][4]).sum() == 0                      # PDD: no guide steps
    # arms that differ only by the mechanism under test: JSRL and PDD agree on every learner hyper-parameter
    for k in hp:
        if k not in ("jsrl_mode", "h_list"):
            assert float(hp[k][0]) == float(hp[k][4]), k


def test_chunk_with_guide_update_evaluation_and_curriculum():
    """ONE compiled chunk_and_eval (100 interaction steps with the guide, >= 1 gradient update, one 64-episode evaluation point, one beta-rule step)."""
    env, L, fns, names, hp = _setup()
    hp0 = {k: v[0] for k, v in hp.items()}
    hp0 = dict(hp0, learn_start=jnp.float32(50))                       # learning starts inside the chunk
    key = jax.random.PRNGKey(0)
    params = fns.init_params(key)
    buf = R.init_buffer(L, None, hp0)
    c = fns.init_interact(params, params, fns.tx.init(params), buf, key, jax.random.fold_in(key, 1005), hp0)
    c = jax.jit(fns.chunk_and_eval)(c, hp0, jax.random.fold_in(key, 3), jax.random.fold_in(key, 4), jnp.array(True))
    assert int(c["t"]) == 100 and int(c["buf"]["ptr"]) == 100 and int(c["cnt"]) > 0   # >= 1 gradient update happened
    assert int(c["guide_steps"]) > 0 and np.isfinite(float(c["dg"]["loss"]))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(c["params"]))
    assert int(c["neval"]) == 1 and c["ev_alone"].shape == (3,) and int(c["cur"]["n"]) == 1
    assert all(np.isfinite(float(c[k][1])) for k in ("ev_alone", "ev_thr", "ev_greedy")) and int(c["ev_gsteps"][1]) == int(c["guide_steps"])


def test_curriculum_step_runs_under_jit_vmap():
    cs = jax.vmap(lambda _: cur_init(3))(jnp.arange(4))
    f = jax.jit(jax.vmap(lambda c, r: cur_step(c, r, 0.05, 10, 1.0, True)))
    for r in range(6):
        cs, adv = f(cs, jnp.full((4,), 5.0))
    assert np.asarray(cs["stage"]).tolist() == [1, 1, 1, 1] and cs["win"].shape == (4, 3)
