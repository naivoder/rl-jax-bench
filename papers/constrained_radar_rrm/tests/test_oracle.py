"""Oracle differential tests (EKF vs filterpy, DDPG update vs CleanRL).

* filterpy 1.4.5 ExtendedKalmanFilter fixture written by scripts/official/filterpy_ekf_harness.py
  (own venv): our predict + ekf_update must reproduce x and P on 3 x 200-step sequences.
* CleanRL ddpg_continuous_action_jax.py @ fe8d8a0: QNetwork, Actor, update_critic and update_actor are extracted
  VERBATIM from the source file with `ast` and exec'd (nothing is copied into src/); our generic ddpg_update, run
  with CleanRL's network apply functions, identical init, identical batches and a_clip = (-1, 1), must reproduce
  CleanRL's parameters after 30 updates.
Both tests skip until `sh scripts/official/setup.sh` has fetched CleanRL and written the filterpy fixture into
$OFFICIAL_ROOT (default external/official).
"""
import ast
import os
import types

import numpy as np
import pytest

from src import common  # noqa: F401
import jax
import jax.numpy as jnp
import optax

from src import physics as ph
from src import algorithm as alg

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))
FIX = os.path.join(OFFICIAL_ROOT, "fixtures", "filterpy_ekf.npz")
CLEANRL = os.path.join(OFFICIAL_ROOT, "cleanrl", "cleanrl", "ddpg_continuous_action_jax.py")
CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"


@pytest.mark.skipif(not os.path.exists(FIX), reason="filterpy fixture absent: run sh scripts/official/setup.sh")
def test_ekf_vs_filterpy():
    f = np.load(FIX)
    worst = 0.0
    for s in range(3):
        x, P = jnp.asarray(f[f"s{s}_x0"]), jnp.asarray(f[f"s{s}_P0"])
        for k in range(f[f"s{s}_z"].shape[0]):
            xp, Pp = ph.predict(x, P, 2.5, 16.0)
            x, P = ph.ekf_update(xp, Pp, jnp.asarray(f[f"s{s}_z"][k]), f[f"s{s}_var_r"][k], f[f"s{s}_var_th"][k])
            rx = np.max(np.abs(np.asarray(x) - f[f"s{s}_x"][k]) / (np.abs(f[f"s{s}_x"][k]) + 1.0))
            rP = np.max(np.abs(np.asarray(P) - f[f"s{s}_P"][k])) / np.max(np.abs(f[f"s{s}_P"][k]))
            worst = max(worst, rx, rP)
    print("EKF-ORACLE worst relative diff", worst)
    assert worst <= 1e-8


def _extract_cleanrl():
    src = open(CLEANRL).read()
    tree = ast.parse(src)
    seg = {}
    main_if = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("QNetwork", "Actor", "TrainState"):
            seg[node.name] = ast.get_source_segment(src, node)
        if isinstance(node, ast.If) and "__main__" in ast.get_source_segment(src, node.test):
            main_if = node
    for node in ast.walk(main_if):
        if isinstance(node, ast.FunctionDef) and node.name in ("update_critic", "update_actor"):
            lines = src.splitlines()[node.lineno - 2:node.end_lineno]      # include @jax.jit
            ind = len(lines[0]) - len(lines[0].lstrip())
            seg[node.name] = "\n".join(l[ind:] for l in lines)
    return seg


@pytest.mark.skipif(not os.path.exists(CLEANRL), reason="CleanRL clone absent: run sh scripts/official/setup.sh")
def test_ddpg_update_vs_cleanrl():
    import flax
    import flax.linen as nn
    from flax.training.train_state import TrainState as _TS
    seg = _extract_cleanrl()
    ns = dict(flax=flax, nn=nn, jax=jax, jnp=jnp, np=np, optax=optax, TrainState=_TS)
    exec(seg["QNetwork"], ns)
    exec(seg["Actor"], ns)
    exec(seg["TrainState"], ns)
    obs_dim, act_dim, lr, gamma, tau, B = 11, 5, 3e-4, 0.9, 0.01, 64
    actor = ns["Actor"](action_dim=act_dim, action_scale=jnp.ones(act_dim, jnp.float32),
                        action_bias=jnp.zeros(act_dim, jnp.float32))
    qf = ns["QNetwork"]()
    ns.update(actor=actor, qf=qf, args=types.SimpleNamespace(gamma=gamma, tau=tau))
    exec(seg["update_critic"], ns)
    exec(seg["update_actor"], ns)
    k = jax.random.PRNGKey(3)
    ka, kq, kd = jax.random.split(k, 3)
    o0 = jnp.zeros((1, obs_dim), jnp.float32)
    a0 = jnp.zeros((1, act_dim), jnp.float32)
    TS = ns["TrainState"]
    a_state = TS.create(apply_fn=actor.apply, params=actor.init(ka, o0), target_params=actor.init(ka, o0),
                        tx=optax.adam(lr))
    q_state = TS.create(apply_fn=qf.apply, params=qf.init(kq, o0, a0), target_params=qf.init(kq, o0, a0),
                        tx=optax.adam(lr))
    ours = alg.DDPGState(a_state.params, q_state.params, a_state.target_params, q_state.target_params,
                         optax.adam(lr).init(a_state.params), optax.adam(lr).init(q_state.params), jnp.zeros((), jnp.int32))
    worst = 0.0
    for i in range(30):
        kb = jax.random.fold_in(kd, i)
        k1, k2, k3, k4 = jax.random.split(kb, 4)
        s = jax.random.normal(k1, (B, obs_dim), jnp.float32)
        a = jax.random.uniform(k2, (B, act_dim), jnp.float32, -1, 1)
        r = jax.random.normal(k3, (B,), jnp.float32)
        s2 = jax.random.normal(k4, (B, obs_dim), jnp.float32)
        term = jnp.zeros(B, jnp.float32)
        q_state, _, _ = ns["update_critic"](a_state, q_state, s, a, s2, r, term)
        a_state, q_state, _ = ns["update_actor"](a_state, q_state, s)
        batch = dict(s=s, a=a, r=r, s2=s2, disc=1.0 - term)
        ours, _ = alg.ddpg_update(ours, batch, actor.apply, qf.apply, optax.adam(lr), optax.adam(lr), gamma, tau,
                                  a_clip=(-1.0, 1.0))
        for mine, theirs in [(ours.actor, a_state.params), (ours.critic, q_state.params),
                             (ours.actor_t, a_state.target_params), (ours.critic_t, q_state.target_params)]:
            d = jax.tree.map(lambda x, y: float(jnp.max(jnp.abs(x - y)) / (jnp.max(jnp.abs(y)) + 1e-12)), mine, theirs)
            worst = max(worst, max(jax.tree.leaves(d)))
    print("DDPG-ORACLE worst relative param diff after 30 updates", worst, "cleanrl", CLEANRL_COMMIT)
    assert worst <= 1e-5
