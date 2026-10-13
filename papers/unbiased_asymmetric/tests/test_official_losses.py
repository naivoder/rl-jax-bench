"""Standing rule 2: our losses, gradients and one full update (critic step, then actor step with the updated critic;
Adam eps=1e-4) reproduce the official asym-rlpo@1a065b0 ones on identical weights and episodes, for all 5 methods on
Heaven-Hell-3 and Shopping-5. Oracle data: scripts/official/export_losses.py (official venv)."""
import os

import pytest

from _helpers import ORACLE, port, unport, oracle_mark

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from algorithm import Episode, init_run_state, make_losses, make_update  # noqa: E402
from envs import make_env  # noqa: E402
from model import METHODS, NetSpec  # noqa: E402

import subprocess
import sys

F64 = os.environ.get("JAX_ENABLE_X64") == "1"
PATH = os.path.join(ORACLE, "losses_f64.npz" if F64 else "losses.npz")
pytestmark = oracle_mark(PATH, "scripts/official/export_losses.py")
CASES = [(e, m) for e in ("heavenhell_3", "shopping_5", "car_flag") for m in METHODS]
LR_PI, LR_V, EPS, GAMMA, LAM = 1e-3, 3e-4, 1e-4, 0.99, 0.37   # as in scripts/official/export_losses.py
C = 100


def _load(env, m):
    d = np.load(PATH)
    ft = np.float64 if F64 else np.float32
    pfx = f"{env}/{m}/"
    sub = lambda k: {n[len(pfx + k):]: d[n] for n in d.files if n.startswith(pfx + k)}  # noqa: E731
    eps = []
    for i in range(2):
        st, ob, ac, rw = (d[pfx + f"ep{i}/{k}"] for k in ("states", "obs", "actions", "rewards"))
        L = len(ac)
        pad = lambda x, n, v=0: np.concatenate([x, np.full((n - len(x),) + x.shape[1:], v, x.dtype)])  # noqa: E731
        if st.dtype.kind == "f":
            st, ob = st.astype(ft), ob.astype(ft)
        eps.append(Episode(states=pad(st, C + 1), obs=pad(ob, C + 1), actions=pad(ac, C + 1),
                           rewards=pad(rw.astype(ft), C), mask=np.arange(C) < L, trunc=np.bool_(L == C),
                           length=np.int32(L)))
    eps = jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *eps)
    return d, pfx, sub, eps


def _close(a, b, tol, what):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-6)
    assert err < tol, f"{what}: rel err {err:.2e}"


# float32 (production dtype): losses/grads to <=1e-4 rel; the post-update parameters to <= 1 % of the learning
# rate in absolute terms (Adam's first step is g/(|g|+eps), which amplifies float32 rounding where |g| ~ eps=1e-4).
# float64 replica (subprocess, JAX_ENABLE_X64=1, oracle run with torch float64): everything to <= 1e-9 rel.
TOL = dict(loss=1e-9, grad=1e-9, post=1e-9) if F64 else dict(loss=1e-4, grad=2e-4, post=None)


@pytest.mark.parametrize("env,m", CASES)
def test_losses_grads_and_update(env, m):
    check(env, m)


def test_float64_replica_matches_tightly():
    if not os.path.exists(os.path.join(ORACLE, "losses_f64.npz")):
        pytest.skip("needs $OFFICIAL_ROOT/out/losses_f64.npz: run sh scripts/official/setup.sh, then "
                    "$OFFICIAL_ROOT/.venv/bin/python scripts/official/export_losses.py --f64")
    r = subprocess.run([sys.executable, __file__], env={**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"},
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "ALL F64 OK" in r.stdout


def check(env, m):
    d, pfx, sub, eps = _load(env, m)
    mode = METHODS[m][0]
    envf = make_env(env)
    spec = NetSpec(envf.kind, envf.n_actions, envf.n_obs, envf.n_states)
    pa, pc = port(sub("pre/"), mode)
    _, pt = port(sub("tgt/"), mode)
    pa, pc, pt = (jax.tree.map(jnp.asarray, x) for x in (pa, pc, pt))
    _, closs, aloss = make_losses(spec, m, GAMMA)
    lc, gc = jax.value_and_grad(closs)(pc, pt, eps)
    (la, (lpol, lneg, _)), ga = jax.value_and_grad(aloss, has_aux=True)(pa, pc, eps, LAM)
    if F64:
        pa, pc, pt, eps = jax.tree.map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                                       (pa, pc, pt, eps))
        lc, gc = jax.value_and_grad(closs)(pc, pt, eps)
        (la, (lpol, lneg, _)), ga = jax.value_and_grad(aloss, has_aux=True)(pa, pc, eps, LAM)
    _close(lc, d[pfx + "critic_loss"], TOL["loss"], "critic loss")
    _close(lpol, d[pfx + "actor_loss"], TOL["loss"], "actor loss")
    _close(lneg, d[pfx + "negentropy_loss"], TOL["loss"], "negentropy loss")
    for kind, g, tag in (("critic", gc, "gc/"), ("actor", ga, "ga/")):
        ours, off = unport(g, mode, kind), sub(tag)
        for k, v in ours.items():
            ref = off[k][:-1] if "observation_model" in k else off[k]
            _close(v, ref, TOL["grad"], f"grad {k}")
    # one full update through the production code path (learn): no target refresh, constant lambda
    cfg = dict(gamma=GAMMA, cap=C, episodes_per_update=2, max_env_steps=10 ** 9, target_period=10 ** 9, n_log=200,
               alpha_pi=LR_PI, alpha_v=LR_V, adam_eps=EPS, lambda0=LAM, lambda_final_factor=1.0,
               negentropy_nsteps=2_000_000)
    upd = make_update(envf, spec, m, cfg)
    rs = init_run_state(pa, pc, jax.random.PRNGKey(0), cfg)
    if F64:
        rs = jax.tree.map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, rs)
    rs = rs._replace(pt=pt, next_target=jnp.int32(10 ** 9), t_env=jnp.int32(5))
    new = upd.learn(rs, eps)
    post_a, post_c = port(sub("post/"), mode)
    for ours, ref in ((new.pa, post_a), (new.pc, post_c)):
        for (kp, x), (_, y) in zip(jax.tree_util.tree_leaves_with_path(ours), jax.tree_util.tree_leaves_with_path(ref)):
            if F64:
                _close(x, y, TOL["post"], f"post-update param {jax.tree_util.keystr(kp)}")
            else:
                lr = LR_PI if ours is new.pa else LR_V
                err = float(np.max(np.abs(np.asarray(x, np.float64) - y)))
                assert err <= 0.01 * lr, f"post-update {jax.tree_util.keystr(kp)}: |diff| {err:.2e} > 1% of lr"


if __name__ == "__main__":
    for e, mm in CASES:
        if F64 and e == "car_flag":
            continue   # see scripts/official/export_losses.py: no float64 replica of the official one-hot path
        check(e, mm)
        print("ok", e, mm, flush=True)
    print("ALL F64 OK" if F64 else "ALL F32 OK")
