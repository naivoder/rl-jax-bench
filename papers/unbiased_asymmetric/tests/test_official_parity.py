"""Standing rule 2 -- OFFICIAL-CODE PARITY on SMALL COMMITTED FIXTURES (tests/fixtures/official/, recorded from the authors'
asym-rlpo@1a065b0 + gym-pomdps@b52e520 + rl-parsers@b2197e3, torch 1.13.1 CPU; see scripts/official/export_*.py and
scripts/official/make_parity_fixtures.py). Unlike tests/test_official_{env,carflag,cleaner,losses}.py (full-size, git-ignored
$OFFICIAL_ROOT/out/ exports) these run on a fresh clone, with no oracle venv.

  * categorical envs (HH-3/4, Shopping-5/6): parsed T/O/R/D/start SHA256 == rl_parsers'; 200 action sequences/env
    replayed step by step -> identical obs / reward / done / state streams (exact).
  * Car-Flag: 200 sequences, float64 replay, 1e-9.   * Cleaner: 20 x 100 joint actions, bit-identical.
  * bookkeeping (avg-100 window, log/target Dispensers, linear negentropy): exact / 1e-5.
  * losses, gradients and ONE full update (critic Adam step, then actor Adam step with the UPDATED online critic,
    eps=1e-4) for 5 methods x {HH-3, Shopping-5, Car-Flag, Cleaner}: weights regenerated from names (scripts/official/parity_common.py),
    episodes sampled by the official behavior policy. Tolerances -- float32: losses 1e-4 rel; gradient norm 2e-4 rel,
    random-projection error <= 2e-4 * |g| * 5/sqrt(n) (i.e. an RMS elementwise error of 2e-4 |g|/sqrt(n) at 5 sigma),
    full small tensors 2e-4 rel-to-max; update (in learning-rate units) elementwise <= 0.01 lr, projections <= 0.05,
    norm 1e-3 rel. float64 replica (subprocess, JAX x64 vs torch float64): every quantity <= 1e-9 rel.
"""
import json
import os
import subprocess
import sys

import pytest

from _helpers import REPO, port, unport

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "scripts", "official"))
import parity_common as pcm  # noqa: E402
from algorithm import Episode, entropy_weight, init_run_state, make_losses, make_update  # noqa: E402
from data import load_pomdp  # noqa: E402
from envs import carflag_step, cat_step, make_cat_env, make_env  # noqa: E402
from model import METHODS, NetSpec, init_params  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "official")
F64 = os.environ.get("JAX_ENABLE_X64") == "1"
CAT_ENVS = ["heavenhell_3", "heavenhell_4", "shopping_5", "shopping_6"]
LOSS_CASES = [(e, m) for e in ("heavenhell_3", "shopping_5", "car_flag", "cleaner") for m in METHODS]
C = 100


def _f(name):
    return np.load(os.path.join(FIX, name))


# ----------------------------------------------------------------------------------------------- environments
@pytest.mark.parametrize("name", CAT_ENVS)
def test_parsed_arrays_sha256(name):
    import hashlib
    d, p = _f("env_traces_small.npz"), load_pomdp(name)
    for k, ours in (("T", p.T), ("O", p.O), ("R", p.R), ("D", p.D), ("start", p.start)):
        x = np.ascontiguousarray(np.asarray(ours, np.float64))
        assert tuple(x.shape) == tuple(d[f"{name}/{k}_shape"]), k
        assert hashlib.sha256(x.tobytes()).hexdigest() == str(d[f"{name}/{k}_sha256"]), k


@pytest.mark.parametrize("name", CAT_ENVS)
def test_categorical_env_streams(name):
    d = _f("env_traces_small.npz")
    g = lambda k: d[f"{name}/{k}"].astype(np.int64) if d[f"{name}/{k}"].dtype.kind == "i" else d[f"{name}/{k}"]  # noqa
    p = load_pomdp(name)
    env = make_cat_env(p)
    act, st, ob, rw, dn, ln = g("act"), g("st"), g("ob"), g("rw"), g("dn"), g("len")
    assert np.all(ob[:, 0] == p.n_obs), "first observation must be the extra reset observation"

    def roll(s0, acts, key):
        def body(s, x):
            a, k = x
            s2, o, r, done = cat_step(env, k, s, jnp.maximum(a, 0))
            return s2, (s2, o, r, done)
        return jax.lax.scan(body, s0, (acts, jax.random.split(key, acts.shape[0])))[1]

    S2, O, R, D = map(np.asarray, jax.jit(jax.vmap(roll))(
        jnp.asarray(g("s0"), jnp.int32), jnp.asarray(act, jnp.int32), jax.random.split(jax.random.PRNGKey(0), len(act))))
    for i in range(len(act)):
        L = ln[i]
        assert np.all(O[i, :L] == ob[i, 1:L + 1]), f"obs seq {i}"
        assert np.allclose(R[i, :L], rw[i, :L]), f"reward seq {i}"
        env_done = dn[i, :L].copy()
        if L == C and st[i, L] != -1:     # official TimeLimit sets done on truncation; the env itself did not
            env_done[-1] = False
        assert np.all(D[i, :L] == env_done), f"done seq {i}"
        nonterm = st[i, 1:L + 1] >= 0      # official state is -1 after a terminal step
        assert np.all(S2[i, :L][nonterm] == st[i, 1:L + 1][nonterm]), f"state seq {i}"


def test_carflag_streams():
    d = _f("carflag_traces_small.npz")
    with jax.enable_x64(True):
        def roll(s0, acts):
            def body(s, a):
                s2, o, r, done = carflag_step(None, s, jnp.maximum(a, 0))
                return s2, (s2, o, r, done)
            return jax.lax.scan(body, s0, acts)[1]
        S, O, R, D = map(np.asarray, jax.jit(jax.vmap(roll))(jnp.asarray(d["s0"]), jnp.asarray(d["act"])))
    for i in range(len(d["len"])):
        L = d["len"][i] if d["len"][i] > 0 else C
        assert np.allclose(O[i, :L], d["ob"][i, :L], atol=1e-9), f"obs seq {i}"
        assert np.allclose(S[i, :L], d["st"][i, :L], atol=1e-9), f"state seq {i}"
        assert np.array_equal(R[i, :L], d["rw"][i, :L]), f"reward seq {i}"
        env_done = d["dn"][i, :L].copy()
        if d["len"][i] == C and d["rw"][i, L - 1] == 0:
            env_done[-1] = False
        assert np.array_equal(D[i, :L], env_done), f"done seq {i}"


def test_cleaner_streams():
    d = _f("cleaner_traces_small.npz")
    st = np.unpackbits(d["st_bits"], axis=-1)[..., :int(d["st_dim"])].astype(np.float32)
    ob = np.unpackbits(d["ob_bits"], axis=-1)[..., :int(d["ob_dim"])].astype(np.float32)
    env = make_env("cleaner")
    s0, o0 = env.reset(jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(s0), d["s0"][0]) and np.array_equal(np.asarray(o0), d["o0"][0])

    def roll(acts):
        def body(s, a):
            s2, o, r, _ = env.step(None, s, a)
            return s2, (s2, o, r)
        return jax.lax.scan(body, s0, acts)[1]
    S, O, R = map(np.asarray, jax.jit(jax.vmap(roll))(jnp.asarray(d["act"], jnp.int32)))
    assert np.array_equal(S, st) and np.array_equal(O, ob) and np.array_equal(R, d["rw"])



def test_cleaner_full_clean_does_not_terminate():
    """paper App. C.2.2 says the task is complete once every cell has been visited, but the official
    env (done = i_step >= 200, wrapped in TimeLimit(100)) never terminates on a clean floor. A greedy covering policy
    recorded in the official env (scripts/official/export_cleaner_fullclean.py) cleans all 71 dirty cells by step 58; the official
    done flag stays False until the TimeLimit at step 100. Ours: identical rewards, done False at every step (the cap
    is applied by the training loop, as for every env)."""
    d = _f("cleaner_fullclean.npz")
    assert int(d["rw"].sum()) == 71 and int(d["left"][-1]) == 0
    k = int(np.argmax(d["left"] == 0))
    assert k + 1 < 100 and not d["dn"][:99].any() and d["dn"][99]          # clean well before the cap; done only at cap
    env = make_env("cleaner")
    s0, _ = env.reset(jax.random.PRNGKey(0))

    def body(s, a):
        s2, o, r, done = env.step(None, s, a)
        return s2, (r, done)
    R, D = map(np.asarray, jax.jit(lambda acts: jax.lax.scan(body, s0, acts)[1])(jnp.asarray(d["act"], jnp.int32)))
    assert np.array_equal(R, d["rw"]) and not D.any()

def test_bookkeeping_schedules():
    d = _f("schedules.npz")
    env = make_env("heavenhell_3")
    spec = NetSpec(env.kind, env.n_actions, env.n_obs, env.n_states, emb=4, hidden=4, head=(4,))
    cfg = dict(gamma=0.99, cap=C, episodes_per_update=2, max_env_steps=int(d["T_MAX"]), target_period=10_000,
               n_log=200, alpha_pi=1e-3, alpha_v=1e-3, adam_eps=1e-4, lambda0=float(d["LAM0"]),
               lambda_final_factor=0.1, negentropy_nsteps=20_000)
    learn = jax.jit(make_update(env, spec, "a2c", cfg).learn)
    pa, pc = init_params(spec, "a2c", jax.random.PRNGKey(0))
    rs = init_run_state(pa, pc, jax.random.PRNGKey(1), cfg)
    nt, nl, lam = [], [], []
    for u in range(len(d["lam"])):
        L = d["lens"][u]
        rw = np.zeros((2, C), np.float32)
        rw[:, 0] = d["rets"][u]
        eps = Episode(states=jnp.zeros((2, C + 1), jnp.int32), obs=jnp.full((2, C + 1), env.n_obs, jnp.int32),
                      actions=jnp.zeros((2, C + 1), jnp.int32), rewards=jnp.asarray(rw),
                      mask=jnp.asarray(np.arange(C)[None] < L[:, None]), trunc=jnp.asarray(L == C),
                      length=jnp.asarray(L, jnp.int32))
        rs = learn(rs, eps)
        nt.append(int(rs.next_target))
        nl.append(int(rs.next_log))
        lam.append(float(entropy_weight(rs.t_env, cfg["lambda0"], 0.1, 20_000)))
    k = int(rs.log_idx)
    assert k == len(d["log_t"]) and np.array_equal(np.asarray(rs.logs["t"][:k]), d["log_t"])
    assert np.allclose(np.asarray(rs.logs["avg100"][:k]), d["log_avg"], atol=1e-5)
    assert np.array_equal(nt, d["next_tgt"]) and np.array_equal(nl, d["next_log"])
    assert np.allclose(lam, d["lam"], rtol=1e-6)


# ---------------------------------------------------------------------------------------- losses and update
TOL = (dict(loss=1e-9, grad=1e-9, proj=1e-9, upd_el=None, upd_proj=None, upd_norm=1e-9) if F64 else
       dict(loss=1e-4, grad=2e-4, proj=2e-4, upd_el=0.01, upd_proj=0.05, upd_norm=1e-3))


def _load_case(d, meta, env, m):
    ft = np.float64 if F64 else np.float32
    pfx = f"{env}/{m}/"
    shapes = {k: tuple(s) for k, s in meta["cases"][f"{env}/{m}"]["shapes"].items()}
    w = lambda tag: {k: v.astype(ft) for k, v in pcm.gen_weights(tag, shapes).items()}  # noqa: E731
    eps = []
    for i in range(2):
        st, ob, ac, rw = (d[pfx + f"ep{i}/{k}"] for k in ("states", "obs", "actions", "rewards"))
        L = len(ac)
        pad = lambda x, n: np.concatenate([x, np.zeros((n - len(x),) + x.shape[1:], x.dtype)])  # noqa: E731
        if st.dtype.kind == "f":
            st, ob = st.astype(ft), ob.astype(ft)
        eps.append(Episode(states=pad(st, C + 1), obs=pad(ob, C + 1), actions=pad(ac, C + 1),
                           rewards=pad(rw.astype(ft), C), mask=np.arange(C) < L, trunc=np.bool_(L == C),
                           length=np.int32(L)))
    return pfx, w("pre"), w("tgt"), jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *eps)


def _check_fp(d, key, ours, tol_rel, tol_proj, what, tol_el=None):
    ours = np.asarray(ours, np.float64).ravel()
    n, nrm = int(d[key + "/n"]), float(d[key + "/norm"])
    assert ours.size == n, f"{what}: size {ours.size} vs {n}"
    assert abs(np.linalg.norm(ours) - nrm) <= tol_rel * max(nrm, 1e-12) + 1e-12, \
        f"{what}: norm {np.linalg.norm(ours):.6e} vs {nrm:.6e}"
    perr = np.max(np.abs(pcm.directions(key.split("/", 3)[3], n) @ ours - d[key + "/proj"]))
    bound = tol_proj if tol_el is not None else tol_proj * max(nrm, 1e-12) * (5.0 / np.sqrt(n) if not F64 else 1.0)
    assert perr <= bound + 1e-12, f"{what}: projection err {perr:.2e} > {bound:.2e}"
    if key + "/full" in d.files:
        full = d[key + "/full"]
        if tol_el is not None:
            assert np.max(np.abs(ours - full)) <= tol_el, f"{what}: elementwise {np.max(np.abs(ours - full)):.2e}"
        else:
            err = np.max(np.abs(ours - full)) / max(np.max(np.abs(full)), 1e-12)
            assert err <= tol_rel, f"{what}: full rel err {err:.2e}"


def check_case(env, m):
    d = _f("losses_compact_f64.npz" if F64 else "losses_compact.npz")
    meta = json.loads(str(d["meta"]))
    gamma, lam, lr_pi, lr_v, adam_eps = meta["gamma"], meta["lam"], meta["lr_pi"], meta["lr_v"], meta["eps"]
    pfx, wpre, wtgt, eps = _load_case(d, meta, env, m)
    mode = METHODS[m][0]
    envf = make_env(env)
    spec = NetSpec(envf.kind, envf.n_actions, envf.n_obs, envf.n_states)
    pa, pc = (jax.tree.map(jnp.asarray, x) for x in port(wpre, mode))
    pt = jax.tree.map(jnp.asarray, port(wtgt, mode)[1])
    _, closs, aloss = make_losses(spec, m, gamma)
    lc, gc = jax.value_and_grad(closs)(pc, pt, eps)
    (_, (lpol, lneg, _)), (ga, ga_c) = jax.value_and_grad(aloss, argnums=(0, 1), has_aux=True)(pa, pc, eps, lam)
    for ours, k in ((lc, "critic_loss"), (lpol, "actor_loss"), (lneg, "negentropy_loss")):
        ref = float(d[pfx + k])
        assert abs(float(ours) - ref) <= TOL["loss"] * max(abs(ref), 1e-6), f"{k}: {float(ours)} vs {ref}"
    for kind, g, tag in (("critic", gc, "gc"), ("actor", ga, "ga")):
        ours = {pcm.canonical(k): v for k, v in unport(g, mode, kind).items()}
        keys = sorted({k.rsplit("/", 1)[0] for k in d.files if k.startswith(pfx + tag + "/")})
        for key in keys:
            name = key.split("/", 3)[3]
            if name.startswith("critic.") and tag == "ga":     # semi-gradient: actor loss sends nothing to the critic
                assert float(d[key + "/norm"]) == 0.0
                continue
            _check_fp(d, key, ours[pcm.canonical(name)], TOL["grad"], TOL["proj"], f"{tag} {name}")
    assert all(float(jnp.max(jnp.abs(x))) == 0.0 for x in jax.tree.leaves(ga_c)), "actor loss must not reach critic"
    cfg = dict(gamma=gamma, cap=C, episodes_per_update=2, max_env_steps=10 ** 9, target_period=10 ** 9, n_log=200,
               alpha_pi=lr_pi, alpha_v=lr_v, adam_eps=adam_eps, lambda0=lam, lambda_final_factor=1.0,
               negentropy_nsteps=2_000_000)
    rs = init_run_state(pa, pc, jax.random.PRNGKey(0), cfg)
    rs = rs._replace(pt=pt, next_target=jnp.int32(10 ** 9), t_env=jnp.int32(5))
    new = make_update(envf, spec, m, cfg).learn(rs, eps)
    for kind, before, after, lr in (("actor", pa, new.pa, lr_pi), ("critic", pc, new.pc, lr_v)):
        delta = jax.tree.map(lambda a, b: (np.asarray(a, np.float64) - np.asarray(b, np.float64)) / lr, after, before)
        for name, v in unport(delta, mode, kind).items():
            key = pfx + "dpost/" + name
            if F64:
                _check_fp(d, key, v, TOL["upd_norm"], TOL["proj"], f"update {name}")
            else:
                _check_fp(d, key, v, TOL["upd_norm"], TOL["upd_proj"], f"update {name}", tol_el=TOL["upd_el"])


@pytest.mark.parametrize("env,m", LOSS_CASES)
def test_losses_grads_update_compact(env, m):
    check_case(env, m)


def test_float64_replica_compact():
    r = subprocess.run([sys.executable, __file__], env={**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"},
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "ALL F64 OK" in r.stdout


if __name__ == "__main__":
    for e, mm in LOSS_CASES:
        if F64 and e in ("car_flag", "cleaner"):
            continue
        check_case(e, mm)
        print("ok", e, mm, flush=True)
    print("ALL F64 OK" if F64 else "ALL F32 OK")


# ------------------------------------------------------------------------------ behavior-policy rollouts
ROLL_CASES = [("heavenhell_3", "a2c_asym_hs"), ("shopping_5", "a2c"), ("shopping_5", "a2c_react_2"), ("car_flag", "a2c")]


@pytest.mark.parametrize("env,m", ROLL_CASES)
def test_behavior_rollout_distribution(env, m):
    """Official sample_episodes vs our batched rollout, same (name-regenerated) actor weights, N=2000 episodes each:
    mean undiscounted return and mean length agree (Welch |z| < 4), and the action-count and first-action-count
    histograms are homogeneous (chi-square p > 1e-4). A distributional check: the two samplers use different RNGs."""
    from scipy.stats import chi2_contingency
    from algorithm import make_rollout_batched
    d = _f("rollout_stats.npz")
    pfx = f"{env}/{m}/"
    ret_o, len_o = d[pfx + "ret"], d[pfx + "len"]
    N = len(ret_o)
    meta = json.loads(str(_f("losses_compact.npz")["meta"]))
    shapes = {k: tuple(s) for k, s in meta["cases"][f"{env}/{m}"]["shapes"].items()}
    pa, _ = port({k: v.astype(np.float32) for k, v in pcm.gen_weights("pre", shapes).items()}, METHODS[m][0])
    pa = jax.tree.map(lambda x: jnp.asarray(x)[None], pa)
    envf = make_env(env)
    spec = NetSpec(envf.kind, envf.n_actions, envf.n_obs, envf.n_states)
    keys = jax.random.split(jax.random.PRNGKey(2024), N)[None]
    ep = jax.jit(make_rollout_batched(envf, spec, m, C))(pa, keys, jnp.ones((1,), bool))
    M = np.asarray(ep.mask[0])
    ret = (np.asarray(ep.rewards[0], np.float64) * M).sum(1)
    ln = M.sum(1)
    A = np.asarray(ep.actions[0])[:, :C]
    for name, a, b in (("return", ret, ret_o), ("length", ln, len_o)):
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        z = (a.mean() - b.mean()) / max(se, 1e-12)
        print(f"[METRIC] rollout {env}/{m} {name} ours {a.mean():.4f} official {b.mean():.4f} z={z:.2f}")
        assert abs(z) < 4, f"{name}: ours {a.mean():.4f} vs official {b.mean():.4f} (z={z:.2f})"
    counts = np.bincount(A[M], minlength=envf.n_actions)
    first = np.bincount(A[:, 0], minlength=envf.n_actions)
    for name, a, b in (("actions", counts, d[pfx + "act_counts"]), ("first actions", first, d[pfx + "first_act_counts"])):
        tab = np.stack([a, b])
        tab = tab[:, tab.sum(0) > 0]
        p = chi2_contingency(tab)[1]
        print(f"[METRIC] rollout {env}/{m} {name} chi2 p={p:.3f}")
        assert p > 1e-4, f"{name} histogram differs: p={p:.2e} ours {a} official {b}"
