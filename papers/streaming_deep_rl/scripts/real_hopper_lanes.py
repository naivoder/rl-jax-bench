"""Several (arm, seed) lanes of OUR ACAgent.observe (unchanged src/) closed-loop on REAL Hopper-v4 stepped in the
authors' venv (gymnasium 0.29.1 + mujoco 2.3.7), version-matched exactly as scripts/ours_on_mujoco_vm.py (same
keys: fold_in(PRNGKey(0), seed)). Recheck: the paper-F.4 no-ObGD arm (Adam 1e-7)
and the start of F.4's halving protocol (3e-4). Records the first step at which a lane's parameters become
non-finite (diverged_by_step) and the policy drift from its init.
Usage: python real_hopper_lanes.py <out_dir> <total_steps> <cap_s> 'armA:1,2,3|armB:1,2'
"""
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import subprocess  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
from rl_jax_bench.jaxlog import info, metric, write_json  # noqa: E402
from src import agents as ag  # noqa: E402
from src import algorithm as alg  # noqa: E402
from src import eval as ev  # noqa: E402
from src import model as mdl  # noqa: E402
from src import train as tr  # noqa: E402
from src.data import Env  # noqa: E402

OUTD = os.path.abspath(sys.argv[1])
T, CAP = int(float(sys.argv[2])), float(sys.argv[3])
LANES = [(arm, int(sd)) for grp in sys.argv[4].split("|") for arm, sds in [grp.rsplit(":", 1)]
         for sd in sds.split(",")]   # "armA:1,2,3|armB:1,2"
L = len(LANES)
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
OFFICIAL_PY = os.path.join(OFFICIAL_ROOT, ".venv/bin/python")
srv = subprocess.Popen([OFFICIAL_PY, os.path.join(REPO, "scripts", "official", "hopper_env_server.py")]
                       + [str(s) for _, s in LANES], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
ENV_DESC = "gymnasium 0.29.1 Hopper-v4, mujoco 2.3.7 (authors' venv, env server)"
agent = ag.ACAgent(Env(None, None, (11,), 3, False, "gym0.29.1-Hopper-v4-mujoco2.3.7"), overshoot_diag=True)


def _read(n):
    b = srv.stdout.read(n * 8)
    assert len(b) == n * 8, "env server died"
    return np.frombuffer(b, np.float64)


lane = tr.stack_lanes("ac", [tr.lane_cfg("ac", a) for a, _ in LANES])
root = jax.random.PRNGKey(0)
run_keys = [jax.random.fold_in(root, s) for _, s in LANES]
step_keys = jnp.stack([jax.random.fold_in(k, 2) for k in run_keys])


def init_one(rk, ln, obs0):
    ik = jax.random.fold_in(rk, 0)
    p_pi = mdl.init_select(jax.random.fold_in(ik, 0), agent.t_pi, ln.init)
    p_v = mdl.init_select(jax.random.fold_in(ik, 1), agent.t_v, ln.init)
    t0 = jnp.float32(-0.5)
    w, on = agent._obs_in(ln, alg.welford_init((11,)), obs0, t0)
    return ag.ACCarry(p_pi, p_v, alg.tree_zeros(p_pi), alg.tree_zeros(p_v), alg.adam_init(p_pi), alg.adam_init(p_v),
                      w, alg.welford_init(), jnp.float32(0.0), jnp.int32(0), on, t0, jnp.float32(0.0), jnp.int32(0),
                      ag.acdiag_zero())


obs0 = _read(L * 11).reshape(L, 11).astype(np.float32)
c = jax.jit(jax.vmap(init_one))(jnp.stack(run_keys), lane, jnp.asarray(obs0))
pi0 = jax.tree.map(lambda x: np.asarray(x).copy(), c.pi)


@jax.jit
def act(c, t, sk):
    def one(cc, k, ln):
        kk = jax.random.split(jax.random.fold_in(k, t), 5)
        mu, std = agent.pi(cc.pi, cc.obs, ln.use_ln)
        return mu + std * jax.random.normal(kk[0], mu.shape)
    return jax.vmap(one)(c, sk, lane)


@jax.jit
def observe(c, a, o2, r, term, rr):
    def one(cc, ln, aa, oo, rw, tm, ro):
        return agent.observe(cc, ln, aa, (jnp.int32(0), oo, rw, tm), (jnp.int32(0), ro))
    return jax.vmap(one)(c, lane, a, o2, r, term, rr)


diverged_by = [None] * len(LANES)


def finite_check(t):
    ok = np.ones(L, bool)
    for leaf in jax.tree.leaves((c.pi, c.v)):
        ok &= np.isfinite(np.asarray(leaf).reshape(L, -1)).all(1)
    for i in range(L):
        if not ok[i] and diverged_by[i] is None:
            diverged_by[i] = t


def drift():
    """Mean |pi - pi0| and max |pi - pi0| per lane (how far the policy has moved from its init)."""
    leaves = jax.tree.leaves(jax.tree.map(lambda a, b: np.abs(np.asarray(a) - b).reshape(L, -1), c.pi, pi0))
    allp = np.concatenate(leaves, 1)
    return allp.mean(1), allp.max(1)


ep = [[] for _ in LANES]
ep_len = np.zeros(L, int)
t_start = time.time()


def flush(t_done, final=False):
    md, mx = drift()
    for i, (a_, s) in enumerate(LANES):
        e = np.asarray(ep[i], float).reshape(-1, 3)
        summ = ev.summarize_run(e, (t_done // 50_000) * 50_000 or 50_000, 50_000)
        d = jax.tree.map(lambda x: float(np.asarray(x)[i]), c.diag)
        write_json(os.path.join(OUTD, a_, f"seed_{s}", "metrics.json"),
                   dict(exp="recheck_no_obgd", diverged_by_step=diverged_by[i], env=ENV_DESC, arm=a_, seed=s, steps_done=t_done, final=final,
                        summary=summ, episodes_cols=["end_step", "return", "length"], episodes=e.tolist(),
                        diag=dict(d._asdict()), pi_drift_mean=float(md[i]), pi_drift_max=float(mx[i]),
                        wall_s=time.time() - t_start))


t = 0
while t < T and time.time() - t_start < CAP:
    a = np.asarray(act(c, jnp.int32(t), step_keys))
    srv.stdin.write(np.asarray(a, np.float64).tobytes())
    srv.stdin.flush()
    o2 = _read(L * 11).reshape(L, 11).astype(np.float32)
    r = _read(L).astype(np.float32)
    term = _read(L) > 0.5
    rr = _read(L * 11).reshape(L, 11).astype(np.float32)
    ep_len += 1
    c, (ret, ln), _ = observe(c, jnp.asarray(a), jnp.asarray(o2), jnp.asarray(r), jnp.asarray(term), jnp.asarray(rr))
    ret = np.asarray(ret)
    for i in range(L):
        if not np.isnan(ret[i]):
            ep[i].append([t + 1, float(ret[i]), int(ep_len[i])])
            ep_len[i] = 0
    t += 1
    if t % 1000 == 0:
        finite_check(t)
    if t % 10_000 == 0:
        flush(t)
        rec = [np.mean([x[1] for x in ep[i] if x[0] > t - 10_000] or [np.nan]) for i in range(L)]
        md, _ = drift()
        metric("progress", steps=t, sps=t / (time.time() - t_start), last10k_mean_return=np.round(rec, 1).tolist(),
               pi_drift_mean=[float(f"{x:.2e}") for x in md])
finite_check(t)
flush(t, final=t >= T)
info("done", steps=t, wall_s=time.time() - t_start, capped=t < T, lanes=str(LANES))
