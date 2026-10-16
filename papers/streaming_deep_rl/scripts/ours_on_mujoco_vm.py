"""Review pass: VERSION-MATCHED copy of scripts/ours_on_mujoco.py. Identical agent, keys, episode
accounting and output schema; the ONLY change is that Hopper-v4 is stepped by scripts/official/hopper_env_server.py inside
the authors' venv (gymnasium 0.29.1 + mujoco 2.3.7, as their reference runs) instead of gymnasium 1.1.1 + mujoco 3.15.
Writes runs/review/ours_mujoco_vm/<ARM>/seed_<n>/metrics.json.
Original docstring follows.
Implementation test for the E6 Hopper divergence: run OUR stream AC(lambda) (src.agents.ACAgent.observe, the exact
update train.py scans; same SparseInit keys / PRNG tree as train.py) closed-loop on the REAL gymnasium MuJoCo
Hopper-v4 (the env the authors' code and Figure 7 use) instead of the Brax positional proxy. Host loop on CPU, several
seeds as vmapped lanes of one jitted update; the env steps in Python. Episode returns are raw env returns
(RecordEpisodeStatistics semantics); truncation at 1000 steps is ours (ep_len >= 1000, = gymnasium TimeLimit).
Usage: JAX_PLATFORMS=cpu python scripts/ours_on_mujoco.py <total_steps> <time_cap_s> <seed> [<seed> ...]
Writes runs/full/ours_mujoco/stream_ac/seed_<n>/metrics.json (partial results flushed every 25k steps)."""
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

T, CAP = int(sys.argv[1]), float(sys.argv[2])
SEEDS = [int(s) for s in sys.argv[3:]]
ARM = os.environ.get("ARM", "stream_ac")
L = len(SEEDS)
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
OFFICIAL_PY = os.path.join(OFFICIAL_ROOT, ".venv/bin/python")
srv = subprocess.Popen([OFFICIAL_PY, os.path.join(REPO, "scripts", "official", "hopper_env_server.py")] + [str(s) for s in SEEDS],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
ENV_DESC = "gymnasium 0.29.1 Hopper-v4, mujoco 2.3.7 (authors' venv, env server)"
agent = ag.ACAgent(Env(None, None, (11,), 3, False, "gym0.29.1-Hopper-v4-mujoco2.3.7"), overshoot_diag=True)


def _read(n):
    b = srv.stdout.read(n * 8)
    assert len(b) == n * 8, "env server died"
    return np.frombuffer(b, np.float64)
lane1 = tr.lane_cfg("ac", ARM)
lane = tr.stack_lanes("ac", [lane1] * L)
root = jax.random.PRNGKey(0)                     # master_seed 0, key tree in src/train.py
run_keys = [jax.random.fold_in(root, s) for s in SEEDS]
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


ep = [[] for _ in SEEDS]
ep_len = np.zeros(L, int)
t_start = time.time()
zero = np.zeros((L, 11), np.float32)


def flush(t_done, final=False):
    for i, s in enumerate(SEEDS):
        e = np.asarray(ep[i], float).reshape(-1, 3)
        summ = ev.summarize_run(e, (t_done // 50_000) * 50_000 or 50_000, 50_000)
        d = jax.tree.map(lambda x: float(np.asarray(x)[i]), c.diag)
        write_json(os.path.join(REPO, "runs", "review", "ours_mujoco_vm", ARM, f"seed_{s}", "metrics.json"),
                   dict(exp="rule1_ours_on_mujoco_hopper_v4", env=ENV_DESC, arm=ARM, seed=s, steps_done=t_done, final=final, summary=summ,
                        episodes_cols=["end_step", "return", "length"], episodes=e.tolist(),
                        diag=dict(d._asdict()), wall_s=time.time() - t_start))


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
    if t % 25_000 == 0:
        flush(t)
        rec = [np.mean([x[1] for x in ep[i] if x[0] > t - 25_000] or [np.nan]) for i in range(L)]
        metric("progress", steps=t, sps=t / (time.time() - t_start), last25k_mean_return=np.round(rec, 1).tolist())
flush(t, final=t >= T)
info("done", steps=t, wall_s=time.time() - t_start, capped=t < T)
