"""Pre-flight for the full-scale run: machine-checks the protocol items that can be checked mechanically and prints the evidence.
    JAX_PLATFORMS=cpu python scripts/preflight.py
"""
import hashlib, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np, yaml
import model as M, algorithm as A, data as D, train as T
from rl_jax_bench import jaxlog

ok = True
def chk(name, cond, ev=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} :: {ev}")

# 1 sources.yaml
a = hashlib.sha256(open(os.path.join(HERE, "..", "src", "configs", "sources.yaml"), "rb").read()).hexdigest()
print("sources.yaml sha256", a[:16])
y = yaml.safe_load(open(os.path.join(HERE, "..", "src", "configs", "sources.yaml")))
untagged = []
def walk(d, path=""):
    if isinstance(d, dict):
        if "source" in d or "value" in d:
            if not str(d.get("source", "")).strip():
                untagged.append(path)
        else:
            for k, v in d.items(): walk(v, path + "/" + str(k))
walk(y)
chk("every sources.yaml entry carries a source tag", not untagged, f"untagged={untagged[:5]}")
# 2 devices / x64
chk("jax_enable_x64 asserted", jax.config.jax_enable_x64, f"devices={jax.devices()}")
# 3 constants from config
chk("T=100, t from 0, beta=0.9, tau=100 from the config (model.py == sources.yaml)", (M.T_HORIZON, M.BETA, M.TAU, M.N_MC) == (100, 0.9, 100, 100), f"{(M.T_HORIZON, M.BETA, M.TAU, M.N_MC)}")
cfg = A.Cfg(); chk("Cfg defaults take T,beta,tau from model.py", (cfg.T, cfg.beta, cfg.tau) == (M.T_HORIZON, M.BETA, M.TAU), str(cfg))
# 4 env checksum
chk("frozen reference trajectory", M.assert_env_checksum() <= M.REF_TOL, f"{M.reference_trajectory_error():.2e}")
# 5 policies share P0 and tie keys
s = D.TABLE_SCENARIOS["t3_q2"]
rk = D.rep_keys(0, "E5", s.sid, 2, 5); p0k, tie, mode = D.split_rep_keys(rk)
rk2 = D.rep_keys(0, "E5", s.sid, 2, 5); P0a = D.sample_P0(s, D.split_rep_keys(rk)[0]); P0b = D.sample_P0(s, D.split_rep_keys(rk2)[0])
chk("P0/tie keys identical for every policy of a cell (run_cell draws P0 once and passes the same P0, tie, mode to every policy)", bool(jnp.all(P0a == P0b)),
    "train.run_cell: one P0,tie,mode per cell; deterministic key tree")
import inspect
src = inspect.getsource(T.run_cell)
chk("run_cell passes the same (P0, tie, mode) to A.simulate for all policies", src.count("A.simulate(bank, P0, tie, mode, pol, K, cfg)") == 2 and "for pol in policies" in src)
# 6 variants labelled
chk("variant labels", {T.variant_name(A.Cfg(skip_neg=True), "printed"), T.variant_name(A.Cfg(), "standard"), T.variant_name(A.Cfg(sampled=True), "printed"), T.variant_name(A.Cfg(pcli1="literal"), "printed")} == {"skipneg", "ft_standard", "sampled", "pcli1_literal"})
# 7 NaN guard aborts (the guard calls os._exit(3) after logging [ERROR]: run it in a child process)
child = r"""
import os, sys
sys.path.insert(0, os.path.join(%r, "..", "src"))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax.numpy as jnp
import model as M, algorithm as A, data as D
A.index_tev = lambda bank, P, cfg=None: (jnp.full(P.shape[0], jnp.nan), jnp.zeros((), jnp.int32))
s = D.TABLE_SCENARIOS["t3_q2"]; p0k, tie, mode = D.split_rep_keys(D.rep_keys(0, "E5", s.sid, 2, 3))
out = A.simulate(D.build_bank(s), D.sample_P0(s, p0k), tie, mode, "tev", 2, A.Cfg(T=3, guard=True)); out["J"].block_until_ready()
print("NOT ABORTED")
""" % HERE
import tempfile
r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, cwd=tempfile.mkdtemp())
chk("NaN/Inf guard aborts the run with [ERROR] on a non-finite index", r.returncode == 3 and "[ERROR] NaN/Inf detected" in (r.stdout + r.stderr), f"exit code {r.returncode}; " + (r.stdout + r.stderr).strip().splitlines()[-1][:100])
print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
