"""Oracle MNIST reference: official DeepFFNN learners on our first tasks (same permutations / example orders as our
seeds), width 100, step 0.003.  JAX_PLATFORMS=cpu python -m scripts.oracle.mnist_reference [--tasks 5] [--procs 3]
Outputs runs/oracle_ref/mnist_<arm>/seed_<s>.npz."""
import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np

from src import data as D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "runs", "oracle_ref")
OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
ORACLE_PY = os.path.join(OFF, ".venv", "bin", "python")
RUNNER = os.path.join(REPO, "scripts", "oracle", "run_oracle.py")
ARMS = {"bp": dict(rho=0.0, wd=0.0, sigma=0.0), "cbp_1e-4": dict(rho=1e-4, wd=0.0, sigma=0.0)}


def run(arm, seed, tasks, n_ex):
    out = os.path.join(OUT, f"mnist_{arm}", f"seed_{seed}.npz")
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    x, y, _ = D.load_mnist()
    kd = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
    perms, orders = zip(*[D.mnist_task(kd, t) for t in range(tasks)])
    ip = out.replace(".npz", "_in.npz")
    np.savez(ip, X=x, Y=y, perms=np.asarray(perms), orders=np.asarray(orders), width=100, lr=0.003, n_ex=n_ex,
             torch_seed=20_000 + seed, **ARMS[arm])
    t = time.time()
    subprocess.run([ORACLE_PY, RUNNER, "mnist_run", ip, out], check=True, capture_output=True)
    os.remove(ip)
    print(f"[INFO] oracle mnist {arm} seed={seed} {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--n_ex", type=int, default=60_000)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args()
    jobs = [(arm, s) for arm in ARMS for s in a.seeds]
    with ThreadPoolExecutor(a.procs) as ex:
        list(ex.map(lambda j: run(j[0], j[1], a.tasks, a.n_ex), jobs))


if __name__ == "__main__":
    main()
