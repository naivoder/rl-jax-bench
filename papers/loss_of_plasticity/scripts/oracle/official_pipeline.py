"""Reference runs of the OFFICIAL end-to-end pipeline (github.com/shibhansh/loss-of-plasticity @ a6b7958, unmodified):
their own SCR data generator (slowly_changing_regression.generate_problem_data), their own experiment entry points
(slowly_changing_regression/expr.py, permuted_mnist/online_expr.py) driven by cfg JSON files that mirror their
cfg/*.json with the swept list resolved to one value.  Nothing of ours touches the data or the learner here.

    uv run python -m scripts.oracle.official_pipeline [--procs 6] [--seeds 0 1 2 3 4] [--only scr|mnist]

The only things we add, outside their code, are (i) torch.manual_seed(<seed>) before each of their scripts runs, so a
run is reproducible (their scripts do not seed; each process otherwise gets a random torch seed), (ii) the MNIST
pickle `data/mnist_` that their load_mnist.py would build with torchvision (not installed in their venv): we build
the same object (x float32 in [0,1], shape (60000,784); y int64) from the canonical idx files, and (iii) shorter
runs (num_data_points / num_tasks), labelled.

Work dir (outside their git tree): $OFFICIAL_ROOT/work/ (default external/official/work)
Outputs: work/out/<case>/seed_<s>.pkl (their own pickles) + runs/oracle_pipeline/summary.json (via --summarise).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
PY = f"{OFF}/.venv/bin/python"
REPO_OFF = f"{OFF}/repo"
WORK = f"{OFF}/work"
SCR_DIR = f"{REPO_OFF}/lop/slowly_changing_regression"
MNIST_DIR = f"{REPO_OFF}/lop/permuted_mnist"
OURS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV = dict(os.environ, PYTHONPATH=REPO_OFF, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

# SCR cases.  base = their cfg file the case mirrors; over = the list entry picked (or paper-stated override).
SCR_BASE = {"num_inputs": 20, "flip_after": 10000, "num_target_features": 100, "num_flipping_bits": 15, "beta": 0.7,
            "num_features": 5, "to_log_features": False, "gen_prob_data": False}
SCR_CASES = {
    # official default cfg/flip_one/{relu,tanh}.json, step_size list entry 0 (0.01), 3 M points  -> Fig. B.10
    "flip_relu_bp_0.01": ("flip_one", 3_000_000, dict(agent="bp", hidden_activation="relu", opt="sgd", step_size=0.01,
                                                     replacement_rate=0, decay_rate=0)),
    "flip_tanh_bp_0.01": ("flip_one", 3_000_000, dict(agent="bp", hidden_activation="tanh", opt="sgd", step_size=0.01,
                                                     replacement_rate=0, decay_rate=0)),
    # official default cfg/sgd/bp/relu.json (entry 0.01), relu_wd.json (1e-2), relu_snp.json (wd 1e-2, sigma 1e-3),
    # cfg/sgd/cbp/relu.json (rho 1e-4): re-sample-all stream, 1 M points.
    "rs_relu_bp_0.01": ("resample", 1_000_000, dict(agent="bp", hidden_activation="relu", opt="sgd", step_size=0.01,
                                                    replacement_rate=0, decay_rate=0)),
    "rs_relu_l2_1e-2": ("resample", 1_000_000, dict(agent="bp", hidden_activation="relu", opt="sgd", step_size=0.01,
                                                    replacement_rate=0, decay_rate=0, weight_decay=1e-2)),
    "rs_relu_snp_1e-3": ("resample", 1_000_000, dict(agent="bp", hidden_activation="relu", opt="sgd", step_size=0.01,
                                                     replacement_rate=0, decay_rate=0, weight_decay=1e-2,
                                                     perturb_scale=1e-3)),
    # their cfg sets "maturity_threshold": 100, which expr.py ignores (it reads "mt", default 10) -> effective m = 10
    "rs_relu_cbp_1e-4_default": ("resample", 1_000_000, dict(agent="cbp", hidden_activation="relu", opt="sgd",
                                                             step_size=0.01, replacement_rate=1e-4, decay_rate=0.99,
                                                             util_type="adaptable_contribution",
                                                             maturity_threshold=100)),
    # the paper / cfg-stated m = 100 actually applied (key "mt"): where the paper and the code's effective value differ
    "rs_relu_cbp_1e-4_mt100": ("resample", 1_000_000, dict(agent="cbp", hidden_activation="relu", opt="sgd",
                                                           step_size=0.01, replacement_rate=1e-4, decay_rate=0.99,
                                                           util_type="adaptable_contribution", mt=100)),
}
# MNIST: official cfg/bp/small_net.json (width 100; step 0.003 entry) and cfg/cbp.json with num_features 100
# (the paper's Fig. 6 runs use width 2000: 26 ms/step on one CPU thread = 26 min/task, out of budget -> labelled).
MNIST_TASKS = 15
MNIST_CASES = {
    "mnist_w100_bp_0.003": dict(agent="bp", opt="sgd", num_features=100, num_hidden_layers=3, step_size=0.003,
                                weight_decay=0, to_log=True, use_gpu=0),
    "mnist_w100_cbp_1e-4": dict(agent="cbp", opt="sgd", num_features=100, num_hidden_layers=3, step_size=0.003,
                                accumulate=True, util_type="adaptable_contribution", replacement_rate=1e-4,
                                decay_rate=0.99, mt=100, to_log=True, use_gpu=0),
}


def gen_scr(stream, seed, n):
    path = f"{WORK}/env_data/{stream}_{n}/{seed}"
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    code = ("import torch; torch.manual_seed(%d); "
            "from lop.slowly_changing_regression.slowly_changing_regression import generate_problem_data as g; "
            "g(flip_after=10000, data_file=%r, num_data_points=%d, num_inputs=20, num_target_features=100, "
            "num_flipping_bits=15, beta=0.7, flip_one=%s)") % (7_000 + seed, path + ".tmp", n, stream == "flip_one")
    subprocess.run([PY, "-c", code], check=True, env=ENV, capture_output=True, cwd=SCR_DIR)
    os.replace(path + ".tmp", path)
    return path


def run_script(script, cfg_path, seed, cwd):
    code = ("import torch, sys, runpy; torch.manual_seed(%d); sys.argv=[%r, '-c', %r]; "
            "runpy.run_path(%r, run_name='__main__')") % (seed, script, cfg_path, script)
    subprocess.run([PY, "-c", code], check=True, env=ENV, capture_output=True, cwd=cwd)


TORCH_SEED_BASE = 30_000   # learner init seed of run s = TORCH_SEED_BASE + s; other bases write to out/<case>_ts<base>/


def scr_job(case, seed):
    stream, n, over = SCR_CASES[case]
    sfx = "" if TORCH_SEED_BASE == 30_000 else f"_ts{TORCH_SEED_BASE}"
    out = f"{WORK}/out/{case}{sfx}/seed_{seed}.pkl"
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env_file = gen_scr(stream, seed, n)
    p = dict(SCR_BASE, num_data_points=n, flip_one=stream == "flip_one", env_file=env_file, data_file=out + ".tmp",
             **over)
    cfg = f"{WORK}/cfg/{case}{sfx}_{seed}.json"
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    json.dump(p, open(cfg, "w"), indent=1)
    t = time.time()
    run_script(f"{SCR_DIR}/expr.py", cfg, TORCH_SEED_BASE + seed, SCR_DIR)
    os.replace(out + ".tmp", out)
    print(f"[INFO] official_pipeline {case}{sfx} seed={seed} {time.time() - t:.0f}s", flush=True)


def mnist_pickle():
    path = f"{WORK}/mnist_cwd/data/mnist_"
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import sys
    sys.path.insert(0, OURS)
    from src.data import load_mnist  # md5-checked canonical idx files
    x, y, _ = load_mnist()
    np.savez(path + ".npz", x=x, y=y)
    code = ("import numpy as np, torch, pickle; d=np.load(%r); x=torch.tensor(d['x'].astype(np.float32)/255.0); "
            "y=torch.tensor(d['y'].astype(np.int64)); pickle.dump([x, y, None, None], open(%r, 'wb'))"
            ) % (path + ".npz", path)
    subprocess.run([PY, "-c", code], check=True, env=ENV)
    os.remove(path + ".npz")


def mnist_job(case, seed):
    out = f"{WORK}/out/{case}/seed_{seed}.pkl"
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p = dict(MNIST_CASES[case], num_tasks=MNIST_TASKS, num_runs=1, data_file=out + ".tmp")
    cfg = f"{WORK}/cfg/{case}_{seed}.json"
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    json.dump(p, open(cfg, "w"), indent=1)
    t = time.time()
    # online_expr draws its permutations with np.random (unseeded in their code) -> seed numpy too
    code = ("import torch, numpy, sys, runpy; torch.manual_seed(%d); numpy.random.seed(%d); "
            "sys.argv=[%r, '-c', %r]; runpy.run_path(%r, run_name='__main__')"
            ) % (40_000 + seed, 40_000 + seed, "online_expr.py", cfg, f"{MNIST_DIR}/online_expr.py")
    subprocess.run([PY, "-c", code], check=True, env=ENV, capture_output=True, cwd=f"{WORK}/mnist_cwd")
    os.replace(out + ".tmp", out)
    print(f"[INFO] official_pipeline {case} seed={seed} {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--mnist_seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--only", default="")
    ap.add_argument("--torch_seed_base", type=int, default=30_000)
    ap.add_argument("--cases", nargs="*", default=None, help="restrict to these SCR / MNIST case names")
    a = ap.parse_args()
    jobs = []
    if a.only in ("", "mnist"):
        mnist_pickle()
        jobs += [("mnist", c, s) for s in a.mnist_seeds for c in MNIST_CASES]
    if a.only in ("", "scr"):
        # longest first
        jobs += [("scr", c, s) for c in ("flip_relu_bp_0.01", "flip_tanh_bp_0.01") for s in a.seeds]
        jobs += [("scr", c, s) for c in SCR_CASES if not c.startswith("flip") for s in a.seeds]
    global TORCH_SEED_BASE
    TORCH_SEED_BASE = a.torch_seed_base
    if a.cases:
        jobs = [j for j in jobs if j[1] in a.cases]
    # pre-generate streams serially (avoid two jobs racing on one file)
    for kind, c, s in jobs:
        if kind == "scr":
            gen_scr(SCR_CASES[c][0], s, SCR_CASES[c][1])

    def go(j):
        kind, c, s = j
        try:
            (scr_job if kind == "scr" else mnist_job)(c, s)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] official_pipeline {c} seed={s}: {e.stderr[-2000:] if e.stderr else e}", flush=True)

    with ThreadPoolExecutor(a.procs) as ex:
        list(ex.map(go, jobs))
    print("[INFO] official_pipeline done", flush=True)


if __name__ == "__main__":
    main()
