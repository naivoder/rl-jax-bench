"""Record SMALL fixtures from the OFFICIAL code (@ a6b7958, run unmodified in its own venv through run_oracle.py) for
tests/test_official_parity.py.  Only the inputs we feed and the oracle's outputs are stored -- never their code.

    JAX_PLATFORMS=cpu uv run python -m scripts.oracle.record_fixtures

Writes tests/fixtures/official/<name>.npz (compressed; keys 'in__*' = what the oracle was given, 'out__*' = what it
returned) and tests/fixtures/official/MANIFEST.json (oracle commit, torch version, sha256 of each fixture).
These are reduced versions of the full differential tests in scripts/oracle/diff.py (runs/oracle_diff/diff.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import jax
import numpy as np

from scripts.oracle.diff import ORACLE_PY, RUNNER
from src import data as D
from src.model import INIT_GAIN, init_mlp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = os.path.join(REPO, "tests", "fixtures", "official")
TMP = os.path.join(REPO, "runs", "oracle_diff", "fixtures_tmp")
COMMIT = "a6b79580d85f3025bdb601566d3627c5f489f13b"


def call(case, inp, name):
    os.makedirs(TMP, exist_ok=True)
    ip, op = os.path.join(TMP, f"{name}_in.npz"), os.path.join(TMP, f"{name}_out.npz")
    np.savez(ip, **inp)
    subprocess.run([ORACLE_PY, RUNNER, case, ip, op], check=True, capture_output=True)
    out = dict(np.load(op))
    path = os.path.join(FX, f"{name}.npz")
    np.savez_compressed(path, **{f"in__{k}": np.asarray(v) for k, v in inp.items()},
                        **{f"out__{k}": v for k, v in out.items()})
    print(f"[INFO] fixture {name}: {os.path.getsize(path) / 1024:.0f} KiB", flush=True)
    return path


def main():
    os.makedirs(FX, exist_ok=True)
    paths = []
    #: teacher-forced CBP (GnT + AdamGnT), FFNN 20->5->1 tanh, all five official util types (30 steps each)
    for ut in ("adaptable_contribution", "contribution", "zero_contribution", "weight", "adaptation"):
        cfg = {"kind": "ffnn", "n_in": 20, "width": 5, "act": "tanh", "opt": "adam", "lr": 0.01, "rho": 0.05,
               "eta": 0.9, "mt": 3, "util_type": ut, "seed": 11, "n_steps": 30}
        paths.append(call("gnt_traj", cfg, f"gnt_ffnn_{ut}"))
    #: teacher-forced CBP, DeepFFNN 30->8x3->10 ReLU, SGD, CE, overall utility, accumulator (30 steps)
    cfg = {"kind": "deep", "n_in": 30, "width": 8, "n_out": 10, "n_hidden": 3, "act": "relu", "opt": "sgd",
           "lr": 0.05, "rho": 0.04, "eta": 0.99, "mt": 3, "util_type": "adaptable_contribution", "seed": 5,
           "n_steps": 30}
    paths.append(call("gnt_traj", cfg, "gnt_deep_relu_sgd"))
    #: SCR target network (FixLTUNet) on our +-1 weights and 2000 inputs, 2 seeds
    spec = D.scr_spec("flip_one")
    inp = {"n_seeds": 2}
    for s in range(2):
        root = jax.random.PRNGKey(s)
        X, _, tgt = D.scr_stream(jax.random.fold_in(root, 0), jax.random.fold_in(root, 1), spec, 1)
        inp.update({f"W_{s}": np.asarray(tgt["W"]), f"b_{s}": np.asarray(tgt["b"]), f"w2_{s}": np.asarray(tgt["w2"]),
                    f"b2_{s}": np.asarray(tgt["b2"]), f"X_{s}": np.asarray(X[:2000]).astype(np.uint8)})
    paths.append(call("ltu", inp, "ltu"))
    #: effective rank / dead units on activation matrices (official compute_matrix_rank_summaries, SciPy gesvd)
    rng = np.random.default_rng(0)
    mats = []
    for (n, k) in ((400, 60), (100, 120)):
        m = np.maximum(rng.normal(size=(n, k)) @ rng.normal(size=(k, k)) * 0.1 + rng.normal(size=k), 0)
        m = m.astype(np.float32)
        m[:, :5] = 0.0
        mats.append(m)
    mats.append((rng.normal(size=(400, 12)) @ rng.normal(size=(12, 60))).astype(np.float32))   # rank 12
    paths.append(call("diag", {"n": len(mats), **{f"m_{i}": m for i, m in enumerate(mats)}}, "diag"))
    #: whole SCR trajectories from identical init on an identical 3000-step flip-one stream
    root = jax.random.PRNGKey(0)
    X, Y, _ = D.scr_stream(jax.random.fold_in(root, 0), jax.random.fold_in(root, 1), spec, 1)
    X, Y = np.asarray(X)[:3000], np.asarray(Y)[:3000]
    for name, act, opt, lr, wd in (("bp_relu_sgd", "relu", "sgd", 0.01, 0.0), ("bp_tanh_sgd", "tanh", "sgd", 0.01, 0.0),
                                   ("l2_relu_sgd", "relu", "sgd", 0.01, 1e-3), ("linear_sgd", "linear", "sgd", 0.01, 0.0),
                                   ("bp_tanh_adam", "tanh", "adam", 0.01, 0.0),
                                   ("bp_tanh_adamgnt", "tanh", "adamgnt", 0.01, 0.0)):
        widths = () if act == "linear" else (5,)
        p = init_mlp(jax.random.PRNGKey(7), 20, widths, 1, INIT_GAIN.get(act, 1.0))
        inp = {"kind": "linear" if act == "linear" else "ffnn", "n_in": 20, "width": 5, "act": act, "opt": opt,
               "lr": lr, "wd": wd, "X": X.astype(np.uint8), "Y": Y}
        for i in range(len(widths) + 1):
            inp[f"W{i}"] = np.asarray(p["params"][f"layer_{i}"]["kernel"]).T.copy()
            inp[f"b{i}"] = np.asarray(p["params"][f"layer_{i}"]["bias"])
        paths.append(call("bp_traj", inp, f"traj_{name}"))
    #: MNIST width 100, SGD 0.003, 1000 steps of task 1 (identical permutation / order / init)
    x_u8, y_u8, _ = D.load_mnist()
    perm, order = D.mnist_task(jax.random.fold_in(root, 0), 0)
    order = np.asarray(order)[:1000]
    Xm, Ym = x_u8[order][:, np.asarray(perm)], y_u8[order].astype(np.int32)
    p = init_mlp(jax.random.PRNGKey(3), 784, (100, 100, 100), 10, INIT_GAIN["relu"])
    inp = {"kind": "mnist", "width": 100, "act": "relu", "opt": "sgd", "lr": 0.003, "wd": 0.0, "X": Xm, "Y": Ym}
    for i in range(4):
        inp[f"W{i}"] = np.asarray(p["params"][f"layer_{i}"]["kernel"]).T.copy()
        inp[f"b{i}"] = np.asarray(p["params"][f"layer_{i}"]["bias"])
    paths.append(call("bp_traj", inp, "traj_mnist_w100"))
    #: one Shrink-and-Perturb step (wd 1e-2, sigma 1e-2) and the same step without noise, width 100 MNIST net
    base = {k: v for k, v in inp.items() if k[0] in "Wb"}
    one = {"width": 100, "lr": 0.003, "wd": 1e-2, "x": Xm[:1].astype(np.float32) / 255.0, "y": Ym[:1], "torch_seed": 1}
    paths.append(call("snp_step", {**one, **base, "sigma": 1e-2}, "snp_sigma1e-2"))
    paths.append(call("snp_step", {**one, **base, "sigma": 0.0}, "snp_sigma0"))
    torch_v = subprocess.run([ORACLE_PY, "-c", "import torch; print(torch.__version__)"], capture_output=True,
                             text=True).stdout.strip()
    man = {"oracle_repo": "https://github.com/shibhansh/loss-of-plasticity", "oracle_commit": COMMIT,
           "oracle_torch": torch_v, "recorder": "scripts/oracle/record_fixtures.py",
           "files": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}}
    json.dump(man, open(os.path.join(FX, "MANIFEST.json"), "w"), indent=1)
    print("[INFO] fixtures written:", len(paths), flush=True)


if __name__ == "__main__":
    main()
