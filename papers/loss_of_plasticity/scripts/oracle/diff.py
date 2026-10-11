"""Differential tests of our JAX implementation against the official PyTorch code (the oracle).

    JAX_PLATFORMS=cpu uv run python -m scripts.oracle.diff [--quick]

Writes runs/oracle_diff/diff.json (+ the oracle's raw npz outputs).  Cases:
  D1 gnt_*   teacher-forced CBP steps: every step starts from the ORACLE's pre-step state (weights, biases, utilities,
             running means, ages, accumulator, Adam moments/step counters); our learner_step must reproduce the
             oracle's post-step state (same replaced units; utilities; weights except the randomly re-drawn incoming
             weights, which are copied from the oracle before comparing; Adam state).
  D2 ltu     our SCR target network vs the oracle FixLTUNet on identical +-1 weights and inputs (Y exact).
  D3 diag    effective rank / dead units on identical activation matrices vs the oracle's SciPy-gesvd path.
  D4 bp_*    whole trajectories from identical init weights on identical streams (SGD, L2, Adam, linear, MNIST).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from src import algorithm as A
from src import data as D
from src import diagnostics as G
from src.model import ACT_ID, INIT_GAIN, REINIT_GAIN, init_mlp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "runs", "oracle_diff")
OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
ORACLE_PY = os.path.join(OFF, ".venv", "bin", "python")
RUNNER = os.path.join(REPO, "scripts", "oracle", "run_oracle.py")
UTIL_MAP = {"adaptable_contribution": "overall", "contribution": "contribution",
            "zero_contribution": "mean_corrected_contribution", "weight": "weight_magnitude",
            "adaptation": "adaptation"}


def oracle(case, inp: dict, name: str):
    os.makedirs(OUT, exist_ok=True)
    ip = os.path.join(OUT, f"{name}_in.npz")
    op = os.path.join(OUT, f"{name}_out.npz")
    np.savez(ip, **inp)
    t = time.time()
    subprocess.run([ORACLE_PY, RUNNER, case, ip, op], check=True, capture_output=True)
    print(f"[INFO] oracle {name} done in {time.time() - t:.1f}s", flush=True)
    return dict(np.load(op))


# ---------------------------------------------------------------------------------------------------- D1
def state_from(o, prefix, n_hidden, opt, rho):
    P, M, V, Tt = {}, {}, {}, {}
    for i in range(n_hidden + 1):
        W = o[f"{prefix}W{i}"]
        b = o[f"{prefix}b{i}"]
        P[f"layer_{i}"] = {"kernel": jnp.asarray(W.T), "bias": jnp.asarray(b)}
        for store, key in ((M, "exp_avg"), (V, "exp_avg_sq"), (Tt, "step")):
            kw, kb = f"{prefix}{key}_W{i}", f"{prefix}{key}_b{i}"
            store[f"layer_{i}"] = {"kernel": jnp.asarray(o[kw].T if kw in o else np.zeros_like(W.T)),
                                   "bias": jnp.asarray(o[kb] if kb in o else np.zeros_like(b))}
    params = {"params": P}
    opt_state = {"m": {"params": M}, "v": {"params": V}, "t": {"params": Tt}} if opt == "adam" else {}
    cbp = {k: [jnp.asarray(o[f"{prefix}{k}{l}"], jnp.float32) for l in range(n_hidden)] for k in ("util", "bcu", "f", "age")}
    cbp["acc"] = [jnp.asarray(np.round(float(o[f"{prefix}acc{l}"]) / rho), jnp.float32) for l in range(n_hidden)]
    return {"params": params, "opt": opt_state, "cbp": cbp}


def gnt_case(name, cfg, n_hidden):
    o = oracle("gnt_traj", cfg, name)
    st = A.Static(n_in=int(cfg["n_in"]), widths=(int(cfg["width"]),) * n_hidden,
                  n_out=int(cfg.get("n_out", 1)), loss="mse" if cfg["kind"] == "ffnn" else "ce", opt=str(cfg["opt"]),
                  replace_mode="accumulate", k_max=int(cfg.get("k_max", 1)))
    act = str(cfg["act"])
    hp = A.make_hp(lr=float(cfg["lr"]), rho=float(cfg["rho"]), mt=int(cfg["mt"]), eta=float(cfg["eta"]),
                   util=UTIL_MAP[str(cfg["util_type"])], act=ACT_ID[act], reinit_gain=REINIT_GAIN[act])
    step = jax.jit(lambda s, x, y, k: A.learner_step(st, s, hp, x, y, k))
    worst = {"weights": 0.0, "util": 0.0, "bcu": 0.0, "f": 0.0, "adam": 0.0}
    n_rep_total, mismatched_sel, tie_sel, steps = 0, 0, 0, int(o["n_steps"])
    for t in range(steps):
        s = state_from(o, f"t{t}_pre_", n_hidden, st.opt, float(cfg["rho"]))
        x = jnp.asarray(o[f"t{t}_x"])
        y = jnp.asarray(o[f"t{t}_y"]) if st.loss == "mse" else jnp.asarray(o[f"t{t}_y"]).astype(jnp.int32)
        s2, info = step(s, x, y, jax.random.PRNGKey(t))
        ref = state_from(o, f"t{t}_post_", n_hidden, st.opt, float(cfg["rho"]))
        tie_step = False
        for l in range(n_hidden):
            rep_o = set(o[f"t{t}_rep{l}"].tolist())
            rep_u = set(np.where(np.asarray(s2["cbp"]["age"][l]) == 0)[0].tolist())
            n_rep_total += len(rep_o)
            if rep_o != rep_u:
                # an exact tie in the bias-corrected utility (e.g. several dead ReLU units at utility 0) is broken
                # differently by torch.topk and lax.top_k; anything else is a real mismatch
                b = o[f"t{t}_post_bcu{l}"]
                vals = {float(b[i]) for i in rep_o ^ rep_u}
                if len(rep_o) == len(rep_u) and len(vals) == 1:
                    tie_sel += 1
                    tie_step = True
                else:
                    mismatched_sel += 1
            # copy the oracle's randomly drawn incoming weights of the replaced units into ours before comparing
            k = s2["params"]["params"][f"layer_{l}"]["kernel"]
            for r in rep_o:
                k = k.at[:, r].set(ref["params"]["params"][f"layer_{l}"]["kernel"][:, r])
            s2["params"]["params"][f"layer_{l}"]["kernel"] = k
            for key in ("util", "bcu", "f"):
                worst[key] = max(worst[key], float(np.abs(np.asarray(s2["cbp"][key][l]) - np.asarray(ref["cbp"][key][l])).max()))
        if tie_step:
            continue
        for a, b in zip(jax.tree.leaves(s2["params"]), jax.tree.leaves(ref["params"])):
            worst["weights"] = max(worst["weights"], float(np.abs(np.asarray(a) - np.asarray(b)).max()))
        if st.opt == "adam":
            for a, b in zip(jax.tree.leaves(s2["opt"]), jax.tree.leaves(ref["opt"])):
                worst["adam"] = max(worst["adam"], float(np.abs(np.asarray(a) - np.asarray(b)).max()))
    ok = mismatched_sel == 0 and n_rep_total > 0 and max(worst.values()) < 1e-4
    return {"pass": bool(ok), "steps": steps, "n_replacements_oracle": n_rep_total,
            "layer_steps_with_different_selection": mismatched_sel,
            "layer_steps_with_tie_broken_differently": tie_sel, "max_abs_diff": worst}


def d1(quick):
    res = {}
    n = 40 if quick else 120
    for ut in ("adaptable_contribution", "contribution", "zero_contribution", "weight", "adaptation"):
        cfg = {"kind": "ffnn", "n_in": 20, "width": 5, "act": "tanh", "opt": "adam", "lr": 0.01, "rho": 0.05,
               "eta": 0.9, "mt": 3, "util_type": ut, "seed": 11, "n_steps": n}
        res[f"gnt_ffnn_tanh_adam_{ut}"] = gnt_case(f"gnt_ffnn_{ut}", cfg, 1)
    cfg = {"kind": "deep", "n_in": 30, "width": 8, "n_out": 10, "n_hidden": 3, "act": "relu", "opt": "sgd",
           "lr": 0.05, "rho": 0.04, "eta": 0.99, "mt": 3, "util_type": "adaptable_contribution", "seed": 5,
           "n_steps": n}
    res["gnt_deep_relu_sgd_overall"] = gnt_case("gnt_deep", cfg, 3)
    return res


# ---------------------------------------------------------------------------------------------------- D2
def d2(quick):
    spec = D.scr_spec("flip_one")
    n_per = 2 if quick else 10
    inp = {"n_seeds": 5}
    ours = {}
    for s in range(5):
        root = jax.random.PRNGKey(s)
        X, Y, tgt = D.scr_stream(jax.random.fold_in(root, 0), jax.random.fold_in(root, 1), spec, n_per)
        inp.update({f"W_{s}": np.asarray(tgt["W"]), f"b_{s}": np.asarray(tgt["b"]), f"w2_{s}": np.asarray(tgt["w2"]),
                    f"b2_{s}": np.asarray(tgt["b2"]), f"X_{s}": np.asarray(X).astype(np.uint8)})
        ours[s] = (np.asarray(Y), np.asarray(tgt["tau"]))
    o = oracle("ltu", inp, "ltu")
    err = max(float(np.abs(o[f"Y_{s}"] - ours[s][0]).max()) for s in range(5))
    tau = max(float(np.abs(o[f"tau_{s}"] - ours[s][1]).max()) for s in range(5))
    import hashlib
    sha = {s: hashlib.sha256(inp[f"X_{s}"].tobytes() + ours[s][0].astype(np.float32).tobytes()).hexdigest()
           for s in range(5)}
    return {"ltu": {"pass": bool(err == 0.0 and tau == 0.0), "max_abs_Y_diff": err, "max_abs_tau_diff": tau,
                    "steps_per_seed": n_per * spec.T, "stream_sha256_seed0_4": sha}}


# ---------------------------------------------------------------------------------------------------- D3
def d3(quick):
    rng = np.random.default_rng(0)
    mats = []
    for (n, k) in ((2000, 100), (2000, 100), (500, 50), (100, 200)):
        m = np.maximum(rng.normal(size=(n, k)) @ rng.normal(size=(k, k)) * 0.1 + rng.normal(size=k), 0).astype(np.float32)
        m[:, :7] = 0.0
        mats.append(m)
    mats.append((rng.normal(size=(2000, 30)) @ rng.normal(size=(30, 100))).astype(np.float32))   # rank 30
    inp = {"n": len(mats), **{f"m_{i}": m for i, m in enumerate(mats)}}
    o = oracle("diag", inp, "diag")
    rows = []
    ok = True
    for i, m in enumerate(mats):
        er = float(G.effective_rank(jnp.asarray(m)))
        dead = int(G.dead_units(jnp.asarray(m)).sum())
        rel = abs(er - float(o[f"erank_{i}"])) / float(o[f"erank_{i}"])
        rows.append({"erank_ours": er, "erank_oracle": float(o[f"erank_{i}"]), "rel_diff": rel, "dead_ours": dead,
                     "dead_oracle": int(o[f"dead_{i}"])})
        ok &= rel < 1e-4 and dead == int(o[f"dead_{i}"])
    return {"diag": {"pass": bool(ok), "matrices": rows}}


# ---------------------------------------------------------------------------------------------------- D4
def our_traj(st, params, hp, X, Y):
    """Our learner on a given stream from given weights.  One PRNG key per step (fold_in(key, t), as src/train.py does);
    a constant key would make the Bernoulli replacement coin identical at every step (recheck)."""
    s = A.init_state(st, params)
    k0 = jax.random.PRNGKey(0)

    def one(s, xyt):
        x, y, t = xyt
        s, info = A.learner_step(st, s, hp, x[None], y[None] if st.loss == "ce" else y[None, None],
                                 jax.random.fold_in(k0, t))
        return s, info["loss"]

    s, errs = jax.jit(lambda s: jax.lax.scan(one, s, (X, Y, jnp.arange(len(X)))))(s)
    return np.asarray(errs), s


def d4(quick):
    res = {}
    spec = D.scr_spec("flip_one")
    n_per = 3 if quick else 20
    root = jax.random.PRNGKey(0)
    X, Y, _ = D.scr_stream(jax.random.fold_in(root, 0), jax.random.fold_in(root, 1), spec, n_per)
    X, Y = np.asarray(X), np.asarray(Y)
    cases = [("bp_relu_sgd_0.01", "relu", "sgd", 0.01, 0.0), ("bp_tanh_sgd_0.01", "tanh", "sgd", 0.01, 0.0),
             ("l2_relu_sgd_0.01_wd1e-3", "relu", "sgd", 0.01, 1e-3), ("bp_tanh_adam_0.01", "tanh", "adam", 0.01, 0.0),
             ("linear_sgd_0.01", "linear", "sgd", 0.01, 0.0),
             ("bp_tanh_adamgnt_0.01", "tanh", "adamgnt", 0.01, 0.0)]
    only = os.environ.get("D4_ONLY")
    for name, act, opt, lr, wd in cases:
        if only and name not in only.split(","):
            continue
        widths = () if act == "linear" else (5,)
        st = A.Static(n_in=20, widths=widths, n_out=1, opt="adam" if opt.startswith("adam") else opt)
        p = init_mlp(jax.random.PRNGKey(7), 20, widths, 1, INIT_GAIN.get(act, 1.0))
        hp = A.make_hp(lr=lr, wd=wd, act=ACT_ID.get(act, 0))
        inp = {"kind": "linear" if act == "linear" else "ffnn", "n_in": 20, "width": 5, "act": act, "opt": opt,
               "lr": lr, "wd": wd, "X": X.astype(np.uint8), "Y": Y}
        for i in range(len(widths) + 1):
            inp[f"W{i}"] = np.asarray(p["params"][f"layer_{i}"]["kernel"]).T.copy()
            inp[f"b{i}"] = np.asarray(p["params"][f"layer_{i}"]["bias"])
        o = oracle("bp_traj", inp, name)
        ours, s = our_traj(st, p, hp, jnp.asarray(X), jnp.asarray(Y))
        d = np.abs(ours - o["errs"])
        bins_o = o["errs"].reshape(-1, spec.T).mean(1)
        bins_u = ours.reshape(-1, spec.T).mean(1)
        wd_ = float(np.abs(np.asarray(s["params"]["params"]["layer_0"]["kernel"]).T - o["W0"]).max())
        rel_bin = float(np.max(np.abs(bins_u - bins_o) / np.abs(bins_o)))
        div = np.where(d > 1e-3)[0]
        # Adam rescales near-zero gradients to O(lr) steps, so float-rounding differences between torch and XLA
        # grow chaotically (scripts/oracle/chaos.py: our OWN run with a 1e-7 relative init perturbation diverges at
        # the same point).  For Adam the criterion is therefore the pre-divergence prefix (first 10k steps).
        ok = rel_bin < 1e-3 if not opt.startswith("adam") else float(d[:10_000].max()) < 1e-3
        res[name] = {"pass": bool(ok), "steps": int(len(ours)), "max_abs_diff_first_10k": float(d[:10_000].max()), "max_abs_step_err_diff": float(d.max()),
                     "median_abs_step_err_diff": float(np.median(d)), "max_rel_10k_bin_diff": rel_bin,
                     "first_step_abs_diff_gt_1e-3": int(div[0]) if len(div) else None,
                     "final_W0_max_abs_diff": wd_, "mse_ours": float(ours.mean()), "mse_oracle": float(o["errs"].mean())}
    if only and "mnist" not in only:
        return res
    # MNIST width 100, SGD 0.003, first task, identical permutation / order / init
    x_u8, y_u8, _ = D.load_mnist()
    perm, order = D.mnist_task(jax.random.fold_in(root, 0), 0)
    n = 3000 if quick else 20000
    perm, order = np.asarray(perm), np.asarray(order)[:n]
    Xm = x_u8[order][:, perm]
    Ym = y_u8[order].astype(np.int32)
    st = A.Static(n_in=784, widths=(100, 100, 100), n_out=10, loss="ce")
    p = init_mlp(jax.random.PRNGKey(3), 784, (100, 100, 100), 10, INIT_GAIN["relu"])
    inp = {"kind": "mnist", "width": 100, "act": "relu", "opt": "sgd", "lr": 0.003, "wd": 0.0, "X": Xm, "Y": Ym}
    for i in range(4):
        inp[f"W{i}"] = np.asarray(p["params"][f"layer_{i}"]["kernel"]).T.copy()
        inp[f"b{i}"] = np.asarray(p["params"][f"layer_{i}"]["bias"])
    o = oracle("bp_traj", inp, "bp_mnist_w100")
    ours, s = our_traj(st, p, A.make_hp(lr=0.003), jnp.asarray(Xm, jnp.float32) / 255.0, jnp.asarray(Ym))
    d = np.abs(ours - o["errs"])
    res["bp_mnist_w100_sgd_0.003"] = {"pass": bool(np.median(d) < 1e-4 and abs(ours.mean() - o["errs"].mean()) < 1e-3),
                                      "steps": n, "max_abs_step_loss_diff": float(d.max()),
                                      "median_abs_step_loss_diff": float(np.median(d)),
                                      "mean_loss_ours": float(ours.mean()), "mean_loss_oracle": float(o["errs"].mean())}
    return res


def main():
    quick = "--quick" in sys.argv
    which = [a for a in sys.argv[1:] if a.startswith("D")] or ["D1", "D2", "D3", "D4"]
    res = {}
    for k in which:
        res.update({"D1": d1, "D2": d2, "D3": d3, "D4": d4}[k](quick))
    for k, v in res.items():
        print(f"[METRIC] oracle_diff {k} pass={v['pass']} " +
              " ".join(f"{a}={b}" for a, b in v.items() if a not in ("pass", "matrices", "stream_sha256_seed0_4")),
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "diff.json" if not quick else "diff_quick.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, default=float)
    print("[INFO] ORACLE_DIFF", "PASS" if all(v["pass"] for v in old.values()) else "FAIL", flush=True)


if __name__ == "__main__":
    main()
