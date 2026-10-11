"""Recheck report (see README): RC1 before/after, the three-way table (paper / SMPyBandits
on our hardware / ours), the RC2 single-realisation block, RC3 family Td and RC4 gap-sweep slopes.

    uv run python scripts/recheck_report.py   -> runs/recheck/report.json, runs/recheck/report_tables.txt

No GPU. Inputs: runs/full (headline runs, report.json and report_eye.json from report.py), runs/recheck
(RC3 runs), runs/x1_gapsweep (RC4), runs/official/ref (SMPyBandits reference runs, CPU).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import yaml
from scipy import stats as st

from rl_jax_bench.stats import holm, paired_compare  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from src import verdicts as V  # noqa: E402
G3 = os.path.join(ROOT, "runs", "full")
SA = os.path.join(ROOT, "runs", "recheck")
ORC = os.path.join(ROOT, "runs", "official", "ref")  # written by scripts/official/oracle_reference.py
OUT: list[str] = []


def p(s=""):
    OUT.append(s)
    print(s)


def recs(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "seed_*", "metrics.json")), key=lambda f: int(f.split("seed_")[1].split("/")[0]))]


def arm(d, key):
    R = recs(d)
    return {"seed_means": np.array([r["policies"][key]["final_regret_mean"] for r in R]),
            "reps": np.concatenate([r["policies"][key]["final_regret"] for r in R]).astype(float),
            "arm1": np.concatenate([r["policies"][key]["final_arm1"] for r in R]).astype(float),
            "at": {k: np.mean([r["policies"][key]["regret_at_mean"][k] for r in R]) for k in R[0]["policies"][key]["regret_at_mean"]},
            "fps": [r["env"]["uniforms_fp_sha256"] for r in R], "seeds": [r["seed"] for r in R],
            "params": R[0]["policies"][key]["params"]}


def judge(rel):
    return V.judge_category(abs(rel))


# ---------------------------------------------------------------- RC1 before / after
def rc1():
    p("## RC1: Figure-1 cells, eye read-off (pre-fix) vs vector read-off (post-fix); our numbers unchanged (no re-run)\n")
    before = json.load(open(os.path.join(G3, "report_eye.json")))["verdicts"]
    after = json.load(open(os.path.join(G3, "report.json")))["verdicts"]
    p("| exp | metric | arm | paper eye (±10/±100) | paper vector (±1/±5) | ours mean ± SE [run SE], n | Δ% eye → vector | judgment eye → vector | per-run pct eye → vector |")
    p("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for e in ("e1", "e2"):
        for b, a in zip(before[e], after[e]):
            assert (b["metric"], b["arm"], b["mean"]) == (a["metric"], a["arm"], a["mean"])
            flip = b["judgment"] != a["judgment"]
            rows.append({"exp": e, "metric": a["metric"], "arm": a["arm"], "eye": b["paper"], "vector": a["paper"], "mean": a["mean"],
                         "rel_eye": b["rel"], "rel_vec": a["rel"], "j_eye": b["judgment"], "j_vec": a["judgment"], "flip": flip})
            jj = f"**{b['judgment']} → {a['judgment']}**" if flip else f"{a['judgment']} (unchanged)"
            p(f"| {e} | {a['metric']} | {a['arm']} | {b['paper']} | {a['paper']:.1f} | {a['mean']:.1f} ± {a['se']:.1f} [{a['se_run']:.1f}], {a['n']} | "
              f"{100 * b['rel']:+.1f} → {100 * a['rel']:+.1f} | {jj} | {b['paper_pct_in_runs']:.1f} → {a['paper_pct_in_runs']:.1f} |")
    for lab, key in (("pre-fix", "j_eye"), ("post-fix", "j_vec")):
        prim = [r[key] for r in rows if r["metric"] == "final R_T" and "secondary" not in r["arm"]]
        p(f"\nPrimary 8 ({lab}): MATCH {prim.count('MATCH')}, CLOSE {prim.count('CLOSE')}, DIVERGENT {prim.count('DIVERGENT')}")
    return rows


# ---------------------------------------------------------------- three-way table
ORC_MAP = {"ucb1": "ucb1_paper(xi=1/2)", "ducb": "ducb_printed2B(alpha=4)", "ducb_padB": "ducb_smpy_default_pad(alpha=1)",
           "swucb": "swucb_paper(alpha=1/2)", "exp3s": "exp3s_paper(S=2)"}


def threeway():
    p("\n## Three-way table: paper (vector) / SMPyBandits on our hardware (third-party, labelled second-best; unpatched, "
      "CPU, 4 seeds × 60 runs) / ours (n = 10⁴)\n")
    p("No official D-UCB / SW-UCB / EXP3.S code exists (see README). SMPyBandits commit 012fc13a, its own random ties and "
      "initialisation, numpy RNG; padding B = SMPy default alpha = 1, printed 2B = alpha = 4.\n")
    p("| exp | arm | paper | SMPy mean ± SE(seed) [run SE] | paper pct in SMPy runs | ours mean [run SE] | paper pct in our runs | ours − SMPy (z) |")
    p("|---|---|---|---|---|---|---|---|")
    out = {}
    for e in ("e1", "e2"):
        ro = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", f"{e}.yaml")))["readoff"]["final_regret"]
        ds = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ORC, f"ref_{e}_s*.json")))]
        for a in ("ucb1", "exp3s", "ducb", "ducb_padB", "swucb"):
            name = ORC_MAP[a]
            reg = np.concatenate([d["res"][name]["regret"] for d in ds]).astype(float)
            sm = np.array([np.mean(d["res"][name]["regret"]) for d in ds])
            ours = arm(os.path.join(G3, f"padb_{e}" if a == "ducb_padB" else e), a).get("reps")
            pv = ro["ducb" if a == "ducb_padB" else a]
            se_o, se_u = reg.std(ddof=1) / math.sqrt(reg.size), ours.std(ddof=1) / math.sqrt(ours.size)
            z = (ours.mean() - reg.mean()) / math.hypot(se_o, se_u)
            out[f"{e}|{a}"] = {"paper": pv, "smpy_mean": reg.mean(), "smpy_se_seed": sm.std(ddof=1) / math.sqrt(sm.size), "smpy_se_run": se_o,
                               "smpy_n": reg.size, "smpy_paper_pct": 100 * (reg < pv).mean(), "ours_mean": ours.mean(), "ours_se_run": se_u,
                               "ours_paper_pct": 100 * (ours < pv).mean(), "z": z}
            q = out[f"{e}|{a}"]
            p(f"| {e} | {a} | {pv:.1f} | {q['smpy_mean']:.1f} ± {q['smpy_se_seed']:.1f} [{se_o:.1f}] | {q['smpy_paper_pct']:.1f} | "
              f"{q['ours_mean']:.1f} [{se_u:.1f}] | {q['ours_paper_pct']:.1f} | {q['ours_mean'] - q['smpy_mean']:+.1f} ({z:+.1f}) |")
    return out


# ---------------------------------------------------------------- RC2 single-realisation reading
def density_p(X, pv, tol, rng, n_pseudo=3000):
    X = np.asarray(X, float).reshape(len(X), -1)
    pv = np.asarray(pv, float)
    d0 = np.all(np.abs(X - pv) <= tol, 1).mean()
    idx = rng.choice(len(X), n_pseudo, replace=False)
    ds = np.array([np.all(np.abs(X - X[i]) <= tol, 1).mean() - 1 / len(X) for i in idx])
    return float(d0), float((ds <= d0).mean())


def rc2():
    p("\n## RC2: single-realisation reading (POST-HOC; the pre-declared mean judgments are unchanged)\n")
    p("Median Δ % and its MATCH/CLOSE/DIVERGENT label; 1-D density-rank p: share of our own runs whose ±20 neighbourhood is "
      "no denser than the paper value's (p-floor ≈ 0.03 when the paper value has no run within ±20); skew of our per-run "
      "distribution.\n")
    p("| exp | arm | paper (vector) | mean Δ% (label) | median | median Δ% (label) | skew | 1-D density-rank p |")
    p("|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    out = {"cells": {}, "panels": {}}
    F = {}
    for e in ("e1", "e2"):
        ro = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", f"{e}.yaml")))["readoff"]["final_regret"]
        for a in ("ucb1", "exp3s", "ducb", "ducb_padB", "swucb"):
            x = arm(os.path.join(G3, f"padb_{e}" if a == "ducb_padB" else e), a)["reps"]
            F[(e, a)] = x
            pv = ro["ducb" if a == "ducb_padB" else a]
            m, med = x.mean(), np.median(x)
            sk = float(((x - m) ** 3).mean() / x.std() ** 3)
            _, pd = density_p(x, [pv], 20, rng)
            out["cells"][f"{e}|{a}"] = {"paper": pv, "rel_mean": (m - pv) / pv, "median": med, "rel_median": (med - pv) / pv, "skew": sk, "p1d": pd}
            p(f"| {e} | {a} | {pv:.1f} | {100 * (m - pv) / pv:+.1f} ({judge((m - pv) / pv)}) | {med:.1f} | {100 * (med - pv) / pv:+.1f} ({judge((med - pv) / pv)}) | "
              f"{sk:+.2f} | {pd:.3f} |")
    p("\nJoint CRN typicality of each whole Figure-1 panel (the paper's 4 final regrets as ONE CRN draw of our model; box ±20 per "
      "arm; 3000 pseudo-papers drawn from our runs):\n")
    p("| panel | arms | joint density-rank p |")
    p("|---|---|---|")
    ro1 = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "e1.yaml")))["readoff"]["final_regret"]
    ro2 = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "e2.yaml")))["readoff"]["final_regret"]
    s7 = arm(os.path.join(G3, "abl2_e2"), "exp3s_S7")["reps"]
    s6 = arm(os.path.join(G3, "abl_e2"), "exp3s_S6")["reps"]
    panels = [
        ("E1, printed arms (headline)", ["ucb1", "exp3s", "ducb", "swucb"], [F[("e1", a)] for a in ("ucb1", "exp3s", "ducb", "swucb")], ro1),
        ("E1, D-UCB padding B", ["ucb1", "exp3s", "ducb_padB", "swucb"], [F[("e1", a)] for a in ("ucb1", "exp3s", "ducb_padB", "swucb")], ro1),
        ("E1, without D-UCB", ["ucb1", "exp3s", "swucb"], [F[("e1", a)] for a in ("ucb1", "exp3s", "swucb")], ro1),
        ("E2, printed arms (headline, S = 2)", ["ucb1", "exp3s", "ducb", "swucb"], [F[("e2", a)] for a in ("ucb1", "exp3s", "ducb", "swucb")], ro2),
        ("E2, padding B, S = 2", ["ucb1", "exp3s", "ducb_padB", "swucb"], [F[("e2", a)] for a in ("ucb1", "exp3s", "ducb_padB", "swucb")], ro2),
        ("E2, padding B, S = 6 (descriptive, descriptive)", ["ucb1", "exp3s_S6", "ducb_padB", "swucb"], [F[("e2", "ucb1")], s6, F[("e2", "ducb_padB")], F[("e2", "swucb")]], ro2),
        ("E2, padding B, S = 7 (descriptive, descriptive)", ["ucb1", "exp3s_S7", "ducb_padB", "swucb"], [F[("e2", "ucb1")], s7, F[("e2", "ducb_padB")], F[("e2", "swucb")]], ro2),
    ]
    for lab, names, cols, ro in panels:
        pv = [ro[n.split("_")[0] if n.startswith(("ducb", "exp3s")) else n] for n in names]
        _, pj = density_p(np.stack(cols, 1), pv, 20, rng)
        out["panels"][lab] = pj
        p(f"| {lab} | {', '.join(names)} | {pj:.3f} |")
    return out


# ---------------------------------------------------------------- RC3 decoupled-eta EXP3.S
def rc3():
    p("\n## RC3: EXP3.S decoupled-η (EXTENSION, labelled; selection seeds 0–4 × 1000, evaluation seeds 20–24 × 2000)\n")
    out = {}
    # (i) bit-identity of eta_mult = 1 on the GPU against the stored Full-scale EXP3.S runs, and against sweepj cells
    for e in ("e1", "e2"):
        d = os.path.join(SA, f"identity_{e}")
        if os.path.isdir(d):
            new = arm(d, "exp3s")["reps"]
            old = np.asarray(recs(os.path.join(G3, e))[0]["policies"]["exp3s"]["final_regret"], float)
            out[f"identity_{e}"] = bool(np.array_equal(new, old))
            p(f"- eta_mult default vs stored Full-scale {e} EXP3.S (seed 0 × 2000, GPU): bit-identical = {out[f'identity_{e}']}")
    sel = json.load(open(os.path.join(SA, "sweepd_selection.json"))) if os.path.exists(os.path.join(SA, "sweepd_selection.json")) else {}
    for e, K in (("e1", 3), ("e2", 2)):
        d = os.path.join(SA, f"sweepd_{e}")
        if not os.path.isdir(d):
            continue
        R = recs(d)
        J = recs(os.path.join(G3, f"sweepj_{e}"))
        same = []
        for gm in ("0.5", "1", "2"):
            for a in ("1e-06", "1e-05", "0.0001", "0.001"):
                kd, kj = f"exp3sd_gx{gm}_e1_a{a}", f"exp3s_gx{gm}_a{a}"
                same.append(all(np.array_equal(r["policies"][kd]["final_regret"], q["policies"][kj]["final_regret"]) for r, q in zip(R, J)))
        out[f"sweepd_vs_sweepj_{e}"] = f"{sum(same)}/{len(same)}"
        p(f"- {e}: sweepd eta_mult = 1 cells bit-identical to the matching sweepj cells: {sum(same)}/{len(same)}")
        m = {n: np.mean([r["policies"][n]["final_regret_mean"] for r in R]) for n in R[0]["policies"]}
        top = sorted(m, key=m.get)[:5]
        p(f"- {e} selection (seeds 0–4 × 1000): " + "; ".join(f"{n} {m[n]:.1f}" for n in top)
          + f". Best eta_mult = 1 cell: {min(v for n, v in m.items() if '_e1_' in n):.1f}. Grid edge: {sel.get(e, {}).get('grid_edge')}")
    fam = {}
    for e in ("e1", "e2"):
        dd, dj = os.path.join(SA, f"tunedd_{e}"), os.path.join(G3, f"tunedj_{e}")
        if not os.path.isdir(dd):
            continue
        X = arm(dd, "exp3s_dec")
        Jx, Jd, Js = arm(dj, "exp3s"), arm(dj, "ducb"), arm(dj, "swucb")
        assert X["seeds"] == Jd["seeds"] == [20, 21, 22, 23, 24] and X["fps"] == Jd["fps"], "CRN with tunedj broken"
        out[f"tunedd_{e}"] = {"mean": float(X["seed_means"].mean()), "se": float(X["seed_means"].std(ddof=1) / math.sqrt(5)),
                              "se_run": float(X["reps"].std(ddof=1) / math.sqrt(X["reps"].size)), "params": X["params"],
                              "arm1": float(X["arm1"].mean()), "at": X["at"], "tj_exp3s": float(Jx["seed_means"].mean())}
        q = out[f"tunedd_{e}"]
        p(f"- {e} evaluation (seeds 20–24 × 2000, CRN with tunedj): EXP3.S-dec {q['mean']:.1f} ± {q['se']:.1f} [{q['se_run']:.1f}] "
          f"(params {X['params']}); Tj EXP3.S {q['tj_exp3s']:.1f}; arm-1 pulls {q['arm1']:.0f}; regret at {X['at']}")
        for nm, A in (("ducb", Jd), ("swucb", Js)):
            pc = paired_compare(A["seed_means"], X["seed_means"])
            t = st.ttest_rel(A["seed_means"], X["seed_means"], alternative="less")
            fam[f"tunedd_{e}:{nm}(Tj)<exp3s_dec"] = dict(pc, p_one=float(t.pvalue), t_stat=float(t.statistic))
        pc = paired_compare(Jx["seed_means"], X["seed_means"])
        t2 = st.ttest_rel(Jx["seed_means"], X["seed_means"])
        out[f"tunedd_{e}"]["tj_exp3s_minus_dec"] = dict(pc, p_two=float(t2.pvalue))
        p(f"  - before/after ({e}, seeds 20–24, CRN): Tj EXP3.S (coupled) − EXP3.S-dec = {pc['diff_mean']:+.1f} ± {pc['diff_se']:.1f} "
          f"(boot 95 % [{pc['boot95'][0]:+.1f}, {pc['boot95'][1]:+.1f}], paired t p = {t2.pvalue:.2g}, n = 5 seed means)")
    if fam:
        adj = holm({k: v["p_one"] for k, v in fam.items()})
        p("\n#### Family Td (4 tests, POST-HOC recheck RC3; Holm within family, one-sided, unit = seed mean, n = 5)\n")
        p("| Claim | diff (mean ± SE) | boot 95 % CI | rel. | t | p (1-sided) | p Holm | HOLDS |")
        p("|---|---|---|---|---|---|---|---|")
        for k, v in fam.items():
            v["p_holm"] = adj[k]["p_holm"]
            v["holds"] = bool(v["p_holm"] < 0.05 and v["boot95"][1] < 0)
            p(f"| {k} | {v['diff_mean']:+.1f} ± {v['diff_se']:.1f} | [{v['boot95'][0]:+.1f}, {v['boot95'][1]:+.1f}] | {100 * v['rel_improvement']:+.1f} % | "
              f"{v['t_stat']:.2f} | {v['p_one']:.2g} | {v['p_holm']:.2g} | {'yes' if v['holds'] else 'NO'} |")
        out["family_Td"] = fam
    return out


# ---------------------------------------------------------------- RC4 gap sweep
def ratios(T, U, delta, xi=0.6):
    tau = round(2 * math.sqrt(T * math.log(T) / U))
    mem = 1 / (math.sqrt(U / T) / 4)
    return tau / (4 * xi * math.log(tau) / delta ** 2), mem / (16 * xi * math.log(mem) / delta ** 2)


def rc4():
    p("\n## RC4: X1 grid a on narrower gaps (EXTENSION, post-hoc; seeds 0–4 × 400; band [0.45, 0.70]; 5-point slope T = 10⁴…10⁶)\n")
    p("| instance (min Δ) | policy | ratio at T = 10⁵ | predicted | regret T = 10⁴ … 10⁶ | slope [95 % CI] | local slopes | in band | prediction |")
    p("|---|---|---|---|---|---|---|---|---|")
    out = {}
    Ts = (10_000, 30_000, 100_000, 300_000, 1_000_000)
    for name, delta in (("d0.2", 0.2), ("d0.4", 0.4)):
        base = os.path.join(ROOT, "runs", "x1_gapsweep", name, "a")
        if not os.path.isdir(base):
            continue
        rsw, rd = ratios(100_000, 4, delta)
        for pol, r in (("swucb", rsw), ("ducb", rd)):
            rows = [[q["policies"][pol]["final_regret_mean"] for q in recs(os.path.join(base, f"T{T}_U4"))] for T in Ts]
            if any(len(r) != 5 for r in rows):  # a point still running
                continue
            Y = np.array(rows)
            sl, ci = V.loglog_slope_boot(Ts, Y)
            loc = np.diff(np.log(Y.mean(1))) / np.diff(np.log(Ts))
            inb = 0.45 <= sl <= 0.70
            pred = r >= 1
            out[f"{name}|{pol}"] = {"ratio_1e5": r, "pred_in_band": pred, "slope": sl, "ci": ci, "means": Y.mean(1).tolist(), "local": loc.tolist(), "in_band": inb}
            p(f"| {name} (Δ = {delta}) | {pol} | {r:.2f} | {'in band' if pred else 'out (above)'} | {Y.mean(1)[0]:.0f} … {Y.mean(1)[-1]:.0f} | "
              f"{sl:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {', '.join(f'{v:.2f}' for v in loc)} | {'yes' if inb else 'no'} | {'CONFIRMED' if inb == pred else 'FAILED'} |")
    return out


def main():
    rep = {"rc1": rc1(), "threeway": threeway(), "rc2": rc2(), "rc3": rc3(), "rc4": rc4()}
    os.makedirs(SA, exist_ok=True)
    json.dump(rep, open(os.path.join(SA, "report.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (bool(o) if isinstance(o, np.bool_) else float(o)))
    open(os.path.join(SA, "report_tables.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
