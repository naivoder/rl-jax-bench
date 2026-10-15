"""Recheck analysis (rules pre-declared in).
Reads runs/recheck/rerun/* (ledger rows 23-30), the recheck probes (runs/recheck/official-oracle J1-J4,
learner-strength vn2018), the official-code oracle runs (Baselines @9fa8e1b: O1, O1x, O2, O5) and the original runs
(runs/full, runs/review); writes runs/recheck/rerun/recheck_report.json and prints the tables.
  E4  IP port: every reading's levels and paired TA - PPO; before/after paired per arm; ours vs Baselines (Welch).
  P4  E4 Brax IP PROXY under ppo2-2018 vs full scale.
  E5  dynamics to 1e6 under ppo2-2018 / ppo1-2018 ( first-hit rule on BOTH Fig. 4 panels; shapes; scaled units).
  E7  Brax hopper PROXY under ppo2-2018, stochastic + deterministic evaluation, vs full scale at the same updates.
  R4  Hopper TA PROXY under ppo2-2018 vs the review's row 20.
Usage: JAX_PLATFORMS=cpu uv run python scripts/recheck/analyze_recheck.py
"""
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R)
import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402
from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402
from rl_jax_bench.jaxlog import write_json  # noqa: E402

from src import eval as E  # noqa: E402
from src import exact as X  # noqa: E402

RUNS = os.path.join(R, "runs")
SA = os.path.join(RUNS, "recheck")
RR = os.path.join(SA, "rerun")
OO = os.path.join(SA, "official-oracle")
FIG = json.load(open(os.path.join(RUNS, "review", "figure_readoff.json")))
ARMS = ("ppo_g1", "ta_ppo_g1", "ppo_g099", "ta_ppo_g099")
out = {}


def load(*roots):
    arms = {}
    for pat in roots:
        for root in sorted(glob.glob(pat)):
            for arm in sorted(os.listdir(root)):
                if os.path.isdir(os.path.join(root, arm)) and arm != "ckpt":
                    arms.setdefault(arm, {}).update(E.load_arm(root, arm))
    return arms


def oracle(*tags):
    """Baselines-2018 harness runs: <tag>/g<g>_ta<0|1>_s<seed>.json -> {arm: {seed: final-window return}}."""
    o = {}
    for tag in tags:
        for p in glob.glob(os.path.join(OO, tag, "g*_ta*_s*.json")):
            r = json.load(open(p))
            g = os.path.basename(p).split("_")[0][1:]
            arm = ("ta_ppo_" if "_ta1_" in p else "ppo_") + ("g1" if g == "1.0" else "g099")
            o.setdefault(arm, {})[int(r["seed"])] = E.final_window_return(r, total=r["steps_run"])
    return o


def fw_of(arms, seeds=None):
    return {a: {s: E.final_window_return(r) for s, r in recs.items() if seeds is None or s in seeds}
            for a, recs in arms.items()}


def ms(v):
    v = np.asarray(list(v), float)
    return [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), int(len(v))]


def pc(a, b):
    ss = sorted(set(a) & set(b))
    return paired_compare([a[s] for s in ss], [b[s] for s in ss])


def pv(p):
    return (f"{p['diff_mean']:+.1f} ± {p['diff_se']:.1f} ({p['wins']}/{p['n']}, p = {p['t_p']:.2g}, "
            f"CI [{p['boot95'][0]:.0f}, {p['boot95'][1]:.0f}])")


PAPER = {}
for g, key in (("g1", "InvertedPendulum|g1"), ("g099", "InvertedPendulum|g0.99")):
    c = FIG["figure3"][key]["curves"]
    PAPER[f"ppo_{g}"] = (c["PPO"]["last10_mean"], c["PPO"]["last10_se"] or 0.0)
    PAPER[f"ta_ppo_{g}"] = (c["TA-PPO"]["last10_mean"], c["TA-PPO"]["last10_se"] or 0.0)


def judge(ours, paper):
    d = (ours - paper) / abs(paper)
    return d, ("MATCH" if abs(d) <= 0.05 else "CLOSE" if abs(d) <= 0.20 else "DIVERGENT")


def gap_vs_paper(p, g):
    """Our paired gap vs the paper's last-3-bin gap: combined-SE z, with the paper gap's SE bounded between
    independent bins (se/sqrt(3) per curve) and fully correlated bins (se per curve) (recheck)."""
    (ta, sta), (pp, spp) = PAPER[f"ta_ppo_{g}"], PAPER[f"ppo_{g}"]
    gp = ta - pp
    lo, hi = np.sqrt((sta ** 2 + spp ** 2) / 3), np.sqrt(sta ** 2 + spp ** 2)
    z = [(p["diff_mean"] - gp) / np.sqrt(p["diff_se"] ** 2 + s ** 2) for s in (hi, lo)]
    return dict(paper_gap=gp, paper_gap_se_bounds=[float(lo), float(hi)], z_range=[float(z[0]), float(z[1])],
                rel=float((p["diff_mean"] - gp) / abs(gp)), paper_in_our_ci=bool(p["boot95"][0] <= gp <= p["boot95"][1]))


# ------------------------------------------------------------------ E4 IP port
def e4():
    S30 = set(range(0, 30000, 1000))
    readings = {
        "ppo2-2019/Tonic-act (original primary, full scale)": fw_of(load(os.path.join(RUNS, "full", "e4_inverted_pendulum_gymv4"))),
        "probe J1: + raw actions": fw_of(load(os.path.join(OO, "J1", "e4_gymv4_J1"))),
        "probe VN2018 only (learner-strength)": fw_of(load(os.path.join(SA, "learner-strength", "e4_vn2018_30s"))),
        "probe J2: raw actions + VN2018": fw_of(load(os.path.join(OO, "J2", "e4_gymv4_J2"))),
        "probe J3: tau remaining z-scored": fw_of(load(os.path.join(OO, "J3", "e4_gymv4_J3"))),
        "probe J4: ppo2-2018 (monkeypatched)": fw_of(load(os.path.join(OO, "J4", "e4_gymv4_J4"))),
        "ppo2-2018 primary (src flags, row 23)": fw_of(load(os.path.join(RR, "e4_ip_gymv4_ppo2_2018"))),
        "ppo2-2018 fresh seeds 30000..59000 (row 24)": fw_of(load(os.path.join(RR, "e4_ip_gymv4_ppo2_2018_fresh"))),
        "ppo1-2018 original (R-2(b))": fw_of(load(os.path.join(RUNS, "review", "e4_ip_gymv4_ppo1_2018"))),
        "ppo1-2018 corrected (row 25)": fw_of(load(os.path.join(RR, "e4_ip_gymv4_ppo1_2018"))),
        "ppo1-2018 + ppo1 numerics (row 34)": fw_of(load(os.path.join(RR, "e4_ip_gymv4_ppo1_2018_numerics"))),
        "ppo1-2018 + ppo1 numerics, fresh seeds (row 35)": fw_of(load(os.path.join(RR, "e4_ip_gymv4_ppo1_2018_numerics_fresh"))),
        "Baselines-2018 O1 (official code, paper-era settings)": oracle("O1"),
        "Baselines-2018 O1x (fresh seeds 30000..59000)": oracle("O1x"),
        "Baselines-2018 O2 (rescale, VN2019, tau inside)": oracle("O2"),
        "Baselines-2018 ppo1 O6 (official code, paper-era ppo1)": oracle("O6"),
    }
    o5 = oracle("O5"); o2 = oracle("O2")
    readings["Baselines-2018 O5 (= our original choices)"] = {a: (o5 if a.startswith("ta") else o2).get(a, {}) for a in ARMS}
    a = readings["ppo2-2018 primary (src flags, row 23)"]; b = readings["ppo2-2018 fresh seeds 30000..59000 (row 24)"]
    if a and b:
        readings["ppo2-2018 pooled 60 seeds (rows 23 + 24)"] = {k: {**a.get(k, {}), **b.get(k, {})} for k in ARMS}
    res = {}
    print("\n## E4 IP port: last-10 % training return (steps run); paired TA − PPO")
    print("paper (last 3 of 30 bins):", {k: [round(v[0], 1), round(v[1], 1)] for k, v in PAPER.items()})
    for name, fw in readings.items():
        fw = {k: v for k, v in fw.items() if v}
        if not fw:
            print(f"- {name}: (no data)")
            continue
        r = dict(levels={k: ms(v.values()) for k, v in fw.items()}, pairs={}, vs_paper={}, gap_vs_paper={})
        for k, (m, s_, n) in r["levels"].items():
            d, j = judge(m, PAPER[k][0])
            r["vs_paper"][k] = dict(rel=d, judgment=j)
        for g in ("g1", "g099"):
            if f"ta_ppo_{g}" in fw and f"ppo_{g}" in fw:
                p = pc(fw[f"ta_ppo_{g}"], fw[f"ppo_{g}"])
                r["pairs"][g] = p
                r["gap_vs_paper"][g] = gap_vs_paper(p, g)
        res[name] = r
        lv = " ".join(f"{k} {m:6.1f}±{s_:4.1f}(n{n},{100*r['vs_paper'][k]['rel']:+.1f}%)" for k, (m, s_, n) in r["levels"].items())
        print(f"- {name}: {lv}")
        for g, p in r["pairs"].items():
            gv = r["gap_vs_paper"][g]
            print(f"    gap {g}: {pv(p)}; paper {gv['paper_gap']:+.1f} (in our CI: {gv['paper_in_our_ci']}; "
                  f"z {gv['z_range'][0]:+.1f}..{gv['z_range'][1]:+.1f})")
    # before / after, paired per arm on the same seeds
    ba = {}
    print("\n## E4 before -> after (same seeds; paired per arm: after − before)")
    for after, before in (("ppo2-2018 primary (src flags, row 23)", "ppo2-2019/Tonic-act (original primary, full scale)"),
                          ("ppo1-2018 corrected (row 25)", "ppo1-2018 original (R-2(b))"),
                          ("ppo2-2018 primary (src flags, row 23)", "probe J4: ppo2-2018 (monkeypatched)")):
        A, B = readings.get(after, {}), readings.get(before, {})
        for k in ARMS:
            if A.get(k) and B.get(k):
                p = pc(A[k], B[k]); ba[f"{after} | {before} | {k}"] = p
                print(f"  {after[:28]:28s} vs {before[:34]:34s} {k:11s} {pv(p)}")
        for g in ("g1", "g099"):
            if g in res.get(after, {}).get("pairs", {}) and g in res.get(before, {}).get("pairs", {}):
                da = res[after]["pairs"][g]["diff_per_seed"]; db = res[before]["pairs"][g]["diff_per_seed"]
                if len(da) == len(db):
                    p = paired_compare(da, db); ba[f"{after} | {before} | gap_{g}"] = p
                    print(f"  {after[:28]:28s} vs {before[:34]:34s} gap_{g:5s} {pv(p)}")
    # ours vs the official code (Welch on arm means)
    wl = []
    print("\n## ours vs Baselines-2018 (Welch, arm means)")
    pairs = [("ppo2-2018 primary (src flags, row 23)", "Baselines-2018 O1 (official code, paper-era settings)", ARMS)]
    if "ppo2-2018 pooled 60 seeds (rows 23 + 24)" in readings:
        pooled_o = {k: {**oracle("O1").get(k, {}), **oracle("O1x").get(k, {})} for k in ARMS}
        readings["Baselines-2018 O1+O1x (pooled seeds)"] = pooled_o
        pairs.append(("ppo2-2018 pooled 60 seeds (rows 23 + 24)", "Baselines-2018 O1+O1x (pooled seeds)", ARMS))
    pairs.append(("probe J4: ppo2-2018 (monkeypatched)", "ppo2-2018 primary (src flags, row 23)", ARMS))
    pairs.append(("ppo1-2018 corrected (row 25)", "Baselines-2018 ppo1 O6 (official code, paper-era ppo1)", ARMS))
    pairs.append(("ppo1-2018 original (R-2(b))", "Baselines-2018 ppo1 O6 (official code, paper-era ppo1)", ARMS))
    pairs.append(("ppo1-2018 + ppo1 numerics (row 34)", "Baselines-2018 ppo1 O6 (official code, paper-era ppo1)", ARMS))
    for x, y, arms in pairs:
        for k in arms:
            u, v = readings.get(x, {}).get(k), readings.get(y, {}).get(k)
            if not u or not v:
                continue
            u_, v_ = np.array(list(u.values())), np.array(list(v.values()))
            t, p = st.ttest_ind(u_, v_, equal_var=False)
            wl.append(dict(ours=x, other=y, arm=k, ours_mean=float(u_.mean()), other_mean=float(v_.mean()),
                           diff=float(u_.mean() - v_.mean()), p=float(p), n=[len(u_), len(v_)]))
            print(f"  {x[:40]:40s} vs {y[:44]:44s} {k:11s} {u_.mean():6.1f} vs {v_.mean():6.1f} diff {u_.mean()-v_.mean():+6.1f} p={p:.2g}")
        for g in ("g1", "g099"):
            px, py = res.get(x, {}).get("pairs", {}).get(g), res.get(y, {}).get("pairs", {}).get(g)
            if px and py:
                t, p = st.ttest_ind(px["diff_per_seed"], py["diff_per_seed"], equal_var=False)
                wl.append(dict(ours=x, other=y, arm=f"gap_{g}", diff=px["diff_mean"] - py["diff_mean"], p=float(p)))
                print(f"  {x[:40]:40s} vs {y[:44]:44s} gap_{g:6s} {px['diff_mean']:+6.1f} vs {py['diff_mean']:+6.1f} "
                      f"diff {px['diff_mean'] - py['diff_mean']:+6.1f} p={p:.2g}")
    out["E4"] = dict(readings=res, before_after=ba, welch=wl, paper=PAPER)


# ------------------------------------------------------------------ E4 Brax IP proxy
def proxy():
    pre = fw_of(load(os.path.join(RUNS, "full", "e4_inverted_pendulum")))
    new = fw_of(load(os.path.join(RR, "e4_inverted_pendulum_ppo2_2018")))
    if not new:
        return
    r = {}
    print("\n## E4 Brax IP PROXY (10 paper seeds)")
    for name, fw in (("original (full scale)", pre), ("ppo2-2018 raw interface (row 26)", new)):
        r[name] = dict(levels={k: ms(v.values()) for k, v in fw.items() if v},
                       pairs={g: pc(fw[f"ta_ppo_{g}"], fw[f"ppo_{g}"]) for g in ("g1", "g099")
                              if fw.get(f"ta_ppo_{g}") and fw.get(f"ppo_{g}")})
        print(f"- {name}: " + " ".join(f"{k} {m:.1f}±{s_:.1f}" for k, (m, s_, n) in r[name]["levels"].items()))
        for g, p in r[name]["pairs"].items():
            print(f"    gap {g}: {pv(p)}")
    out["proxy"] = r


# ------------------------------------------------------------------ E5 dynamics
def target(tm, g, T, t):
    k = T - t
    if tm == "none":
        return np.full(t.shape, float(X.chain_fixed_point_closed_form(g, 0.95, T)))
    return (1 - g ** k) / (1 - g) if g < 1 else k.astype(float)


def e5_one(arms, g):
    scale = 100.0 if g < 1 else 1000.0
    res, first = {}, {}
    for a, recs in arms.items():
        seeds = sorted(recs)
        tm = recs[seeds[0]]["lane"]["time_mode"]; T = int(recs[seeds[0]]["lane"]["T"])
        steps = [x["step"] for x in recs[seeds[0]]["readouts"]]
        us = [x["u"] for x in recs[seeds[0]]["readouts"]]
        err = np.full((len(seeds), len(steps)), np.nan)
        v0 = np.full((len(seeds), len(steps)), np.nan)
        sig = np.full((len(seeds), len(steps)), np.nan)
        for i, s in enumerate(seeds):
            rs = np.asarray(recs[s]["per_update"].get("ret_scale", np.ones(10 ** 4)))
            for j, ro in enumerate(recs[s]["readouts"]):
                t = np.asarray(ro["t"]); v = np.asarray(ro["v"]); full = np.asarray(ro["full"], bool)
                sig[i, j] = float(rs[min(int(ro["u"]), len(rs)) - 1]) if recs[s]["config"].get("reward_scaling", True) else 1.0
                if full.any():
                    err[i, j] = float(np.abs(v[full] - target(tm, g, T, t)[None]).mean())
                    v0[i, j] = float(v[full][:, 0].mean())
        hit = np.nan_to_num(err, nan=np.inf) <= 0.1 * scale
        fh = [steps[int(np.argmax(h))] if h.any() else np.inf for h in hit]
        first[a] = dict(zip(seeds, fh))
        t0 = float(target(tm, g, T, np.array([0]))[0])
        res[a] = dict(steps=steps, updates=us, err_mean=np.nanmean(err, 0).round(2).tolist(), first_hit_per_seed=fh,
                      first_hit_median=float(np.median(fh)), v_t0_mean=np.nanmean(v0, 0).round(2).tolist(),
                      target_t0=t0, sigma_mean=np.nanmean(sig, 0).round(4).tolist(),
                      v_t0_scaled=(np.nanmean(v0 / sig, 0)).round(3).tolist(),
                      target_t0_scaled=(t0 / np.nanmean(sig, 0)).round(3).tolist(),
                      n_seeds_full=[int(np.isfinite(err[:, j]).sum()) for j in range(len(steps))])
    ta, pp = [k for k in arms if k.startswith("ta")][0], [k for k in arms if k.startswith("ppo")][0]
    ss = sorted(first[ta])
    d = np.array([np.sign(first[pp][s] - first[ta][s]) for s in ss])
    npos, nneg = int((d > 0).sum()), int((d < 0).sum())
    p_sign = float(st.binomtest(npos, npos + nneg, 0.5).pvalue) if npos + nneg else 1.0
    res["ta_first"] = dict(ta_earlier=npos, std_earlier=nneg, ties=int((d == 0).sum()), sign_p=p_sign,
                           reproduced=bool(npos > len(ss) / 2 and p_sign < 0.05))
    res["final_shape"] = {}
    for a, recs in arms.items():
        seeds = sorted(recs)
        rv = np.array([recs[s]["readout_values"] for s in seeds]); al = np.array([recs[s]["readout_alive"] for s in seeds])
        full = al.all(-1)
        if not full.any():
            res["final_shape"][a] = dict(frac_full=0.0)
            continue
        V = np.nanmean(np.where(full[..., None], rv, np.nan), axis=(0, 1))
        k = 1000 - np.arange(len(V))
        ct = (1 - g ** k) / (1 - g) if g < 1 else k.astype(float)
        lr = st.linregress(ct, V)
        res["final_shape"][a] = dict(v_t0=float(V[0]), v_t500=float(V[500]), v_t999=float(V[-1]),
                                     slope_vs_closed_ta=float(lr.slope), r2=float(lr.rvalue ** 2), frac_full=float(full.mean()))
    fw = fw_of(arms)
    res["e4_metric_at_end"] = dict(levels={k: ms(v.values()) for k, v in fw.items()}, pair=pc(fw[ta], fw[pp]))
    return res


def e5():
    r = {}
    for name, pat in (("ppo2-2019/Tonic-act (original, row 18)", os.path.join(RUNS, "review", "e5_ip_dynamics_gymv4_{g}")),
                      ("ppo2-2018 (row 27)", os.path.join(RR, "e5_ppo2_2018_{g}")),
                      ("ppo1-2018 corrected (row 28)", os.path.join(RR, "e5_ppo1_2018_{g}")),
                      ("ppo1-2018 + ppo1 numerics (row 36)", os.path.join(RR, "e5_ppo1n_2018_{g}"))):
        for gname, g in (("g099", 0.99), ("g1", 1.0)):
            arms = load(pat.format(g=gname))
            if len(arms) < 2 or any(not v or "readout_values" not in next(iter(v.values())) for v in arms.values()):
                continue
            x = e5_one(arms, g)
            r[f"{name} | {gname}"] = x
            print(f"\n## E5 {name} {gname}: TA first {x['ta_first']}")
            for a in sorted(k for k in x if k not in ("ta_first", "final_shape", "e4_metric_at_end")):
                y = x[a]
                print(f"  {a:12s} first-hit median {y['first_hit_median']:.0f} per seed {y['first_hit_per_seed']}")
                print(f"     err {dict(zip(y['steps'], y['err_mean']))}")
                print(f"     V(t=0) {dict(zip(y['steps'], y['v_t0_mean']))} target {y['target_t0']:.2f}")
                if any(s_ != 1.0 for s_ in y["sigma_mean"]):
                    print(f"     sigma {y['sigma_mean']}\n     V(0)/sigma {y['v_t0_scaled']} target/sigma {y['target_t0_scaled']}")
            print(f"  final shape {json.dumps(x['final_shape'])}")
            print(f"  E4 metric at the end: TA − PPO {pv(x['e4_metric_at_end']['pair'])}")
    out["E5"] = r


# ------------------------------------------------------------------ E7 Hopper PEB proxy
def e7():
    new = load(os.path.join(RR, "e7_hopper_ppo2_2018_p*"))
    pre = load(os.path.join(RUNS, "full", "e7_hopper_p*"))
    if not new or any(len(v) < 10 for v in new.values()):
        return
    r = {}

    def evals(arms, key, us):
        o = {}
        for a, recs in arms.items():
            o[a] = {}
            for s, rec in recs.items():
                ev = {e["u"]: np.mean(e[key]) for e in rec["evals"] if key in e}
                if all(u in ev for u in us):
                    o[a][s] = float(np.mean([ev[u] for u in us]))
        return o
    print("\n## E7 Brax hopper PROXY at ~1e6 steps (u = 490), 10 seeds")
    for label, arms, key in (("original full scale, stochastic", pre, "disc"), ("ppo2-2018 row 29, stochastic", new, "disc"),
                             ("ppo2-2018 row 29, deterministic", new, "det_disc")):
        for wname, us in (("final eval u=490", [490]), ("last three evals u=392,441,490", [392, 441, 490])):
            v = evals(arms, key, us)
            if not all(v.values()):
                continue
            lv = {a: ms(x.values()) for a, x in v.items()}
            p6 = pc(v["peb_ppo_T300"], v["ppo_T300"]); p7 = pc(v["peb_ppo_T300"], v["ppo_T1000"])
            r[f"{label} | {wname}"] = dict(levels=lv, peb_minus_ppo300=p6, peb_minus_ppo1000=p7)
            print(f"- {label} | {wname}: " + " ".join(f"{a} {m:.1f}±{s_:.1f}" for a, (m, s_, n) in lv.items()))
            print(f"    PEB − PPO(300) {pv(p6)}; PEB − PPO(1000) {pv(p7)}")
    out["E7"] = r


# ------------------------------------------------------------------ Hopper TA proxy
def r4():
    new = load(os.path.join(RR, "e4_hopper_ta_ppo2_2018_p*"))
    pre = load(os.path.join(RUNS, "review", "e4_hopper_ta_proxy_p*"))
    if not new or any(len(v) < 10 for v in new.values()):
        return
    r = {}
    print("\n## R-4 Hopper TA PROXY, gamma = 1, 2e6 steps, 10 seeds")
    for label, arms in (("original (row 20)", pre), ("ppo2-2018 (row 30)", new)):
        fw = fw_of(arms)
        r[label] = dict(levels={k: ms(v.values()) for k, v in fw.items()}, pair=pc(fw["ta_ppo_g1"], fw["ppo_g1"]))
        print(f"- {label}: " + " ".join(f"{k} {m:.1f}±{s_:.1f}" for k, (m, s_, n) in r[label]["levels"].items())
              + f"; TA − PPO {pv(r[label]['pair'])}")
    out["R4"] = r


if __name__ == "__main__":
    for fn in (e4, proxy, e5, e7, r4):
        try:
            fn()
        except Exception as ex:  # a missing run is reported, not fatal
            import traceback
            traceback.print_exc()
            print(f"[WARN] {fn.__name__} skipped: {type(ex).__name__}: {ex}")
    write_json(os.path.join(RR, "recheck_report.json"), out)
