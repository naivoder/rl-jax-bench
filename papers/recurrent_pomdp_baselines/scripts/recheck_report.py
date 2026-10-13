"""Recheck report (pre-declared in the README): fp32 re-runs vs the small stage TF32 runs,
the authors' logs and the public (official) code on our hardware; both metrics (raw pre-declared primary, and the
paper's smoothed metric src.eval.paper_metric).  -> runs/recheck/rerun/report.json + stdout."""
import glob, itertools, json, os, re, sys
import numpy as np
import csv
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from src import eval as E  # noqa: E402
from src.eval import PAPER_METRIC_ARGS, paper_metric, judge_detail  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
OREPO = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "repo")
OO = f"{R}/recheck/official-oracle"
L = E.AUTHOR_LOGS
OUT = {}


def perm(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = np.r_[x, y]; n = len(x); obs = abs(x.mean() - y.mean()); c = t = 0
    for idx in itertools.combinations(range(len(z)), n):
        m = np.zeros(len(z), bool); m[list(idx)] = True; t += 1
        c += abs(z[m].mean() - z[~m].mean()) >= obs - 1e-12
    return c / t


def read_csv(p):
    """progress.csv -> list of row dicts with float values (NaN for empty cells)."""
    out = []
    for r in csv.DictReader(open(p)):
        out.append({k: (float(v) if v not in (None, "") else float("nan")) for k, v in r.items()})
    return out


def train_rows(p):
    return [r for r in read_csv(p) if not np.isnan(r.get("rl_loss/qf1_loss", float("nan")))]


def welch(x, y):
    return float(stats.ttest_ind(x, y, equal_var=False).pvalue) if len(x) > 1 and len(y) > 1 else float("nan")


def ours(d, v):
    out = {}
    for f in sorted(glob.glob(f"{R}/{d}/{v}/seed_*/metrics.json")):
        m = json.load(open(f)); out[m["seed"]] = m
    return out


def curve(m):
    return np.array(m["eval_env_steps"], float), np.array(m["eval_returns"], float)


def at(c, s):
    x, y = c; return float(y[x == s][0])


def win(c, lo, hi):
    x, y = c; return float(y[(x >= lo) & (x <= hi)].mean())


def author_curves(g):
    return [(c[:, 0], c[:, 1]) for c in E.author_runs(g)]


def public_curves(prefix):
    out = []
    for lf in sorted(glob.glob(f"{OO}/logs/{prefix}_s*.log")):
        m = re.search(r"Logging to (\S+)", open(lf).read())
        p = os.path.join(OREPO, m.group(1), "progress.csv") if m else ""
        if p and os.path.exists(p) and os.path.getsize(p) > 0:
            d = [r for r in read_csv(p) if not np.isnan(r["metrics/return_eval_total"])]
            out.append((np.array([r["z/env_steps"] for r in d]), np.array([r["metrics/return_eval_total"] for r in d])))
    return out


def ms(v):
    v = np.asarray(v, float); return [round(float(v.mean()), 2), round(float(v.std(ddof=1) / np.sqrt(len(v))), 2) if len(v) > 1 else None, len(v)]


def paired(a, b):
    from stats import paired_compare
    pc = paired_compare(np.asarray(a), np.asarray(b))
    p1 = pc["t_p"] / 2 if pc["t"] > 0 else 1 - pc["t_p"] / 2
    return {"diff": round(pc["diff_mean"], 3), "se": round(pc["diff_se"], 3), "t95": [round(v, 2) for v in pc["t95"]],
            "wins": pc["wins"], "n": pc["n"], "p1_one_sided": p1, "per_seed": pc["diff_per_seed"]}


# ----------------  / P5: Semi-Circle td3-lstm-64-or -------------------------------------------------------------
SC = "meta/PointRobotSparse-v0/{a}/len-64/*/*/*/or/*/progress.csv"
ST = [60000, 75000, 90000, 105000, 120000, 135000, 150000]
f32s, t32s = ours("recheck/rerun/S_td3_separate_fp32", "separate"), ours("small/S_td3_separate", "separate")
f32h, t32h = ours("recheck/rerun/S_td3_shared_fp32", "official"), ours("small/S_td3_shared", "official")
A, Ah = author_curves(SC.format(a="td3_lstm")), author_curves(SC.format(a="td3_lstm_shared"))
P = public_curves("sc_td3_sep")
seeds = sorted(set(f32s) & set(t32s))
print(f"=== P5 /  Semi-Circle separate: fp32 seeds {sorted(f32s)}, TF32 seeds {sorted(t32s)}, public n={len(P)}, authors n={len(A)}")
rows = {}
for s in ST:
    r = {"authors": ms([at(c, s) for c in A]), "tf32": ms([at(curve(t32s[k]), s) for k in seeds]),
         "fp32": ms([at(curve(f32s[k]), s) for k in sorted(f32s)])}
    if all(s in c[0] for c in P) and P:
        r["public"] = ms([at(c, s) for c in P])
    rows[s] = r
    print(f"  {s:7d} " + " | ".join(f"{k} {v[0]:7.2f}±{v[1] if v[1] is not None else float('nan'):5.2f} (n{v[2]})" for k, v in r.items()))
OUT["P5_curve"] = rows
f120 = [at(curve(f32s[k]), 120000) for k in sorted(f32s)]
t120 = [at(curve(t32s[k]), 120000) for k in seeds]
p120 = [at(c, 120000) for c in P]
a120 = [at(c, 120000) for c in A]
pts = [60000, 75000, 90000, 105000, 120000]
auc = lambda cs: [np.mean([at(c, s) for s in pts]) for c in cs]
fA = auc([curve(f32s[k]) for k in sorted(f32s)]); tA = auc([curve(t32s[k]) for k in seeds]); pA = auc(P); aA = auc(A)
oo1 = {"fp32_vs_public_120k": {"diff": float(np.mean(f120) - np.mean(p120)), "welch_p": welch(f120, p120), "perm_p": perm(f120, p120),
                               "fp32": f120, "public": p120},
       "tf32_vs_public_120k": {"diff": float(np.mean(t120) - np.mean(p120)), "welch_p": welch(t120, p120), "perm_p": perm(t120, p120)},
       "fp32_vs_public_mean60_120k": {"diff": float(np.mean(fA) - np.mean(pA)), "welch_p": welch(fA, pA), "perm_p": perm(fA, pA)},
       "fp32_vs_authors_120k": {"diff": float(np.mean(f120) - np.mean(a120)), "welch_p": welch(f120, a120), "perm_p": perm(f120, a120)},
       "fp32_vs_authors_mean60_120k": {"diff": float(np.mean(fA) - np.mean(aA)), "perm_p": perm(fA, aA)},
       "tf32_minus_fp32_paired_120_150k": paired([win(curve(t32s[k]), 120000, 150000) for k in seeds],
                                                 [win(curve(f32s[k]), 120000, 150000) for k in seeds])}
OUT["P5_precision_check"] = oo1
for k, v in oo1.items():
    print(f"  {k}: " + json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}))


def levels(name, ours_runs, logs_curves, env, T, lo, hi):
    """raw window (lo..hi) and the paper metric (truncated at T) for ours and the logs, with the judgement rule."""
    o_raw = [win(curve(m), lo, hi) for m in ours_runs]
    l_raw = [win(c, lo, hi) for c in logs_curves]
    o_pm = [paper_metric(*curve(m), **PAPER_METRIC_ARGS[env], max_x=T) for m in ours_runs]
    l_pm = [paper_metric(*c, **PAPER_METRIC_ARGS[env], max_x=T) for c in logs_curves]
    res = {"raw": {"ours": ms(o_raw), "logs": ms(l_raw), "judge": judge_detail(float(np.mean(o_raw)), l_raw), "welch_p": welch(o_raw, l_raw)},
           "paper_metric": {"ours": ms(o_pm), "logs": ms(l_pm), "judge": judge_detail(float(np.mean(o_pm)), l_pm), "welch_p": welch(o_pm, l_pm)},
           "per_seed_raw": [round(v, 2) for v in o_raw], "per_seed_pm": [round(v, 2) for v in o_pm]}
    for mt in ("raw", "paper_metric"):
        j = res[mt]["judge"]
        print(f"  {name:28s} {mt:12s} ours {res[mt]['ours'][0]:9.2f}±{res[mt]['ours'][1]:6.2f}  logs {res[mt]['logs'][0]:9.2f} (sd {j['ref_sd']:.2f})"
              f"  raw {j['raw_rel_pct']:+6.1f}%  gap_sd {j['gap_in_ref_sd']:+5.2f}  -> {j['verdict']}{' (spread only)' if j['match_by_spread_only'] else ''}  Welch p {res[mt]['welch_p']:.3f}")
    OUT[f"levels:{name}"] = res
    return o_raw, o_pm


print("=== P5 levels (120k-150k, truncated at 150k)")
ofs = levels("P5 separate fp32", [f32s[k] for k in sorted(f32s)], A, "semicircle", 150000, 120000, 150000)
ots = levels("P5 separate TF32 (small stage)", [t32s[k] for k in sorted(t32s)], A, "semicircle", 150000, 120000, 150000)
ofh = levels("P5 shared fp32", [f32h[k] for k in sorted(f32h)], Ah, "semicircle", 150000, 120000, 150000) if f32h else None
oth = levels("P5 shared TF32 (small stage)", [t32h[k] for k in sorted(t32h)], Ah, "semicircle", 150000, 120000, 150000)
pooled = levels("P5 separate TF32 8 seeds", [t32s[k] for k in sorted(t32s)] + list(ours("recheck/learner-strength/fresh_semi_separate", "separate").values()),
                A, "semicircle", 150000, 120000, 150000)
if f32h and sorted(f32h) == sorted(f32s):
    OUT["P5_fp32_test"] = {"raw": paired(ofs[0], ofh[0]), "paper_metric": paired(ofs[1], ofh[1])}
    print("  P5 fp32 paired separate - shared:", json.dumps(OUT["P5_fp32_test"]))

# ----------------  / P3 shared TD3 Pendulum-V ------------------------------------------------------------------
PV = "pomdp/Pendulum/V/{a}/len--1/*/*/*/oar/*/progress.csv"
f3h, t3h = ours("recheck/rerun/P_td3_shared_fp32", "official"), ours("small/P_td3_shared", "official")
t3s = ours("small/P_td3_separate", "separate")
if f3h:
    print("=== P3 shared-official (TD3, Pendulum-V, 16k-20k) levels")
    levels("P3 shared fp32", [f3h[k] for k in sorted(f3h)], author_curves(PV.format(a="td3_lstm_shared")), "pendulum", 20000, 16000, 20000)
    levels("P3 shared TF32 (small stage)", [t3h[k] for k in sorted(t3h)], author_curves(PV.format(a="td3_lstm_shared")), "pendulum", 20000, 16000, 20000)
    MAP = {"qf1_loss": "rl_loss/qf1_loss", "off_rnn": "rl_loss/rnn_grad_norm", "off_q": "rl_loss/q_grad_norm"}
    AU = [train_rows(f) for f in sorted(glob.glob(f"{L}/pomdp/Pendulum/V/td3_lstm_shared/len--1/**/oar/*/progress.csv", recursive=True))]
    PU = []
    for lf in sorted(glob.glob(f"{OO}/logs/pv_td3_shared_20k_s*.log")):
        m = re.search(r"Logging to (\S+)", open(lf).read())
        if m and os.path.exists(os.path.join(OREPO, m.group(1), "progress.csv")):
            PU.append(train_rows(os.path.join(OREPO, m.group(1), "progress.csv")))
    od = lambda runs: [[{MAP.get(kk, kk): vv for kk, vv in r.items()} for r in runs[k]["train"]] for k in sorted(runs)]
    F, T = od(f3h), od(t3h)
    diag = {}
    print("===  P3 shared diagnostics (train rows), authors / public / ours TF32 / ours fp32")
    for w in [(0, 1), (6, 26), (26, 50)]:
        for k in MAP.values():
            vs = {nm: np.array([np.mean([r[k] for r in d[w[0]:w[1]]]) for d in Ls if len(d) >= w[1] and k in d[0]])
                  for nm, Ls in [("authors", AU), ("public", PU), ("tf32", T), ("fp32", F)]}
            pf, pt = welch(vs["authors"], vs["fp32"]), welch(vs["authors"], vs["tf32"])
            diag[f"{w}:{k}"] = {nm: ms(v) for nm, v in vs.items() if len(v)} | {"p_auth_vs_fp32": pf, "p_auth_vs_tf32": pt}
            print(f"  rows {str(w):8s} {k:22s} " + " | ".join(f"{nm} {v.mean():8.3g}" for nm, v in vs.items() if len(v)) + f"   p(auth,fp32)={pf:.3f} p(auth,tf32)={pt:.3f}")
    OUT["P3_shared_diag"] = diag
    med = lambda runs, key: np.array([np.median([r[key] for r in runs[k]["train"] if key in r]) for k in sorted(runs)])
    sh_f, sh_t, sep_t = med(f3h, "off_rnn"), med(t3h, "off_rnn"), med(t3s, "off_q_rnn")
    OUT["SGrad_P3_td3"] = {"shared_fp32_median_rnn": sh_f.round(2).tolist(), "shared_tf32_median_rnn": sh_t.round(2).tolist(),
                           "separate_tf32_critic_rnn": sep_t.round(2).tolist(),
                           "ratio_tf32_geomean": float(np.exp(np.mean(np.log(sh_t / sep_t)))),
                           "ratio_fp32shared_over_tf32sep_geomean_MIXED": float(np.exp(np.mean(np.log(sh_f / sep_t))))}
    print("  S-Grad Pendulum TD3:", json.dumps(OUT["SGrad_P3_td3"]))

# ---------------- P1 separate, pooled seeds (descriptive) ------------------------------------------------
print("=== P1 separate 51k, pooled TF32 0-3 + fp32 8-11 (descriptive)")
p1 = list(ours("full/P1_sac_separate", "separate").values()) + list(ours("full_fp32/P1b_sac_separate", "separate").values())
levels("P1 separate pooled 8 seeds", p1, author_curves(PV.format(a="sac_lstm")), "pendulum", 51000, 40800, 51000)
levels("P1 separate fp32 8-11", list(ours("full_fp32/P1b_sac_separate", "separate").values()), author_curves(PV.format(a="sac_lstm")), "pendulum", 51000, 40800, 51000)
json.dump(OUT, open(f"{R}/recheck/rerun/report.json", "w"), indent=1, default=float)
