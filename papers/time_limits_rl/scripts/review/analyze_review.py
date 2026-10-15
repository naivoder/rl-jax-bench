"""Review-stage analysis (rules pre-declared in). Reads runs/review/* and runs/full/*;
writes runs/review/review_report.json and prints the tables.
  ppo1 readings of E4/E5 (rows 19a/19b) vs the ppo2 primary (full-scale port, 30 seeds; same seeds).
  E5 dynamics to 1e6 (row 18): first read-out step at which each critic is within 10 % of its target.
  Hopper TA vs PPO, gamma = 1 (row 20, Brax PROXY).
Usage: JAX_PLATFORMS=cpu uv run python scripts/review/analyze_review.py
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

RV = os.path.join(R, "runs", "review")
G3 = os.path.join(R, "runs", "full")
PAPER = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
FIG = json.load(open(os.path.join(RV, "figure_readoff.json")))
out = {}


def load(pattern):
    arms = {}
    for root in sorted(glob.glob(pattern)):
        for arm in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, arm)) and arm != "ckpt":
                arms.setdefault(arm, {}).update(E.load_arm(root, arm))
    return arms


def pv(p):
    return f"{p['diff_mean']:+.1f} ± {p['diff_se']:.1f} ({p['wins']}/{p['n']}, p = {p['t_p']:.2g})"


def fw_block(arms, seeds=None):
    fw = {a: {s: E.final_window_return(r) for s, r in recs.items() if seeds is None or s in seeds}
          for a, recs in arms.items()}
    lv = {a: mean_se(list(v.values())) for a, v in fw.items()}
    pairs = {}
    for g in ("g1", "g099"):
        a, b = f"ta_ppo_{g}", f"ppo_{g}"
        if a in fw and b in fw:
            ss = sorted(set(fw[a]) & set(fw[b]))
            pairs[g] = paired_compare([fw[a][s] for s in ss], [fw[b][s] for s in ss])
    return dict(levels=lv, pairs=pairs)


def final_readout(arms):
    o = {}
    for a, recs in arms.items():
        seeds = sorted(recs)
        if "readout_values" not in recs[seeds[0]]:
            continue
        rv = np.array([recs[s]["readout_values"] for s in seeds])
        al = np.array([recs[s]["readout_alive"] for s in seeds])
        full = al.all(-1)
        if not full.any():
            o[a] = dict(frac_full=0.0)
            continue
        V = np.nanmean(np.where(full[..., None], rv, np.nan), axis=(0, 1))
        o[a] = dict(frac_full=float(full.mean()), v_t0=float(V[0]), v_t500=float(V[500]), v_t999=float(V[-1]))
    return o


# ------------------------------------------------------------------
def f2():
    base = load(os.path.join(G3, "e4_inverted_pendulum_gymv4"))
    r = dict(ppo2_primary=dict(all30=fw_block(base), paper=fw_block(base, PAPER)))
    for name in ("e4_ip_gymv4_ppo1bundle", "e4_ip_gymv4_ppo1_2018"):
        arms = load(os.path.join(RV, name))
        if not arms or any(len(v) < 30 for v in arms.values()):
            continue
        r[name] = dict(all30=fw_block(arms), paper=fw_block(arms, PAPER), readout=final_readout(arms))
    out["ppo1_vs_ppo2"] = r
    print("\n## ppo2 primary vs ppo1 readings, IP gym-v4 port, last-10 % return on steps run (30 seeds)")
    for k, v in r.items():
        lv = v["all30"]["levels"]
        print(f"| {k} | " + " | ".join(f"{a} {lv[a][0]:.1f} ± {lv[a][1]:.1f}" for a in sorted(lv)) + " |")
        for g, p in v["all30"]["pairs"].items():
            print(f"|   TA − PPO {g}: all30 {pv(p)}; paper seeds {pv(v['paper']['pairs'][g])} |")
        if "readout" in v:
            print("|   E5 read-out:", {a: {kk: round(x, 1) for kk, x in d.items()} for a, d in v["readout"].items()})


# ------------------------------------------------------------------
def target(arm_tm, g, T, t):
    k = T - t
    if arm_tm == "none":
        return np.full(t.shape, float(X.chain_fixed_point_closed_form(g, 0.95, T)))
    return (1 - g ** k) / (1 - g) if g < 1 else k.astype(float)


def f3():
    r = {}
    for gname, g in (("g099", 0.99), ("g1", 1.0)):
        arms = load(os.path.join(RV, f"e5_ip_dynamics_gymv4_{gname}"))
        if not arms:
            continue
        scale = 100.0 if g < 1 else 1000.0
        res, first = {}, {}
        for a, recs in arms.items():
            seeds = sorted(recs)
            tm = recs[seeds[0]]["lane"]["time_mode"]
            T = int(recs[seeds[0]]["lane"]["T"])
            steps = [x["step"] for x in recs[seeds[0]]["readouts"]]
            err = np.full((len(seeds), len(steps)), np.nan)
            for i, s in enumerate(seeds):
                for j, ro in enumerate(recs[s]["readouts"]):
                    t = np.asarray(ro["t"])
                    v = np.asarray(ro["v"])                     # (episodes, len(t))
                    # intermediate read-outs store only the per-episode full-length flag (not per-step alive), so
                    # the error is taken on FULL-LENGTH episodes; none full -> not learned at that read-out (NaN).
                    # Deviation from's "steps alive", recorded in  before any row-18 result was read.
                    full = np.asarray(ro["full"], bool)
                    if full.any():
                        err[i, j] = float(np.abs(v[full] - target(tm, g, T, t)[None]).mean())
            hit = np.nan_to_num(err, nan=np.inf) <= 0.1 * scale
            fh = [steps[int(np.argmax(h))] if h.any() else np.inf for h in hit]
            # EXPLORATORY (not pre-declared; added after the declared 10 % result was read): stricter thresholds
            expl = {}
            for thr in (0.05, 0.02):
                hh = np.nan_to_num(err, nan=np.inf) <= thr * scale
                expl[str(thr)] = [steps[int(np.argmax(h))] if h.any() else np.inf for h in hh]
            first[a] = dict(zip(seeds, fh))
            res[a] = dict(steps=steps, err_mean=np.nanmean(err, 0).round(2).tolist(),
                          first_hit_per_seed=fh, first_hit_median=float(np.median(fh)),
                          exploratory_first_hit=expl, err_per_seed_final=err[:, -1].round(2).tolist(),
                          n_full_final=int(sum(np.asarray(recs[s]["readouts"][-1]["full"]).sum() for s in seeds)),
                          final_return=mean_se([E.final_window_return(recs[s]) for s in seeds]))
        ta, pp = f"ta_ppo_{gname}", f"ppo_{gname}"
        ss = sorted(first[ta])
        d = np.array([np.sign(first[pp][s] - first[ta][s]) for s in ss])   # +1: TA learned first
        n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
        p_sign = float(st.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue) if n_pos + n_neg else 1.0
        res["ta_first"] = dict(ta_earlier=n_pos, std_earlier=n_neg, ties=int((d == 0).sum()), sign_p=p_sign,
                               reproduced=bool(n_pos > len(ss) / 2 and p_sign < 0.05))
        ex = {}
        for thr in ("0.05", "0.02"):
            a_ = dict(zip(sorted(arms[ta]), res[ta]["exploratory_first_hit"][thr]))
            b_ = dict(zip(sorted(arms[pp]), res[pp]["exploratory_first_hit"][thr]))
            dd = np.array([np.sign(b_[s] - a_[s]) if np.isfinite(a_[s]) or np.isfinite(b_[s]) else 0 for s in ss])
            npos, nneg = int((dd > 0).sum()), int((dd < 0).sum())
            ex[thr] = dict(ta_earlier=npos, std_earlier=nneg, ties=int((dd == 0).sum()),
                           sign_p=float(st.binomtest(npos, npos + nneg, 0.5).pvalue) if npos + nneg else 1.0,
                           ta_median=float(np.median(list(a_.values()))), std_median=float(np.median(list(b_.values()))))
        res["ta_first_exploratory"] = ex
        res["e4_at_1e6_exploratory"] = fw_block(arms)   # by-product: E4 metric at 10x the Figure-3 IP budget
        # final shape on full-length episodes (full-scale statistics)
        res["final_shape"] = {}
        for a, recs in arms.items():
            seeds = sorted(recs)
            rv = np.array([recs[s]["readout_values"] for s in seeds])
            al = np.array([recs[s]["readout_alive"] for s in seeds])
            full = al.all(-1)
            if not full.any():
                continue
            V = np.nanmean(np.where(full[..., None], rv, np.nan), axis=(0, 1))
            k = 1000 - np.arange(len(V))
            ct = (1 - g ** k) / (1 - g) if g < 1 else k.astype(float)
            lr = st.linregress(ct, V)
            res["final_shape"][a] = dict(v_t0=float(V[0]), v_t500=float(V[500]), v_t999=float(V[-1]),
                                         slope_vs_closed_ta=float(lr.slope), r2=float(lr.rvalue ** 2),
                                         frac_full=float(full.mean()))
        r[gname] = res
    out["e5_dynamics"] = r
    print("\n## E5 critic error vs its target over training (1e6 steps, 10 paper seeds)")
    for gname, res in r.items():
        for a in sorted(k for k in res if k not in ("ta_first", "final_shape", "ta_first_exploratory",
                                                    "e4_at_1e6_exploratory")):
            x = res[a]
            print(f"| {gname} {a} | first hit median {x['first_hit_median']:.0f} | per seed {x['first_hit_per_seed']} |")
            print(f"|   err by step: {dict(zip(x['steps'], x['err_mean']))} |")
        print(f"| {gname} TA learned first: {res['ta_first']} |")
        print(f"| {gname} EXPLORATORY stricter thresholds: {res['ta_first_exploratory']} |")
        print(f"| {gname} EXPLORATORY E4 metric at 1e6: TA − PPO {pv(res['e4_at_1e6_exploratory']['pairs'][gname])} |")
        print(f"| {gname} final 10 % training return (1e6 steps): "
              f"{ {a: [round(x, 1) for x in res[a]['final_return']] for a in res if a.startswith(('ta_ppo', 'ppo'))} } |")
        print(f"| {gname} final shape: {res['final_shape']} |")


# ------------------------------------------------------------------
def f5():
    arms = load(os.path.join(RV, "e4_hopper_ta_proxy_p*"))
    if not arms or any(len(v) < 10 for v in arms.values()):
        return
    b = fw_block(arms)
    paper = FIG["figure3"]["Hopper|g1"]["curves"]
    b["paper_last10"] = {k: paper[k]["last10_mean"] for k in ("TA-PPO", "PPO")}
    b["curves"] = {}
    for a, recs in arms.items():
        seeds = sorted(recs)
        ret = []
        for s in seeds:
            ep = np.asarray(recs[s]["episodes"])
            edges = np.arange(0, E.steps_run(recs[s]) + 1, 2e5)
            ret.append([float(ep[(ep[:, 0] > lo) & (ep[:, 0] <= hi), 1].mean()) if ((ep[:, 0] > lo) & (ep[:, 0] <= hi)).any()
                        else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
        b["curves"][a] = np.nanmean(np.array(ret), 0).round(1).tolist()
    out["hopper_ta_gamma1"] = b
    print("\n## Hopper (Brax PROXY), gamma = 1, T = 1000, 2e6 steps, 10 seeds")
    print({a: [round(x, 1) for x in v] for a, v in b["levels"].items()}, "TA − PPO", pv(b["pairs"]["g1"]),
          "paper", b["paper_last10"])
    print("curves (2e5-step bins):", b["curves"])


if __name__ == "__main__":
    for fn in (f2, f3, f5):
        try:
            fn()
        except Exception as ex:
            print(f"[WARN] {fn.__name__} skipped: {type(ex).__name__}: {ex}")
    write_json(os.path.join(RV, "review_report.json"), out)
