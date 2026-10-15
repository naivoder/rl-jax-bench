"""full-scale report: the pre-declared Holm family ( G3-0), paper comparisons, secondary claims, ablations.

Reads runs/toys/toys.json, runs/full/*, runs/oracle/ref_report.json (Tonic reference numbers); writes
runs/full/report.json and prints markdown fragments. Exact computations (E2 horizon ablation, Queue-of-Cars optima)
run on CPU.
Usage: JAX_PLATFORMS=cpu uv run python scripts/full/report.py
"""
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")  # exact solvers in float64 (dtype policy)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402
from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402
from rl_jax_bench.jaxlog import write_json  # noqa: E402

from src import eval as E  # noqa: E402

G3 = os.path.join(REPO, "runs", "full")
G1 = os.path.join(REPO, "runs", "toys")
REF = json.load(open(os.path.join(REPO, "runs", "oracle", "ref_report.json")))
PAPER_SEEDS = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
FRESH_SEEDS = list(range(10000, 30000, 1000))
rep = {}


def fmt(m, s, d=4):
    return f"{m:.{d}g} ± {s:.2g}" if s is not None else f"{m:.{d}g}"


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.ptp(d) == 0:  # zero-variance rule (G3-0): exact two-sided sign test
        n = len(d)
        return dict(diff_mean=float(d.mean()), diff_se=0.0, wins=int((d > 0).sum()), n=n,
                    t_p=float(min(1.0, 2 * 0.5 ** n)) if d[0] != 0 else 1.0, note="zero variance: sign test",
                    boot95=[float(d[0])] * 2)
    return paired_compare(a, b)


def load(exp_glob):
    """{arm: {seed: rec}} merged over directories matching runs/full/<exp_glob>."""
    out = {}
    for root in sorted(glob.glob(os.path.join(G3, exp_glob))):
        for arm in sorted(os.listdir(root)):
            if arm == "ckpt" or not os.path.isdir(os.path.join(root, arm)):
                continue
            out.setdefault(arm, {}).update(E.load_arm(root, arm))
    return out


def pv(p):
    return f"{p['diff_mean']:+.4g} ± {p['diff_se']:.2g} ({p['wins']}/{p['n']}, p = {p['t_p']:.2g})"


# ------------------------------------------------------------------ E2 (toy-stage full-scale data) + exact ablations
def e2():
    g = json.load(open(os.path.join(G1, "toys.json")))["e2"]
    std, ta, peb = g["standard"], g["time_aware"], g["peb"]
    r = dict(P1=paired(peb["value_inf"], std["value_inf"]), P2=paired(ta["value_T3"], std["value_T3"]),
             values=dict(std_T3=mean_se(std["value_T3"]), std_inf=mean_se(std["value_inf"]),
                         ta_T3=mean_se(ta["value_T3"]), peb_inf=mean_se(peb["value_inf"])))
    # exact horizon / bump ablation: the bias of treating the timeout as terminal, in closed form
    import jax.numpy as jnp
    from src import data as D
    from src import exact as X
    hz = {}
    for bump in (-1.0, 0.0):
        tg = D.two_goal(bump)
        ng = ~np.asarray(tg.is_terminal)
        m = lambda V: float(np.asarray(V)[ng].mean())  # noqa: E731
        q_inf = np.asarray(X.infinite_q(tg, 0.99))
        v_peb = m(X.policy_value_infinite(tg, 0.99, X.greedy_onehot(jnp.asarray(q_inf))))
        for T in ((2, 3, 4, 5, 10, 20) if bump == -1.0 else (3,)):
            pb, _ = X.bootstrap_fraction(tg, T)
            q_std = np.asarray(X.infinite_q(tg, 0.99, pb=pb))
            pi_std = X.greedy_onehot(jnp.asarray(q_std))
            q_ta = np.asarray(X.finite_horizon_q(tg, 0.99, T))
            pi_ta = jnp.stack([X.greedy_onehot(jnp.asarray(q_ta[T - t])) for t in range(T)])
            v_std_T = m(X.policy_value_finite(tg, 0.99, jnp.stack([pi_std] * T)))
            v_ta_T = m(X.policy_value_finite(tg, 0.99, pi_ta))
            v_std_inf = m(X.policy_value_infinite(tg, 0.99, pi_std))
            hz[f"bump{bump:g}_T{T}"] = dict(
                T=T, bump=bump, v_ta_T=v_ta_T, v_std_T=v_std_T, ta_gain_T=v_ta_T - v_std_T, v_peb_inf=v_peb,
                v_std_inf=v_std_inf, peb_gain_inf=v_peb - v_std_inf,
                max_abs_q_bias_std_vs_inf=float(np.abs(q_std - q_inf)[ng].max()),
                mean_pb=float(np.asarray(pb)[ng].mean()))
    r["horizon_exact"] = hz
    rep["e2"] = r


# ------------------------------------------------------------------ E3
# Recheck: exact cell values from the PDF's vector fills (runs/recheck/measurement/fig2_vector_cells.json);
# the raster read-off used before (+-0.05) is kept as PAPER_FIG2_RASTER.
PAPER_FIG2 = {0: [0], 3: [1, 0.992, 0.945, 0], 6: [1, 1, 1, 0.995, 0.522, 0.102, 0],
              9: [1, 1, 1, 1, 1, 1, 0.906, 0.098, 0], 14: [1] * 8 + [0.969]}
PAPER_FIG2_RASTER = {0: [0], 3: [1, 1, 0.95, 0], 6: [1, 1, 1, 0.98, 0.5, 0.1, 0], 9: [1, 1, 1, 1, 1, 1, 0.9, 0.1, 0],
                     14: [1] * 8 + [0.97]}


def e3_one(name, sem):
    arms = load(name)
    if not arms:
        return None
    from src import data as D
    from src import exact as X
    outc = D.qoc_outcomes(D.QOC_CRASH_FIRST if sem == "crash_first" else D.QoCParams())
    out, vals = {}, {}
    for arm, recs in arms.items():
        seeds = sorted(recs)
        T = int(recs[seeds[0]]["lane"]["T"])
        g = float(recs[seeds[0]]["lane"]["gamma"])
        pd = {s: np.asarray(recs[s]["per_update"]["p_danger"])[:, :T] for s in seeds}
        vals[arm] = [E.qoc_exact_value(pd[s][-1], g, sem) for s in seeds]
        U = pd[seeds[0]].shape[0]
        idx = sorted(set(np.linspace(0, U - 1, 10).astype(int)))
        curve = {int(u + 1) * 2048: float(np.mean([E.qoc_exact_value(pd[s][u], g, sem) for s in seeds])) for u in idx}
        vstar, popt = X.qoc_optimal(outc, g, T)
        out[arm] = dict(T=T, final=mean_se(vals[arm]), per_seed=vals[arm], curve=curve, v_star=float(vstar),
                        final_table_mean=np.mean([pd[s][-1] for s in seeds], 0).round(3).tolist(),
                        seeds=seeds)
        if T == 15 and arm.startswith("ta"):
            tab = np.mean([pd[s][-1] for s in seeds], 0)
            cells, agree, agree_opt = 0, 0, 0
            for t, row in PAPER_FIG2.items():
                for x, pv_ in enumerate(row):
                    if popt[t, x] == 0.5:  # exact tie in the optimum: excluded (declared)
                        continue
                    cells += 1
                    agree += abs(tab[t, x] - pv_) <= 0.1
            reach = np.array([[x <= t for x in range(9)] for t in range(T)])
            nt = reach & (popt != 0.5)
            agree_opt = float((np.abs(tab - popt) <= 0.1)[nt].mean())
            out[arm]["fig2"] = dict(cells=cells, agree_pm0_1=int(agree), frac=agree / cells,
                                    frac_agree_exact_opt_reachable=agree_opt,
                                    paper_cells_agree_with_exact_opt=float(np.mean(
                                        [abs(popt[t, x] - v) <= 0.1 for t, row in PAPER_FIG2.items()
                                         for x, v in enumerate(row) if popt[t, x] != 0.5])))
    # pairs: ta-like arm vs the matching ppo arm (same T)
    pairs = {}
    for a in arms:
        if a.startswith("ta"):
            T = out[a]["T"]
            b = [p for p in arms if p.startswith("ppo") and out[p]["T"] == T][0]
            pairs[f"{a}_minus_{b}"] = paired(vals[a], vals[b])
    out["pairs"] = pairs
    if sem == "literal":
        for T in sorted({o["T"] for k, o in out.items() if k != "pairs"}):
            bm, _ = X.qoc_best_memoryless(outc, 0.99, T, n_restarts=16)
            out[f"best_memoryless_T{T}"] = float(bm)
    return out


def e3():
    rep["e3"] = {n: e3_one(n, s) for n, s in (("e3_queue_of_cars", "literal"),
                                               ("e3_queue_of_cars_crash_first", "crash_first"),
                                               ("e3_queue_of_cars_horizon", "literal"))}


# ------------------------------------------------------------------ E4 / E5
def readout(recs, seeds):
    from src import exact as X
    rv = np.array([recs[s]["readout_values"] for s in seeds])
    al = np.array([recs[s]["readout_alive"] for s in seeds])
    lane = recs[seeds[0]]["lane"]
    g, T = float(lane["gamma"]), int(lane["T"])
    k = T - np.arange(rv.shape[-1])
    closed_ta = (1 - g ** k) / (1 - g) if g < 1 else k.astype(float)
    full = al.all(-1)
    if not full.any():
        return dict(frac_full=0.0)
    V = np.nanmean(np.where(full[..., None], rv, np.nan), axis=(0, 1))  # mean curve over full episodes
    t = np.arange(len(V))
    slope, icpt, rr, _, _ = st.linregress(closed_ta, V)
    slope_t = st.linregress(t, V).slope
    chain = float(X.chain_fixed_point_closed_form(g, 0.95, T))
    return dict(frac_full=float(full.mean()), v_t0=float(V[0]), v_t500=float(V[500]), v_t999=float(V[-1]),
                closed_ta_t0=float(closed_ta[0]), closed_ta_t500=float(closed_ta[500]), closed_ta_t999=float(closed_ta[-1]),
                fit_vs_closed_ta=dict(slope=float(slope), intercept=float(icpt), r2=float(rr ** 2)),
                slope_per_step=float(slope_t), aliased_chain_fixed_point=chain,
                per_seed_v_t0=[float(np.nanmean(np.where(full[i], rv[i, :, 0], np.nan))) for i in range(len(seeds))])


def e4_one(name, seeds_sel=None):
    arms = load(name)
    if not arms:
        return None
    out, fw = {}, {}
    for arm, recs in arms.items():
        seeds = sorted(recs) if seeds_sel is None else [s for s in seeds_sel if s in recs]
        fw[arm] = {s: E.final_window_return(recs[s]) for s in seeds}  # window on steps run
        out[arm] = dict(final=mean_se(list(fw[arm].values())), n=len(seeds))
        if "readout_values" in recs[seeds[0]]:
            out[arm]["readout"] = readout(recs, seeds)
        out[arm]["wall"] = recs[seeds[0]]["timing"]
    pairs = {}
    for a in arms:
        for b in arms:
            if a == b or not b.startswith("ppo"):
                continue
            if arms[a][next(iter(arms[a]))]["lane"]["gamma"] != arms[b][next(iter(arms[b]))]["lane"]["gamma"]:
                continue
            ss = sorted(set(fw[a]) & set(fw[b]))
            if a.startswith("ta"):
                pairs[f"{a}_minus_{b}"] = paired([fw[a][s] for s in ss], [fw[b][s] for s in ss])
    out["pairs"] = pairs
    out["_fw"] = {a: {str(s): v for s, v in d.items()} for a, d in fw.items()}
    return out


def e4():
    r = {}
    r["port_paper_seeds"] = e4_one("e4_inverted_pendulum_gymv4", PAPER_SEEDS)
    r["port_fresh_seeds"] = e4_one("e4_inverted_pendulum_gymv4", FRESH_SEEDS)
    r["port_all30"] = e4_one("e4_inverted_pendulum_gymv4")
    r["proxy"] = e4_one("e4_inverted_pendulum")
    for v in ("encoding", "taunorm", "advbatch", "annealLR", "ppo1like", "annealLR_g099", "ppo1like_g099"):
        r[f"abl_{v}"] = e4_one(f"e4_ip_gymv4_ablation_{v}")
    # paper §2.4: the TA gap is larger at gamma = 1 than at 0.99 -> per-seed (gain_g1 - gain_g099), all 30 port seeds
    a = r["port_all30"]["_fw"] if r["port_all30"] else None
    if a:
        ss = sorted(a["ppo_g1"], key=int)
        g1 = [a["ta_ppo_g1"][s] - a["ppo_g1"][s] for s in ss]
        g0 = [a["ta_ppo_g099"][s] - a["ppo_g099"][s] for s in ss]
        r["gain_g1_minus_gain_g099"] = paired(g1, g0)
    # one-sided confirmation on fresh seeds (G3-0)
    f = r["port_fresh_seeds"]
    if f and "ta_ppo_g1_minus_ppo_g1" in f["pairs"]:
        p = f["pairs"]["ta_ppo_g1_minus_ppo_g1"]
        t = p["t"]
        r["confirm_one_sided_p_g1"] = float(st.t.sf(t, p["n"] - 1))
    # does each ablation strengthen the no-TA baseline? (ablation ppo_g1 vs default ppo_g1, same seeds)
    base = r["port_paper_seeds"]["_fw"] if r["port_paper_seeds"] else None
    if base:
        cmp_ = {}
        for v in ("taunorm", "advbatch", "annealLR", "ppo1like"):
            a = r.get(f"abl_{v}")
            if not a:
                continue
            for arm in ("ppo_g1", "ta_ppo_g1"):
                ss = sorted(set(a["_fw"][arm]) & set(base[arm]))
                cmp_[f"{v}_{arm}_minus_default"] = paired([a["_fw"][arm][s] for s in ss], [base[arm][s] for s in ss])
        r["ablation_vs_default"] = cmp_
    rep["e4"] = r


# ------------------------------------------------------------------ E7
def e7():
    arms = load("e7_hopper_p*")
    if not arms:
        return
    out, fin, at1e6 = {}, {}, {}
    for arm, recs in arms.items():
        seeds = sorted(recs)
        ev = {s: recs[s]["evals"] for s in seeds}
        fin[arm] = [float(np.mean(ev[s][-1]["disc"])) for s in seeds]
        i6 = int(np.argmin([abs(e["step"] - 1e6) for e in ev[seeds[0]]]))
        at1e6[arm] = [float(np.mean(ev[s][i6]["disc"])) for s in seeds]
        curve = {int(e["step"]): float(np.mean([np.mean(ev[s][i]["disc"]) for s in seeds]))
                 for i, e in enumerate(ev[seeds[0]])}
        last3 = [float(np.mean([np.mean(ev[s][j]["disc"]) for j in (-3, -2, -1)])) for s in seeds]
        out[arm] = dict(final=mean_se(fin[arm]), per_seed=fin[arm], n=len(seeds),
                        final_step=int(ev[seeds[0]][-1]["step"]), at_1e6=mean_se(at1e6[arm]),
                        step_1e6=int(ev[seeds[0]][i6]["step"]), last3_evals=mean_se(last3), curve=curve,
                        final_len=mean_se([float(np.mean(ev[s][-1]["length"])) for s in seeds]),
                        final_undisc=mean_se([float(np.mean(ev[s][-1]["undisc"])) for s in seeds]),
                        train_final_window=mean_se([E.final_window_return(recs[s]) for s in seeds]),
                        timing=recs[seeds[0]]["timing"])
        out[arm]["_last3"] = last3
    pairs = {}
    for b in ("ppo_T300", "ppo_T1000"):
        pairs[f"peb_ppo_T300_minus_{b}"] = paired(fin["peb_ppo_T300"], fin[b])
        pairs[f"peb_ppo_T300_minus_{b}_at1e6"] = paired(at1e6["peb_ppo_T300"], at1e6[b])
        pairs[f"peb_ppo_T300_minus_{b}_last3"] = paired(out["peb_ppo_T300"]["_last3"], out[b]["_last3"])
    pairs["ppo_T1000_minus_ppo_T300"] = paired(fin["ppo_T1000"], fin["ppo_T300"])
    out["pairs"] = pairs
    rep["e7"] = out


# ------------------------------------------------------------------ E9
def e9_one(name):
    root = os.path.join(G3, name)
    if not os.path.isdir(root):
        return None
    out = {}
    for Bd in sorted(os.listdir(root), key=lambda x: int(x[1:])):
        fin, curves = {}, {}
        for arm in ("no_peb", "peb"):
            recs = E.load_arm(os.path.join(root, Bd), arm)
            seeds = sorted(recs)
            r = np.array([recs[s]["bin_return_mean"] for s in seeds])
            k = max(1, int(round(r.shape[1] * E.FINAL_FRAC)))
            fin[arm] = r[:, -k:].mean(1)
            m = r.mean(0)
            hit = np.where(m >= -65.0)[0]
            out.setdefault(Bd, {})[arm] = dict(final=mean_se(fin[arm]), n=len(seeds), curve=m.round(2).tolist(),
                                               first_bin_mean_ge_m65=int(hit[0]) if len(hit) else None,
                                               bin_steps=recs[seeds[0]]["bin_steps"])
        out[Bd]["peb_minus_no_peb"] = paired(fin["peb"], fin["no_peb"])
    return out


def e9():
    rep["e9"] = e9_one("e9_difficult_grid")
    rep["e9_batch_mean"] = e9_one("e9_difficult_grid_batch_mean")


# ------------------------------------------------------------------ Holm families
def family():
    fam = {}
    e2r, e3r, e4r, e7r, e9r = rep.get("e2"), rep.get("e3", {}), rep.get("e4", {}), rep.get("e7"), rep.get("e9")
    fam["P1"] = e2r["P1"]
    fam["P2"] = e2r["P2"]
    if e3r.get("e3_queue_of_cars"):
        fam["P3"] = e3r["e3_queue_of_cars"]["pairs"]["ta_ppo_minus_ppo"]
    if e7r:
        fam["P6"] = e7r["pairs"]["peb_ppo_T300_minus_ppo_T300"]
    if e9r:
        fam["P7"] = e9r["B1000000"]["peb_minus_no_peb"]
        fam["P8"] = e9r["B100"]["peb_minus_no_peb"]
    out = {}
    for label, src in (("A_proxy", "proxy"), ("B_port", "port_paper_seeds")):
        f = dict(fam)
        s = e4r.get(src)
        if s:
            f["P4"] = s["pairs"]["ta_ppo_g1_minus_ppo_g1"]
            f["P5"] = s["pairs"]["ta_ppo_g099_minus_ppo_g099"]
        h = holm({k: v["t_p"] for k, v in f.items()})
        out[label] = {k: dict(diff_mean=f[k]["diff_mean"], diff_se=f[k]["diff_se"], wins=f[k]["wins"], n=f[k]["n"],
                              boot95=f[k].get("boot95"), p_raw=h[k]["p_raw"], p_holm=h[k]["p_holm"],
                              supported=bool(h[k]["p_holm"] < 0.05 and f[k]["diff_mean"] > 0))
                      for k in sorted(f)}
    rep["holm"] = out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    return o


if __name__ == "__main__":
    for fn in (e2, e3, e4, e7, e9, family):
        try:
            fn()
        except Exception as ex:  # partial results while runs are still going
            print(f"[WARN] {fn.__name__} skipped: {type(ex).__name__}: {ex}")
    write_json(os.path.join(G3, "report.json"), strip(rep))
    for fam, d in rep.get("holm", {}).items():
        print(f"\n## Holm family {fam}")
        for k, v in d.items():
            print(f"| {k} | {v['diff_mean']:+.4g} ± {v['diff_se']:.2g} | {v['wins']}/{v['n']} | p {v['p_raw']:.3g} | "
                  f"Holm {v['p_holm']:.3g} | {'SUPPORTED' if v['supported'] else 'not supported'} |")
    print("wrote", os.path.join(G3, "report.json"))
