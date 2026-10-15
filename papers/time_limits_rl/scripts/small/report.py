"""small-scale report: direction + sanity of every arm (descriptive paired statistics; the pre-declared Holm family is
evaluated at full scale). Reads runs/toys/* and runs/small/*; writes runs/small/report.json and prints markdown rows.
Usage: JAX_PLATFORMS=cpu uv run python scripts/small/report.py
"""
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402

from src import eval as E  # noqa: E402
from rl_jax_bench.jaxlog import write_json  # noqa: E402

G2 = os.path.join(REPO, "runs", "small")
G1 = os.path.join(REPO, "runs", "toys")
rep = {}


def fmt(m, s):
    return f"{m:.4g} ± {s:.2g}"


def e2():
    g = json.load(open(os.path.join(G1, "toys.json")))["e2"]
    std, ta, peb = g["standard"], g["time_aware"], g["peb"]
    P1 = E.paired(peb["value_inf"], std["value_inf"])
    P2 = E.paired(ta["value_T3"], std["value_T3"]) if np.ptp(np.subtract(ta["value_T3"], std["value_T3"])) > 0 else \
        dict(diff_mean=float(np.mean(np.subtract(ta["value_T3"], std["value_T3"]))), diff_se=0.0, wins=10, n=10,
             note="zero variance: every seed's difference is identical (deterministic greedy policies)")
    rep["e2"] = dict(P1_peb_minus_standard_inf=P1, P2_ta_minus_standard_T3=P2,
                     values=dict(standard=dict(T3=E.mean_se(std["value_T3"]), inf=E.mean_se(std["value_inf"])),
                                 ta=dict(T3=E.mean_se(ta["value_T3"])), peb=dict(inf=E.mean_se(peb["value_inf"]))))
    print(f"| E2 P1 PEB−std (∞-horizon value) | {fmt(P1['diff_mean'], P1['diff_se'])} | wins {P1['wins']}/{P1['n']} |")
    print(f"| E2 P2 TA−std (T=3 value) | {P2['diff_mean']:.4f} ± {P2['diff_se']:.2g} | wins {P2['wins']}/{P2['n']} |")


def e3(name, sem):
    root = os.path.join(G2, name)
    if not os.path.isdir(root):
        return
    arms = sorted(os.listdir(root))
    vals, curves = {}, {}
    for arm in arms:
        recs = E.load_arm(root, arm)
        if not recs:
            continue
        seeds = sorted(recs)
        pd = {s: np.asarray(recs[s]["per_update"]["p_danger"]) for s in seeds}  # (U, 15, 9)
        vals[arm] = [E.qoc_exact_value(pd[s][-1], 0.99, sem) for s in seeds]
        U = pd[seeds[0]].shape[0]
        idx = sorted(set(np.linspace(0, U - 1, 8).astype(int)))
        curves[arm] = {int(u + 1) * 2048: float(np.mean([E.qoc_exact_value(pd[s][u], 0.99, sem) for s in seeds]))
                       for u in idx}
        if arm == "ta_ppo":
            rep.setdefault(name, {})["ta_final_table_mean"] = np.mean([pd[s][-1] for s in seeds], 0).round(3).tolist()
        if arm == "ppo":
            rep.setdefault(name, {})["ppo_final_rows"] = {s: pd[s][-1][0].round(2).tolist() for s in seeds}
    r = rep.setdefault(name, {})
    r["exact_value_final"] = {a: dict(mean_se=E.mean_se(v), per_seed=v) for a, v in vals.items()}
    r["exact_value_curve"] = curves
    for a in arms:
        if a != "ppo" and a in vals:
            r[f"P3_{a}_minus_ppo"] = E.paired(vals[a], vals["ppo"])
    for a, v in vals.items():
        m, s = E.mean_se(v)
        print(f"| {name} {a} exact value (0,0) | {fmt(m, s)} | n={len(v)} |")
    for a in arms:
        if a != "ppo" and a in vals:
            p = r[f"P3_{a}_minus_ppo"]
            print(f"| {name} {a} − ppo | {fmt(p['diff_mean'], p['diff_se'])} | wins {p['wins']}/{p['n']}, p={p['t_p']:.2g} |")


def e9():
    root = os.path.join(G2, "e9_difficult_grid")
    if not os.path.isdir(root):
        return
    out = {}
    for Bd in sorted(os.listdir(root), key=lambda x: int(x[1:])):
        fin = {}
        for arm in ("no_peb", "peb"):
            recs = E.load_arm(os.path.join(root, Bd), arm)
            seeds = sorted(recs)
            r = np.array([recs[s]["bin_return_mean"] for s in seeds])
            k = max(1, int(round(r.shape[1] * E.FINAL_FRAC)))
            fin[arm] = r[:, -k:].mean(1)
            out.setdefault(Bd, {})[arm] = dict(final=E.mean_se(fin[arm]), curve=r.mean(0).round(1).tolist())
        p = E.paired(fin["peb"], fin["no_peb"])
        out[Bd]["peb_minus_no_peb"] = p
        print(f"| E9 {Bd} final no-PEB / PEB | {fmt(*out[Bd]['no_peb']['final'])} / {fmt(*out[Bd]['peb']['final'])} | "
              f"diff {p['diff_mean']:.1f}, wins {p['wins']}/{p['n']} |")
    rep["e9"] = out


def e4():
    root = os.path.join(G2, "e4_inverted_pendulum")
    if not os.path.isdir(root):
        return
    out = {}
    fw = {}
    for arm in sorted(os.listdir(root)):
        recs = E.load_arm(root, arm)
        if not recs:
            continue
        seeds = sorted(recs)
        fw[arm] = [E.final_window_return(recs[s]) for s in seeds]  # window on steps run
        # E5 value readout
        rv = np.array([recs[s]["readout_values"] for s in seeds])  # (S, ep, T)
        al = np.array([recs[s]["readout_alive"] for s in seeds])
        g = recs[seeds[0]]["lane"]["gamma"]
        T = recs[seeds[0]]["lane"]["T"]
        k = T - np.arange(rv.shape[-1])
        closed_ta = (1 - g ** k) / (1 - g) if g < 1 else k.astype(float)
        full = al.all(-1)  # episodes that never fell
        v_at = lambda t: float(np.nanmean(np.where(full, rv[..., t], np.nan))) if full.any() else float("nan")  # noqa
        out[arm] = dict(final_window=dict(mean_se=E.mean_se(fw[arm]), per_seed=fw[arm]),
                        readout=dict(frac_full_episodes=float(full.mean()), v_t0=v_at(0), v_t500=v_at(500),
                                     v_t999=v_at(rv.shape[-1] - 1), closed_form_ta_t0=float(closed_ta[0]),
                                     closed_form_ta_t500=float(closed_ta[500]),
                                     closed_form_ta_t999=float(closed_ta[-1])))
        print(f"| E4 {arm} final-10% return | {fmt(*E.mean_se(fw[arm]))} | V(t=0,500,999) = "
              f"{v_at(0):.1f}, {v_at(500):.1f}, {v_at(rv.shape[-1] - 1):.1f}; full eps {full.mean():.2f} |")
    for gname in ("g099", "g1"):
        a, b = f"ta_ppo_{gname}", f"ppo_{gname}"
        if a in fw and b in fw:
            p = E.paired(fw[a], fw[b])
            out[f"{a}_minus_{b}"] = p
            print(f"| E4 {a} − {b} | {fmt(p['diff_mean'], p['diff_se'])} | wins {p['wins']}/{p['n']}, p={p['t_p']:.2g} |")
    rep["e4"] = out


def e7():
    root = os.path.join(G2, "e7_hopper")
    if not os.path.isdir(root):
        return
    out, fin = {}, {}
    for arm in sorted(os.listdir(root)):
        recs = E.load_arm(root, arm)
        if not recs:
            continue
        seeds = sorted(recs)
        ev = {s: recs[s]["evals"] for s in seeds}
        fin[arm] = [float(np.mean(ev[s][-1]["disc"])) for s in seeds]
        curve = {e["step"]: float(np.mean([np.mean(ev[s][i]["disc"]) for s in seeds])) for i, e in enumerate(ev[seeds[0]])}
        lens = float(np.mean([np.mean(ev[s][-1]["length"]) for s in seeds]))
        out[arm] = dict(final_eval_disc=dict(mean_se=E.mean_se(fin[arm]), per_seed=fin[arm]), curve=curve,
                        final_eval_len=lens, train_final_window=E.mean_se([E.final_window_return(recs[s])
                                                                           for s in seeds]))
        print(f"| E7 {arm} final eval disc. return | {fmt(*E.mean_se(fin[arm]))} | eval len {lens:.0f}; curve "
              f"{', '.join(f'{v:.0f}' for v in curve.values())} |")
    for a in ("ppo_T300", "ppo_T1000"):
        if "peb_ppo_T300" in fin and a in fin:
            p = E.paired(fin["peb_ppo_T300"], fin[a])
            out[f"peb_ppo_T300_minus_{a}"] = p
            print(f"| E7 peb_ppo_T300 − {a} | {fmt(p['diff_mean'], p['diff_se'])} | wins {p['wins']}/{p['n']} |")
    rep["e7"] = out


if __name__ == "__main__":
    for f in (e2, lambda: e3("e3_queue_of_cars", "literal"), lambda: e3("e3_queue_of_cars_crash_first", "crash_first"),
              e9, e4, e7):
        f()
    write_json(os.path.join(G2, "report.json"), rep)
