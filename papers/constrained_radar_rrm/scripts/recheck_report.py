"""The re-evaluation report (review fixes; numpy/scipy only). Reads runs/full_v2/eval (+ runs/full/eval
for the full-run record) and writes runs/full_v2/recheck.json. Sections:
  arms_case1   paper-literal arms (runs/recheck, lane sets ctl/req/lit/n0zero[/eps01]) vs their same-program control 'ctl'
               (paired by seed: same k_lane) and vs OA / DA (same draws); budget satisfaction on executed AND requested g
  arms_case3   per-omega agents vs the full-run mixed-omega agent, OA and OA-blind, on the matching omega
  ctl_vs_full program-shape check: 'ctl' (= full-run declared config, new program) vs full-run A04 lanes
  replica      v1 replica (runs/full/eval) vs re-digitised (runs/full_v2/eval) per method, Case I/II/IV
  oa_knowledge Case III OA informed vs blind; Case IV informed vs blind
  case4_scan   case4 vs case4n / case4p for baselines and the Case I learners (zero-shot)
  tinit_conv   T_init gaps / scans / scans+censored, Case I, baselines and learners
  refused      n_refused per case and method (sum over draws)
Usage: python scripts/recheck_report.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_report import Learn, ms, one_sided, seedvals_base  # noqa: E402
from rl_jax_bench.stats import paired_compare  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV2 = os.path.join(REPO, "runs", "full_v2", "eval")
EV1 = os.path.join(REPO, "runs", "full", "eval")
NP, NR, CATA = 5, 20, -1.0e6


def load(p):
    return json.load(open(p))


def B(ev, case, ck="A04"):
    p = os.path.join(ev, f"base_{case}_{ck}.json")
    return load(p) if os.path.exists(p) else None


def L(ev, name):
    p = os.path.join(ev, f"learn_{name}.json")
    return Learn(p) if os.path.exists(p) else None


def brief(x):
    d = ms(x)
    return dict(mean=d["mean"], se=d["se"], per_seed=d["per_seed"], n_catastrophic=int((np.asarray(x) < CATA).sum()))


def cmp(a, b, side="two"):
    pc = paired_compare(a, b)
    p = pc["t_p"] if side == "two" else one_sided(pc, greater=(side == "greater"))
    return dict(diff=pc["diff_mean"], t95=pc["t95"], boot95=pc["boot95"], p=p if np.isfinite(p) else 1.0,
                wins=int((np.asarray(a) > np.asarray(b)).sum()), side=side)


def tinit_conv(m, idx=None, sl="replica"):
    """Seed-level T_init under the three conventions from per-draw sums; m = summary dict, idx = learner row or None
    (baseline: pairing groups)."""
    g = np.asarray(m[sl]["tinit_sum"], float)
    n = np.asarray(m[sl]["tinit_n"], float)
    nv = np.asarray(m[sl]["never"], float)
    cs = np.asarray(m[sl].get("tinit_cens_sum", np.zeros_like(g)), float)
    if idx is None:
        g, n, nv, cs = (x.reshape(NP, NR).sum(1) for x in (g, n, nv, cs))
    else:
        g, n, nv, cs = (x[idx].sum(1) for x in (g, n, nv, cs))
    return dict(gaps=(g / np.maximum(n, 1)).tolist(), scans=((g + n) / np.maximum(n, 1)).tolist(),
                scans_censored=((g + n + cs + nv) / np.maximum(n + nv, 1)).tolist(), never=nv.tolist())


def learner_rows(lr, v):
    idx, seeds = lr.rows(v)
    return [i for _, i in sorted(zip(seeds, idx))]


def main():
    out = {}
    b1 = B(EV2, "case1")
    ua, ud, uo = (seedvals_base(b1[k], "U") for k in ("ea", "da", "oa"))
    bh = ud if ud.mean() >= ua.mean() else ua

    # ------------------------------------------------ arms, Case I (A-04)
    arms = {}
    for algo in ("ddpg", "dql"):
        lr = L(EV2, f"recheck_{algo}_case1_case1")
        g3 = L(EV2, f"{algo}_case1_case1")
        if lr is None:
            continue
        ctl = lr.vals("ctl", "U")
        res = {}
        for v in sorted({x for x, _ in lr.lanes}):
            u = lr.vals(v, "U")
            res[v] = dict(U=brief(u), sat_exec=ms(lr.vals(v, "sat_rate")), sat_req=ms(lr.vals(v, "sat_rate_req")),
                          share=ms(lr.vals(v, "track_share")), g_exec=ms(lr.vals(v, "g_mean")),
                          g_req=ms(lr.vals(v, "g_req_mean")), lam_max=ms(lr.vals(v, "lam_max")),
                          tinit=ms(lr.tinit(v)), G=ms((u - bh) / (uo - bh)),
                          vs_ctl=cmp(u, ctl) if v != "ctl" else None, vs_oa=cmp(u, uo), vs_da=cmp(u, ud),
                          sat_exec_vs_ctl=cmp(lr.vals(v, "sat_rate"), lr.vals("ctl", "sat_rate")) if v != "ctl" else None)
        if g3 is not None:
            u3 = g3.vals("A04", "U")
            res["full_A04_v2eval"] = dict(U=brief(u3), sat_exec=ms(g3.vals("A04", "sat_rate")),
                                           sat_req=ms(g3.vals("A04", "sat_rate_req")),
                                           share=ms(g3.vals("A04", "track_share")))
            out.setdefault("ctl_vs_full", {})[algo] = dict(
                ctl=brief(ctl), full=brief(u3), diff=cmp(ctl, u3),
                max_rel_seed=float(np.max(np.abs(ctl - u3) / np.abs(u3))))
        arms[algo] = res
    out["arms_case1"] = arms

    # ------------------------------------------------ per-omega Case III
    a3 = {}
    for case, own in (("case3f", "w_fast"), ("case3s", "w_slow")):
        bb = B(EV2, case)
        if bb is None:
            continue
        uo3 = seedvals_base(bb["oa"], "U")
        uob = seedvals_base(bb["oa_blind"], "U") if "oa_blind" in bb else None
        for algo in ("ddpg", "dql"):
            lr = L(EV2, f"recheck_{algo}_case3_{case}")
            g3 = L(EV2, f"{algo}_case3_{case}")
            if lr is None:
                continue
            d = {}
            for v in ("w_fast", "w_slow"):
                u = lr.vals(v, "U")
                d[v] = dict(U=brief(u), median=float(np.median(u)), matching=(v == own),
                            sat_exec=ms(lr.vals(v, "sat_rate")), vs_oa=cmp(u, uo3),
                            vs_oa_blind=cmp(u, uob) if uob is not None else None)
            if g3 is not None:
                u3 = g3.vals("A04", "U")
                d["full_mixed_A04"] = dict(U=brief(u3), median=float(np.median(u3)))
                d["own_vs_mixed"] = cmp(lr.vals(own, "U"), u3)
            a3[f"{algo}|{case}"] = d
    out["arms_case3"] = a3

    # ------------------------------------------------ replica effect, baselines and declared learners
    rep = {}
    for case in ("case1", "case2", "case4"):
        for ck in ("A04", "A04a"):
            o, n = B(EV1, case, ck), B(EV2, case, ck)
            if o is None or n is None:
                continue
            for k in o:
                if k in n:
                    rep[f"{case}|{ck}|{k}"] = dict(full=float(np.mean(o[k]["replica"]["U"])),
                                                   v2=float(np.mean(n[k]["replica"]["U"])))
            run = "case2" if case == "case2" else "case1"
            for algo in ("ddpg", "dql"):
                lo, ln = L(EV1, f"{algo}_{run}_{case}"), L(EV2, f"{algo}_{run}_{case}")
                if lo is None or ln is None:
                    continue
                uo_, un_ = lo.vals(ck, "U"), ln.vals(ck, "U")
                rep[f"{case}|{ck}|{algo}[{ck}]"] = dict(full=float(uo_.mean()), v2=float(un_.mean()),
                                                         full_median=float(np.median(uo_)),
                                                         v2_median=float(np.median(un_)),
                                                         cata_full=int((uo_ < CATA).sum()),
                                                         cata_v2=int((un_ < CATA).sum()))
    out["replica"] = rep

    # ------------------------------------------------ OA knowledge
    ok = {}
    for case in ("case3f", "case3s", "case4"):
        for ck in ("A04", "A04a"):
            bb = B(EV2, case, ck)
            if bb is None or "oa_blind" not in bb:
                continue
            ui, ub = seedvals_base(bb["oa"], "U"), seedvals_base(bb["oa_blind"], "U")
            ok[f"{case}|{ck}"] = dict(oa_informed=ms(ui), oa_blind=ms(ub), informed_minus_blind=cmp(ui, ub),
                                      ea=ms(seedvals_base(bb["ea"], "U")), da=ms(seedvals_base(bb["da"], "U")))
            o1 = B(EV1, case, ck)
            if o1 is not None:
                ok[f"{case}|{ck}"]["full_oa"] = float(np.mean(o1["oa"]["replica"]["U"]))
    out["oa_knowledge"] = ok

    # ------------------------------------------------ Case IV scan interference
    c4 = {}
    for ck in ("A04", "A04a"):
        for case in ("case4", "case4n", "case4p"):
            bb = B(EV2, case, ck)
            if bb is None:
                continue
            row = {k: dict(U=ms(seedvals_base(bb[k], "U")), tinit=ms(np.asarray(tinit_conv(bb[k])["gaps"])),
                           nmiss=ms(seedvals_base(bb[k], "nmiss"))) for k in bb}
            for algo in ("ddpg", "dql"):
                lr = L(EV2, f"{algo}_case1_{case}")
                if lr is not None:
                    row[f"{algo}[{ck}]"] = dict(U=brief(lr.vals(ck, "U")), nmiss=ms(lr.vals(ck, "nmiss")))
            c4[f"{case}|{ck}"] = row
    out["case4_scan"] = c4

    # ------------------------------------------------ T_init conventions, Case I
    tc = {}
    for ck in ("A04", "A04a"):
        bb = B(EV2, "case1", ck)
        if bb is None:
            continue
        for k in bb:
            tc[f"{ck}|{k}"] = tinit_conv(bb[k])
        for algo in ("ddpg", "dql"):
            lr = L(EV2, f"{algo}_case1_case1")
            if lr is not None:
                tc[f"{ck}|{algo}[{ck}]"] = tinit_conv(lr.m, np.asarray(learner_rows(lr, ck)))
    out["tinit_conv"] = tc

    # ------------------------------------------------ refused confirmations
    rf = {}
    for p in sorted(glob.glob(os.path.join(EV2, "base_*_A04.json"))):
        case = os.path.basename(p).split("_")[1]
        m = load(p)
        rf[case] = {k: dict(sum=float(np.sum(m[k]["replica"]["n_refused"])), draws=len(m[k]["replica"]["n_refused"]))
                    for k in m}
    for p in sorted(glob.glob(os.path.join(EV2, "learn_*_case1.json")) + glob.glob(os.path.join(EV2, "learn_*_case2.json"))):
        m = load(p)
        rf[os.path.basename(p)[6:-5]] = dict(sum=float(np.sum(m["replica"]["n_refused"])),
                                            draws=int(np.size(m["replica"]["n_refused"])))
    out["refused"] = rf

    with open(os.path.join(REPO, "runs", "full_v2", "recheck.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote runs/full_v2/recheck.json")


if __name__ == "__main__":
    main()
