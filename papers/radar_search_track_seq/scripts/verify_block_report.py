"""Standing-rule-1 verification block report (README), CPU only.

Cells (arm x seeds 0-2, common random numbers with their control): MHSA at lr 1e-4 / 3e-4 / 1e-3, MHSA literal sign,
MHSA unmasked; Flat at lr 1e-4 / 3e-4 / 1e-3, Flat with the single-scan SSV normaliser. Primary readout: the selected
model, stochastic; co-reported: selected deterministic ( applies) and final parameters stochastic. Ratios to the
stored full-scale Random seeds 0-2 (paired by seed index). Applies the declared GO / KILL rules and the attribution rule.

    python scripts/verify_block_report.py      # -> runs/verify_block/report.json, report.txt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VB = os.path.join(REPO, "runs", "verify_block")
G3 = os.path.join(REPO, "runs", "full")
SEEDS = [0, 1, 2]
SR_SINGLE = 43.81423996167378        # ssv_pmax(fifo) / ssv_pmax(single), exact ( test)
REFS, BAND = [0.412, 0.487], (0.351, 0.564)
CLOSE_TOP = 0.487 * 1.2
F4_MIN = 12
CELLS = ["mhsa_lr1e-4", "mhsa", "mhsa_lr1e-3", "mhsa_lit_crn", "mhsa_unmasked", "flat_lr1e-4", "flat", "flat_lr1e-3",
         "flat_single"]
MODES = {"sel_stoch": ("eval_stochastic", "per_episode"), "sel_det": ("eval_deterministic", "per_episode_det"),
         "final_stoch": ("eval_final_stochastic", "per_episode_final")}


FLAT_REFS, FLAT_BAND = [0.767, 0.834], (0.652, 0.955)


def grade(x, flat=False):
    if not np.isfinite(x):
        return "UNDEFINED"
    refs, band = (FLAT_REFS, FLAT_BAND) if flat else (REFS, BAND)
    rel = min(abs(x / r - 1) for r in refs)
    if band[0] <= x <= band[1] or rel <= 0.05:
        return "MATCH"
    return "CLOSE" if rel <= 0.20 else "DIVERGENT"


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v))), per_seed=[float(x) for x in v])


def main():
    base = json.load(open(os.path.join(G3, "baselines.json")))
    keys_sha = base["eval_scenarios"]["keys_sha256"]
    rnd = {k: np.array([base["arms"]["random"][str(s)][k] for s in SEEDS]) for k in ("m_cov", "sr", "gospa_d",
                                                                                      "gospa_miss", "n_init")}
    M = {}
    for c in CELLS:
        fs = [os.path.join(VB, c, f"seed_{s}", "metrics.json") for s in SEEDS]
        if all(os.path.exists(f) for f in fs):
            M[c] = [json.load(open(f)) for f in fs]
            for m in M[c]:
                assert m["eval_scenarios"]["keys_sha256"] == keys_sha, c
    rep = dict(cells={}, checks={}, steps=None)
    for c, ms_ in M.items():
        single = ms_[0]["ssv_norm"] == "single"
        row = dict(best_step=[m["best_step"] for m in ms_], env_steps=ms_[0]["env_steps"],
                   ent_last=[m["train_hist"]["entropy"][-1] for m in ms_], ev_last=[m["train_hist"]["ev"][-1] for m in ms_],
                   cnn_alive_best=[m["cnn_alive"]["best"]["alive"] for m in ms_],
                   cnn_alive_final=[m["cnn_alive"]["final"]["alive"] for m in ms_])
        rep["steps"] = ms_[0]["env_steps"]
        for mode, (ek, _) in MODES.items():
            v = {k: np.array([m[ek][k] for m in ms_], float) for k in ("m_cov", "sr", "gospa_d", "gospa_miss", "n_init",
                                                                        "mean_tracks", "n_episodes_no_track")}
            tb = (100 - v["n_episodes_no_track"]).astype(int).tolist()
            r = v["m_cov"] / rnd["m_cov"]
            srr = v["sr"] / (rnd["sr"] * (SR_SINGLE if single else 1.0))
            undefined = min(tb) <= F4_MIN
            row[mode] = dict(m_cov=ms(v["m_cov"]), m_cov_ratio=ms(r), grade="UNDEFINED (too few track-bearing episodes)" if undefined else grade(r.mean(), flat=c.startswith("flat")),
                             sr_ratio=ms(srr), gospa_d_ratio=ms(v["gospa_d"] / rnd["gospa_d"]),
                             gospa_miss_ratio=ms(v["gospa_miss"] / rnd["gospa_miss"]), n_init=ms(v["n_init"]),
                             mean_tracks=ms(v["mean_tracks"]), track_bearing=tb)
        rep["cells"][c] = row

    def get(c, mode="sel_stoch", k="m_cov_ratio"):
        return np.array(rep["cells"][c][mode][k]["per_seed"])

    def rescue(c, ctl="mhsa"):
        x, y = get(c), get(ctl)
        g = rep["cells"][c]["sel_stoch"]["grade"]
        lower3 = bool(np.all(x < y))
        d = x - y
        p = float(st.ttest_1samp(d, 0.0, alternative="less").pvalue)
        verdict = "GO" if (g in ("MATCH", "CLOSE") and lower3) else ("KILL" if g == "DIVERGENT" else "unresolved")
        return dict(cell=c, ratio=float(x.mean()), grade=g, control_ratio=float(y.mean()), diff=ms(d),
                    paired_p_one_sided=p, lower_in_3_of_3=lower3, verdict=verdict,
                    final_params=dict(ratio=float(get(c, "final_stoch").mean()),
                                      control=float(get(ctl, "final_stoch").mean()),
                                      grade=rep["cells"][c]["final_stoch"]["grade"]))
    C = rep["checks"]
    if all(c in M for c in ("mhsa_lr1e-4", "mhsa", "mhsa_lr1e-3")):
        alts = [rescue("mhsa_lr1e-4"), rescue("mhsa_lr1e-3")]
        grades = [rep["cells"][c]["sel_stoch"]["grade"] for c in ("mhsa_lr1e-4", "mhsa", "mhsa_lr1e-3")]
        v = "GO" if any(a["verdict"] == "GO" for a in alts) else ("KILL" if all(g == "DIVERGENT" for g in grades) else
                                                                     "unresolved")
        desc = {}
        for lr, (mc, fc) in {"1e-4": ("mhsa_lr1e-4", "flat_lr1e-4"), "3e-4": ("mhsa", "flat"),
                             "1e-3": ("mhsa_lr1e-3", "flat_lr1e-3")}.items():
            if fc in M:
                x, y = get(mc), get(fc)
                desc[lr] = dict(mhsa=float(x.mean()), flat=float(y.mean()), mhsa_minus_flat=float(x.mean() - y.mean()),
                                welch_p_two_sided=float(st.ttest_ind(x, y, equal_var=False).pvalue))
        flat_lr = {}
        for fc in ("flat_lr1e-4", "flat_lr1e-3"):
            if fc in M and "flat" in M:
                d = get(fc) - get("flat")
                flat_lr[fc] = dict(diff=ms(d), paired_p_two_sided=float(st.ttest_1samp(d, 0.0).pvalue))
        C["check1_lr"] = dict(verdict=v, mhsa_grades=dict(zip(("1e-4", "3e-4", "1e-3"), grades)), alternatives=alts,
                              mhsa_vs_flat_per_lr=desc, flat_vs_lr3e4=flat_lr)
    if "mhsa_unmasked" in M:
        C["check2_unmasked"] = rescue("mhsa_unmasked")
    if "mhsa_lit_crn" in M:
        c3 = rescue("mhsa_lit_crn")
        li, fl = rep["cells"]["mhsa_lit_crn"]["sel_stoch"], rep["cells"]["mhsa"]["sel_stoch"]
        c3["p_sign"] = dict(n_init_lit=li["n_init"]["mean"], n_init_flip=fl["n_init"]["mean"],
                            miss_ratio_lit=li["gospa_miss_ratio"]["mean"], miss_ratio_flip=fl["gospa_miss_ratio"]["mean"],
                            go=bool(li["n_init"]["mean"] > fl["n_init"]["mean"]
                                    and li["gospa_miss_ratio"]["mean"] > fl["gospa_miss_ratio"]["mean"]))
        C["check3_literal"] = c3
    if "flat_single" in M and "flat" in M:
        a, b = get("flat_single", k="sr_ratio"), get("flat", k="sr_ratio")
        v = ("reward scale discriminates" if (a.mean() <= 0.5 * b.mean() and np.all(a < b)) else
             ("does not" if a.mean() >= b.mean() else "partial"))
        C["D4_single"] = dict(verdict=v, sr_ratio_single=ms(a), sr_ratio_fifo=ms(b),
                              m_cov_ratio_single=rep["cells"]["flat_single"]["sel_stoch"]["m_cov_ratio"],
                              m_cov_ratio_fifo=rep["cells"]["flat"]["sel_stoch"]["m_cov_ratio"],
                              final_sr_ratio_single=rep["cells"]["flat_single"]["final_stoch"]["sr_ratio"]["mean"],
                              final_sr_ratio_fifo=rep["cells"]["flat"]["final_stoch"]["sr_ratio"]["mean"])
    vs = [C.get(k, {}).get("verdict") for k in ("check1_lr", "check2_unmasked", "check3_literal")]
    rep["attribution"] = ("ours (" + ", ".join(k for k, x in zip(("lr", "unmasked", "literal sign"), vs) if x == "GO") + ")"
                          if "GO" in vs else ("paper-side (all three KILL; standing-rule list complete)"
                                              if all(x == "KILL" for x in vs) else "unresolved"))
    rep["verdicts"] = dict(zip(("check1_lr", "check2_unmasked", "check3_literal"), vs))
    json.dump(rep, open(os.path.join(VB, "report.json"), "w"), indent=1, default=float)
    out = [f"steps per lane: {rep['steps']}"]
    out.append("| cell | M_cov/Rnd sel-stoch | grade | final-stoch | sel-det (track-bearing) | SR/Rnd | Σd/Rnd | Σmiss/Rnd | "
               "init | tracks | ent last | best steps | CNN alive best |")
    for c, r in rep["cells"].items():
        s, f, d = r["sel_stoch"], r["final_stoch"], r["sel_det"]
        out.append(f"| {c} | {s['m_cov_ratio']['mean']:.3f} ± {s['m_cov_ratio']['se']:.3f} | {s['grade']} | "
                   f"{f['m_cov_ratio']['mean']:.3f} ± {f['m_cov_ratio']['se']:.3f} ({f['grade']}) | "
                   f"{d['m_cov_ratio']['mean']:.3f} {d['track_bearing']} | {s['sr_ratio']['mean']:.2f} | "
                   f"{s['gospa_d_ratio']['mean']:.3f} | {s['gospa_miss_ratio']['mean']:.3f} | {s['n_init']['mean']:.2f} | "
                   f"{s['mean_tracks']['mean']:.3f} | {np.mean(r['ent_last']):.2f} | {r['best_step']} | "
                   f"{[round(x, 2) for x in r['cnn_alive_best']]} |")
    out.append(json.dumps(rep["checks"], indent=1, default=float))
    out.append("verdicts: " + json.dumps(rep["verdicts"]) + "  attribution: " + rep["attribution"])
    txt = "\n".join(out)
    open(os.path.join(VB, "report.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
