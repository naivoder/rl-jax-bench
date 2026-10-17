"""Aggregation: per-seed metrics.json and a seed-level summary for a run directory written by train.py.

Metrics, per lane = (experiment, reading, seed) and method:
  rbar_c   per-CPI mean reward over the eval CPIs (Figure 3/4 y-axis)
  J        mean of rbar_c over all eval CPIs;  J_conv  over CPIs 11..70 (when eval_cpis >= 70, else = J)
  sigma_within  SD (ddof=1) of rbar_c over the J_conv CPIs
  coll_pct, missed_pct (N_c = 0 and N_mo > 0), bw_mhz, adapt_pct (Def. 3 over consecutive eval pulses)
  J_seg    per E3 segment (and J_seg_conv excluding the first 3 CPIs of a segment when it has > 3)
  f        gap fraction (J_conv - J_SAA) / (J* - J_SAA), only where the EXACT J* - J_SAA >= 0.02 (Table D1)
Seed is the unit: summaries are mean +- SE over seeds and seed-level 95 % t intervals.
Also: exact_pi_star (the exact solver re-exported for the positive control) and the stream-identity check.

Usage: python src/eval.py runs/small
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as bl  # noqa: E402
import data as D  # noqa: E402
from rl_jax_bench.stats import mean_se, paired_compare, t_interval  # noqa: E402

exact_pi_star = bl.exact_solve
HERE = os.path.dirname(os.path.abspath(__file__))
COLS = dict(r=0, coll=1, miss=2, size=3, adapt=4)


def file_sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def exact_values(exp: str, reading: str) -> dict:
    """Exact stationary J of pi*, SAA, random per eval segment (segment chains of the experiment)."""
    ex = D.experiment(exp)
    R = D.reward_table(reading)
    out = []
    for ch in ex["eval"]:
        sol = bl.exact_solve(ch, R)
        out.append(dict(pistar=bl.stationary_J(ch, R, sol["pi"]), saa=bl.stationary_J(ch, R, bl.saa_policy(ch)),
                        random=bl.stationary_J(ch, R, None)))
    return dict(segments=out, seg_cpis=ex["seg_cpis"], seg_p=ex["seg_p"])


def lane_metrics(o: np.ndarray, seg_cpis: list[int]) -> dict:
    """o: [n_eval_cpis, 8] per-CPI sums for one lane and method."""
    n = o.shape[0]
    rc = o[:, 0] / D.CPI
    skip = 10 if n >= 70 else 0
    conv = rc[skip:]
    pulses = n * D.CPI
    m = dict(rbar_c=rc.tolist(), J=float(rc.mean()), J_conv=float(conv.mean()),
             sigma_within=float(conv.std(ddof=1)) if len(conv) > 1 else 0.0,
             coll_pct=float(100 * o[:, 1].sum() / pulses), missed_pct=float(100 * o[:, 2].sum() / pulses),
             bw_mhz=float(20 * o[:, 3].sum() / pulses), adapt_pct=float(100 * o[:, 4].sum() / (pulses - 1)))
    if len(seg_cpis) > 1:
        ends = np.cumsum(seg_cpis)
        starts = ends - np.array(seg_cpis)
        m["J_seg"] = [float(rc[s:e].mean()) for s, e in zip(starts, ends)]
        m["J_seg_conv"] = [float(rc[s + 3:e].mean()) if e - s > 3 else float(rc[s:e].mean()) for s, e in zip(starts, ends)]
    return m


def load_run(run_dir: str) -> dict:
    lj = json.load(open(os.path.join(run_dir, "lanes.json")))
    cfg, ids = lj["config"], [tuple(x) for x in lj["ids"]]
    ntr, nev = cfg["train_cpis"], cfg["eval_cpis"]
    meth = {}
    fps = {}
    bz = np.load(os.path.join(run_dir, "baselines.npz"))
    for name in ("pistar", "saa", "constb", "random", "pi", "pi_static"):
        if f"{name}__outs" in bz.files:
            meth[name] = bz[f"{name}__outs"]
            fps[name] = bz[f"{name}__fps"]
    frozen = {}
    probe = {}
    for arm in cfg["arms"]:
        p = os.path.join(run_dir, f"arm_{arm}.npz")
        if not os.path.exists(p):
            continue
        a = np.load(p)
        meth[arm] = a["outs"][:, ntr:]
        fps[arm] = a["fps"][:, ntr:]
        probe[arm] = dict(err=a["probe_err"], agree=a["probe_agree"], train_curve=a["outs"][:, :ntr, 0] / D.CPI,
                          td_loss=a["outs"][:, :, 5].sum(-1) / np.maximum(a["outs"][:, :, 7].sum(-1), 1))
        if "frozen_outs" in a.files:
            frozen[arm + "_frozen"] = a["frozen_outs"]
    meta = json.load(open(os.path.join(run_dir, "baselines_meta.json")))
    return dict(cfg=cfg, ids=ids, meth=meth, frozen=frozen, fps=fps, probe=probe, meta=meta, nev=nev)


def evaluate(run_dir: str, write: bool = True) -> dict:
    R = load_run(run_dir)
    cfg, ids, meth = R["cfg"], R["ids"], dict(R["meth"], **R["frozen"])
    data_sha = file_sha(os.path.join(HERE, "data.py"))
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    # stream identity: every method's eval fingerprint equals pi*'s, lane by lane
    ident = {m: bool(np.array_equal(R["fps"][m], R["fps"]["pistar"])) for m in R["fps"]}
    per_lane = {}
    for li, (e, r, s) in enumerate(ids):
        ex = exact_values(e, r)
        seg = list(np.asarray(D.scale_segments(ex["seg_cpis"], R["nev"]) if sum(ex["seg_cpis"]) != R["nev"]
                              else ex["seg_cpis"]))
        lm = {m: lane_metrics(meth[m][li], seg) for m in meth}
        # gap fraction where defined by the EXACT gap (stationary experiments; per segment for E3)
        g = [x["pistar"] - x["saa"] for x in ex["segments"]]
        for m in meth:
            if len(ex["segments"]) == 1:
                if g[0] >= 0.02:
                    lm[m]["f"] = (lm[m]["J_conv"] - lm["saa"]["J_conv"]) / (ex["segments"][0]["pistar"] - ex["segments"][0]["saa"])
            else:
                lm[m]["f_seg"] = [((lm[m]["J_seg_conv"][k] - lm["saa"]["J_seg_conv"][k]) / g[k]) if g[k] >= 0.02 else None
                                  for k in range(len(g))]
        per_lane[(e, r, s)] = dict(metrics=lm, exact=ex,
                                   eval_stream_sha256=R["meta"]["eval_stream_sha256"][li],
                                   pi_train_stream_sha256=R["meta"]["pi_train_stream_sha256"][li],
                                   pi_meta=R["meta"]["pi_meta"][li])
    seeds = sorted({s for _, _, s in ids})
    if write:
        for s in seeds:
            out = dict(seed=s, config_sha256=cfg_sha, env_data_py_sha256=data_sha, primary_reading="eq3",
                       stream_identity_all_methods=ident, lanes={})
            for (e, r, ss), v in per_lane.items():
                if ss == s:
                    out["lanes"][f"{e}/{r}"] = dict(reading=r, primary=(r == "eq3"), **v)
            p = os.path.join(run_dir, f"seed_{s}", "metrics.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(out, f, indent=1)
    # seed-level summary
    summ = {}
    for e in cfg["experiments"]:
        for r in cfg["readings"]:
            key = f"{e}/{r}"
            lanes = [per_lane[(e, r, s)] for s in seeds]
            ex = lanes[0]["exact"]
            row = dict(exact=ex["segments"])
            for m in meth:
                vals = {k: np.array([ln["metrics"][m][k] for ln in lanes])
                        for k in ("J", "J_conv", "sigma_within", "coll_pct", "missed_pct", "bw_mhz", "adapt_pct")}
                d = {k: dict(zip(("mean", "se"), mean_se(v)), t95=t_interval(v)) for k, v in vals.items()}
                if "f" in lanes[0]["metrics"][m]:
                    fv = np.array([ln["metrics"][m]["f"] for ln in lanes])
                    d["f"] = dict(zip(("mean", "se"), mean_se(fv)), t95=t_interval(fv))
                if "J_seg" in lanes[0]["metrics"][m]:
                    js = np.array([ln["metrics"][m]["J_seg"] for ln in lanes])
                    d["J_seg_mean"] = js.mean(0).tolist()
                    d["J_seg_se"] = (js.std(0, ddof=1) / np.sqrt(len(js))).tolist() if len(js) > 1 else None
                    if "f_seg" in lanes[0]["metrics"][m]:
                        fs = [[ln["metrics"][m]["f_seg"][k] for ln in lanes] for k in range(len(js[0]))]
                        d["f_seg_mean"] = [float(np.mean(x)) if x[0] is not None else None for x in fs]
                row[m] = d
            # descriptive paired comparisons on J_conv (not the Holm family; that is full-scale)
            pc = {}
            for a, b in (("ddrqn", "dqn"), ("ddrqn", "pi"), ("dqn", "pi"), ("ddrqn", "saa"), ("dqn", "saa")):
                if a in meth and b in meth and len(seeds) >= 3:
                    pa = [per_lane[(e, r, s)]["metrics"][a]["J_conv"] for s in seeds]
                    pb = [per_lane[(e, r, s)]["metrics"][b]["J_conv"] for s in seeds]
                    c = paired_compare(pa, pb, n_boot=2000)
                    pc[f"{a}-{b}"] = {k: c[k] for k in ("diff_mean", "diff_se", "t95", "t_p", "wins", "n")}
            row["paired_desc"] = pc
            summ[key] = row
    out = dict(run_dir=run_dir, seeds=seeds, config=cfg, stream_identity=ident, summary=summ,
               probe={arm: dict(final_err_mean=float(np.nanmean(v["err"][:, -1])),
                                final_agree_mean=float(np.nanmean(v["agree"][:, -1])))
                      for arm, v in R["probe"].items()})
    if write:
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(out, f, indent=1)
    return out


def print_table(out: dict) -> None:
    for key, row in out["summary"].items():
        ex = row["exact"]
        print(f"== {key}  exact pi*={[round(x['pistar'], 4) for x in ex]} SAA={[round(x['saa'], 4) for x in ex]} "
              f"rand={[round(x['random'], 4) for x in ex]}")
        for m, d in row.items():
            if m in ("exact", "paired_desc"):
                continue
            s = (f"  {m:14s} J={d['J']['mean']:.4f}±{d['J']['se']:.4f}  Jconv={d['J_conv']['mean']:.4f}  "
                 f"sd_in={d['sigma_within']['mean']:.4f}  coll={d['coll_pct']['mean']:.1f} miss={d['missed_pct']['mean']:.1f} "
                 f"bw={d['bw_mhz']['mean']:.1f} adapt={d['adapt_pct']['mean']:.1f}")
            if "f" in d:
                s += f"  f={d['f']['mean']:.2f} t95=[{d['f']['t95'][0]:.2f},{d['f']['t95'][1]:.2f}]"
            if "J_seg_mean" in d:
                s += "  Jseg=" + ",".join(f"{x:.3f}" for x in d["J_seg_mean"])
            print(s)
        for k, v in row["paired_desc"].items():
            print(f"  paired {k}: diff={v['diff_mean']:+.4f}±{v['diff_se']:.4f} t95=[{v['t95'][0]:+.4f},{v['t95'][1]:+.4f}] "
                  f"p={v['t_p']:.3g} wins={v['wins']}/{v['n']}")
    print("stream identity:", out["stream_identity"])
    print("probe:", out["probe"])


if __name__ == "__main__":
    o = evaluate(sys.argv[1])
    print_table(o)
