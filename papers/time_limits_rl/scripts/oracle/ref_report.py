"""Summarise the oracle-stage reference runs (authors' library Tonic @0e20c89 on gym MuJoCo-v4, harness/ref_run.py) next
to our small-scale numbers. Metrics are the same definitions on both sides: E4 = mean undiscounted return of training
episodes ending in the last 10 % of env steps (src.eval.final_window_return); E7 = discounted (0.99) eval return
over H = 3000 with no time limit, 5 stochastic episodes, final evaluation. Paired comparisons share the seed.
Usage: uv run python scripts/oracle/ref_report.py  -> runs/oracle/ref_report.json
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

from src.eval import final_window_return  # noqa: E402

REF = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "ref_runs")


def load(exp, arm, config):
    out = {}
    for p in glob.glob(os.path.join(REF, f"{exp}_{arm}_{config}_s*.json")):
        r = json.load(open(p))
        out[r["seed"]] = r
    return dict(sorted(out.items()))


def main():
    rep = {"source": REF, "tonic_commit": "0e20c894ee68278ab68322de61bb2c7204a11d5f"}
    for config in ("tonic", "paper", "paper_vn"):
        ip = {a: load("ip", a, config) for a in ("ppo_g099", "ta_ppo_g099", "ppo_g1", "ta_ppo_g1")}
        e4 = {}
        for a, runs in ip.items():
            v = {s: final_window_return(r, r["steps"]) for s, r in runs.items()}
            # the harness's own last-10 % number must equal ours on the same episodes (metric parity)
            assert all(abs(v[s] - r["last10_return"]) < 1e-9 for s, r in runs.items()), a
            e4[a] = dict(mean_se=mean_se(list(v.values())), per_seed=v, n=len(v))
        for g in ("g099", "g1"):
            A, B = e4[f"ta_ppo_{g}"]["per_seed"], e4[f"ppo_{g}"]["per_seed"]
            seeds = sorted(set(A) & set(B))
            if len(seeds) >= 2:
                e4[f"ta_minus_ppo_{g}"] = paired_compare([A[s] for s in seeds], [B[s] for s in seeds])
        hop = {a: load("hopper", a, config) for a in ("peb_ppo_T300", "ppo_T300", "ppo_T1000")}
        e7 = {}
        for a, runs in hop.items():
            fin = {s: float(np.mean([e["disc"] for e in r["evals"][-1]["episodes"]])) for s, r in runs.items()}
            ln = {s: float(np.mean([e["length"] for e in r["evals"][-1]["episodes"]])) for s, r in runs.items()}
            curve = {}
            for s, r in runs.items():
                for ev in r["evals"]:
                    curve.setdefault(ev["step"], []).append(float(np.mean([e["disc"] for e in ev["episodes"]])))
            e7[a] = dict(final_eval_disc=dict(mean_se=mean_se(list(fin.values())), per_seed=fin),
                         final_eval_len=float(np.mean(list(ln.values()))) if ln else None,
                         curve={k: float(np.mean(v)) for k, v in sorted(curve.items())},
                         train_final_window=mean_se([final_window_return(r, r["steps"]) for r in runs.values()]),
                         wall_s=float(np.mean([r["wall_s"] for r in runs.values()])) if runs else None, n=len(fin))
        for b in ("ppo_T300", "ppo_T1000"):
            A, B = e7["peb_ppo_T300"]["final_eval_disc"]["per_seed"], e7[b]["final_eval_disc"]["per_seed"]
            seeds = sorted(set(A) & set(B))
            if len(seeds) >= 2:
                e7[f"peb_ppo_T300_minus_{b}"] = paired_compare([A[s] for s in seeds], [B[s] for s in seeds])
        rep[config] = dict(e4=e4, e7=e7)
    # OUR JAX PPO on the MJX gym-v4 IP port (runs/oracle/e4_inverted_pendulum_gymv4, same metric + E5 read-out)
    root = os.path.join(REPO, "runs", "oracle", "e4_inverted_pendulum_gymv4")
    if os.path.isdir(root):
        from src import eval as E
        ours, fw = {}, {}
        for arm in sorted(os.listdir(root)):
            recs = E.load_arm(root, arm)
            if not recs:
                continue
            seeds = sorted(recs)
            fw[arm] = {s: E.final_window_return(recs[s]) for s in seeds}  # window on steps run
            rv = np.array([recs[s]["readout_values"] for s in seeds])
            al = np.array([recs[s]["readout_alive"] for s in seeds])
            full = al.all(-1)
            v_at = lambda t: float(np.nanmean(np.where(full, rv[..., t], np.nan))) if full.any() else None  # noqa
            ours[arm] = dict(mean_se=mean_se(list(fw[arm].values())), per_seed=fw[arm], n=len(seeds),
                             readout=dict(frac_full=float(full.mean()), v_t0=v_at(0), v_t500=v_at(500),
                                          v_t999=v_at(rv.shape[-1] - 1)))
        for g in ("g099", "g1"):
            A, B = fw.get(f"ta_ppo_{g}"), fw.get(f"ppo_{g}")
            if A and B:
                ss = sorted(set(A) & set(B))
                ours[f"ta_minus_ppo_{g}"] = paired_compare([A[x] for x in ss], [B[x] for x in ss])
        rep["ours_gymv4_e4"] = ours
        print("== ours (JAX PPO, MJX gym-v4 IP port)")
        for a, v in ours.items():
            if "mean_se" in v:
                print(f"  E4 {a:12s} {v['mean_se'][0]:8.1f} ± {v['mean_se'][1] or 0:6.1f} (n={v['n']}) readout {v['readout']}")
            else:
                print(f"  E4 {a:20s} diff {v['diff_mean']:+.1f} ± {v['diff_se']:.1f} wins {v['wins']}/{v['n']} "
                      f"p={v['t_p']:.3g}")
    os.makedirs(os.path.join(REPO, "runs", "oracle"), exist_ok=True)
    out = os.path.join(REPO, "runs", "oracle", "ref_report.json")
    json.dump(rep, open(out, "w"), indent=1, default=float)
    for config in ("tonic", "paper", "paper_vn"):
        print(f"== Tonic config: {config}")
        for a, v in rep[config]["e4"].items():
            if "mean_se" in v:
                print(f"  E4 {a:12s} {v['mean_se'][0]:8.1f} ± {v['mean_se'][1] or 0:6.1f} (n={v['n']})")
            else:
                print(f"  E4 {a:20s} diff {v['diff_mean']:+.1f} ± {v['diff_se']:.1f} wins {v['wins']}/{v['n']} "
                      f"p={v['t_p']:.3g}")
        for a, v in rep[config]["e7"].items():
            if "final_eval_disc" in v:
                m = v["final_eval_disc"]["mean_se"]
                print(f"  E7 {a:12s} {m[0]:8.1f} ± {m[1] or 0:6.1f} len {v['final_eval_len']} curve {v['curve']}")
            else:
                print(f"  E7 {a:30s} diff {v['diff_mean']:+.1f} ± {v['diff_se']:.1f} wins {v['wins']}/{v['n']} "
                      f"p={v['t_p']:.3g}")


if __name__ == "__main__":
    main()
