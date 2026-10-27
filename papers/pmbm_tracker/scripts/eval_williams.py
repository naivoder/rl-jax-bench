"""OSPA of the OFFICIAL Williams TOMB/MOMB code (run under Octave via scripts/official/drive_williams.m) on the same measurement sets.
Labelled 'Williams official code, untuned settings': the paper §VII gives only the 0.7 existence threshold; the code has no
gating, a 1e-4 r-prune and a 1e-4 Poisson prune (sources.yaml), unlike our PMBM (gate 20, 1e-5 prunes).

Estimate rule (--rule): thr = paper §VII, Bernoullis with r > 0.7 ('est_thr', graded; algos tomb/momb);
                        map = ancillary main.m MAP-cardinality rule ('est_map', sensitivity row; algos tomb_map/momb_map).
Writes / MERGES into runs/<tag>/seed_<r>/metrics.json (never overwrites an existing algo entry unless --force; --tag is required and
must not be empty; config_id / pd / lambda_c are recorded).
usage: python scripts/eval_williams.py <meas.npz> <dump_root containing out_TOMB,out_MOMB> <tag> [--rule thr|map] [--force]"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "official")]
from src import eval as EV
import load_dump as LD
from rl_jax_bench.jaxlog import write_json

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("root"); ap.add_argument("tag")
ap.add_argument("--rule", choices=("thr", "map"), default="thr")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
if not a.tag.strip() or os.path.basename(a.tag.rstrip("/")) != a.tag.rstrip("/"):
    sys.exit("eval_williams: --tag must be a non-empty plain directory name")
f = np.load(a.npz)
truth = dict(X=f["X"], alive=f["alive"])
meta = {k[5:]: f[k][()].item() for k in f.files if k.startswith("meta_")}
R = f["z"].shape[0]
key = "est_thr" if a.rule == "thr" else "est_map"
suffix = "" if a.rule == "thr" else "_map"
for r in range(R):
    p = os.path.join(HERE, "..", "runs", a.tag, f"seed_{r}")
    mp_ = os.path.join(p, "metrics.json")
    rec = json.load(open(mp_)) if os.path.exists(mp_) else dict(run=r, algos={}, diag={})
    for k_ in ("config_id", "pd", "lambda_c"):          # record the cell
        if k_ in meta:
            rec.setdefault(k_, meta[k_])
    for alg in ("TOMB", "MOMB"):
        name = alg.lower() + suffix
        if name in rec["algos"] and not a.force:
            sys.exit(f"eval_williams: {mp_} already holds '{name}'; refusing to overwrite (use --force)")
        d = LD.load_williams_run(os.path.join(a.root, f"out_{alg}", f"wil_{alg}_run_{r:03d}.mat"))
        K = d["K"]; E = 16
        m = np.zeros((K, E), bool); x = np.zeros((K, E, 4)); n_over = 0
        for k, est in enumerate(d[key]):
            n = min(len(est), E); n_over += max(len(est) - E, 0); m[k, :n] = True; x[k, :n] = est[:n]
        if n_over:                                  # the E-slot estimate array would truncate -> zero-counter rule
            sys.exit(f"eval_williams: run {r} {alg}: {n_over} estimates beyond E={E}; refusing to score a truncated set")
        rr = EV.evaluate_run(m, x, truth)
        rec["algos"][name] = {k: v.tolist() for k, v in rr.items()}
        rec["diag"][name] = dict(seconds=float(np.sum(d["t_step"])), est_over=int(n_over), rule=key,
                                 source="Williams official code under Octave, untuned settings (no gating, r-prune 1e-4)")
    os.makedirs(p, exist_ok=True)
    write_json(mp_, rec)
print("wrote", R, "runs to runs/" + a.tag, "rule", a.rule)
