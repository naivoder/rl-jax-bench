"""Rerun Random seed 0, Static seed 2, hand rule, ceiling on the eval scenarios (CPU) and diff vs baselines.json."""
import sys, os, json
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import src.common as C
import jax, numpy as np
from src.configs.config import Config, K_EVAL_SCEN
from src import baselines as B, eval as E
from full_baselines import run
cfg_all = Config(); cfg = cfg_all.env
root = jax.random.PRNGKey(cfg_all.seed_root)
sk = E.scenario_keys(root, K_EVAL_SCEN, cfg_all.n_eval_scenarios)
base = json.load(open("runs/full/baselines.json"))
which = sys.argv[1:]
plan = {"random": (B.make_random(cfg), 0), "static": (B.make_static(cfg), 2), "hand": (B.make_hand_rule(cfg, 12, 4500.0), 0),
        "ceiling": (B.make_ceiling(cfg), 0)}
for name in which:
    (pi, po), s = plan[name]
    out = run(pi, po, sk, cfg, s)
    ref = base["arms"][name][str(s)]["per_episode"]
    for k in ("m_cov", "gospa_d", "gospa_miss", "sr", "n_init", "mean_tracks"):
        a = np.asarray(out[k], float); b = np.asarray(ref[k], float)
        same = np.array_equal(a, b, equal_nan=True)
        md = np.nanmax(np.abs(a - b)) if not same else 0.0
        print(name, s, k, "bit-identical" if same else f"DIFF max {md:.3g}", flush=True)
