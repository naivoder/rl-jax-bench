"""Falsification of 'SR(Cov)/SR(Rnd) = 0.38 is explained by gamma': at the gamma that gives Cov/Rnd ~ 0.38, does
Static/Rnd also land near the paper's 25.2/0.13 ~ 194 ? (CPU, eval scenarios, seed-0 action keys, as full_baselines)"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import src.common as C
import jax, numpy as np
from dataclasses import replace
from src.configs.config import Config, K_EVAL_SCEN
from src import baselines as B, eval as E
from full_baselines import run
ca = Config(); cg = replace(ca.env, gospa_metrics=False)
sk = E.scenario_keys(jax.random.PRNGKey(ca.seed_root), K_EVAL_SCEN, ca.n_eval_scenarios)
for gam in map(float, sys.argv[1:]):
    c2 = replace(cg, gamma_sv=gam, n_sv=100)
    row = {n: float(np.mean(run(*mk(c2), sk, c2, 0)["sr"])) for n, mk in (("static", B.make_static), ("random", B.make_random), ("coverage", B.make_coverage))}
    print(json.dumps(dict(gamma=gam, **row, cov_rnd=row["coverage"]/row["random"], static_rnd=row["static"]/row["random"])), flush=True)
