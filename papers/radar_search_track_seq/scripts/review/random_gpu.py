"""Device confound check: Random seed 0 on the 100 eval scenarios on GPU vs the CPU baselines.json row."""
import sys, json, time
t0 = time.time()
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import src.common as C
import jax, numpy as np
from src.configs.config import Config, K_EVAL_SCEN
from src import baselines as B, eval as E
from full_baselines import run
print(jax.devices())
ca = Config(); cfg = ca.env
sk = E.scenario_keys(jax.random.PRNGKey(ca.seed_root), K_EVAL_SCEN, ca.n_eval_scenarios)
out = run(*B.make_random(cfg), sk, cfg, 0)
ref = json.load(open("runs/full/baselines.json"))["arms"]["random"]["0"]["per_episode"]
for k in ("n_init", "n_del", "mean_tracks", "m_cov", "gospa_d", "gospa_miss", "sr"):
    a = np.asarray(out[k], float); b = np.asarray(ref[k], float)
    rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
    print(k, "bit-identical" if np.array_equal(a, b, equal_nan=True) else f"max rel diff {rel:.3g}; episodes differing {(~np.isclose(a,b,rtol=1e-9,equal_nan=True)).sum()}; mean gpu {np.nanmean(a):.6g} cpu {np.nanmean(b):.6g}")
print("wall_s", round(time.time() - t0, 1))
