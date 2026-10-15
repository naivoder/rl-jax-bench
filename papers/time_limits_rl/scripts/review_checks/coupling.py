""": does a lane's trajectory depend on the OTHER lanes in the vmapped batch beyond float round-off?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import train as T
import numpy as np
SC = os.path.join("runs", "review_checks", "coupling")
base = T.load_config("src/configs/e4_inverted_pendulum_gymv4.yaml")
base.update(total_timesteps=2048 * 4, chunk_updates=4, readout_T=None)
comps = {
    "alone": ({"ppo_g1": {"time_mode": "none", "T": 1000, "gamma": 1.0}}, [0]),
    "alone_again": ({"ppo_g1": {"time_mode": "none", "T": 1000, "gamma": 1.0}}, [0]),
    "pair": ({"ppo_g1": {"time_mode": "none", "T": 1000, "gamma": 1.0}, "ta_ppo_g1": {"time_mode": "elapsed_pm1", "T": 1000, "gamma": 1.0}}, [0]),
    "big": ({"ppo_g1": {"time_mode": "none", "T": 1000, "gamma": 1.0}, "ta_ppo_g1": {"time_mode": "elapsed_pm1", "T": 1000, "gamma": 1.0},
             "ppo_g099": {"time_mode": "none", "T": 1000, "gamma": 0.99}, "ta_ppo_g099": {"time_mode": "elapsed_pm1", "T": 1000, "gamma": 0.99}}, [0, 1000, 2000, 3000, 4000]),
}
res = {}
for k, (arms, seeds) in comps.items():
    cfg = dict(base, arms=arms, seeds=seeds, name=f"coup_{k}")
    out = T.train_ppo(cfg, os.path.join(SC, k), ckpt=False)
    rec = [r for r in out if r["arm"] == "ppo_g1" and r["seed"] == 0][0]
    res[k] = {m: np.asarray(rec["per_update"][m], np.float64) for m in ("loss", "explained_variance", "ret_scale", "ep_ret_sum")}
    ta = [r for r in out if r["arm"] == "ta_ppo_g1" and r["seed"] == 0]
    if ta: res[k + "_ta"] = {m: np.asarray(ta[0]["per_update"][m], np.float64) for m in ("loss", "explained_variance")}
for k in res:
    print(k, {m: np.array2string(v, precision=9) for m, v in res[k].items()})
for k in ("alone_again", "pair", "big"):
    print(f"ppo_g1 seed0 |{k} - alone| per update:", {m: np.abs(res[k][m] - res["alone"][m]).max() for m in res["alone"]},
          "rel loss", (np.abs(res[k]["loss"] - res["alone"]["loss"]) / np.abs(res["alone"]["loss"])).tolist())
print("ta pair vs big:", np.abs(res["pair_ta"]["loss"] - res["big_ta"]["loss"]).tolist())
