"""Consolidated per-seed metrics: runs/seed_<n>/metrics.json = for every game and arm that has this seed, the full-scale window scores
(EARLY/BEST/FINAL, end-of-run greedy, demo Q-gap) and the path of the raw per-bin file it was computed from."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze as G

out = {}
for game in G.GAMES:
    arms, meta, _ = G.load_game(game)
    if not arms:
        continue
    per, eb, wb = G.per_arm_scores(arms, meta, 0.05, 0.03)
    for arm, p in per.items():
        for i, s in enumerate(p["seeds"]):
            row = dict(early=float(p["early"][i]), best=float(p["best"][i]), final=float(p["final"][i]), raw=arms[arm][s]["_dir"] + f"/seed_{s}/metrics.json")
            if p["greedy"] is not None and not np.isnan(p["greedy"][i]):
                row["final_greedy"] = float(p["greedy"][i])
            if "gap" in p and not np.isnan(p["gap"][i]):
                row["final_demo_gap"] = float(p["gap"][i])
            out.setdefault(s, {}).setdefault(f"{game}/subopt", {})[arm] = row
            out[s][f"{game}/subopt"]["_meta"] = dict(steps=meta["steps"], bin_width=meta["bin_width"], pretrain_steps=meta["pretrain_steps"],
                                                       demonstrator_true_mean=meta["demonstrator_info"]["true_mean"], demo_sha256=meta["demo_sha256"])
for s, d in out.items():
    os.makedirs(os.path.join(G.REPO, "runs", f"seed_{s}"), exist_ok=True)
    json.dump(d, open(os.path.join(G.REPO, "runs", f"seed_{s}", "metrics.json"), "w"), indent=1)
print("wrote", sorted(out))
