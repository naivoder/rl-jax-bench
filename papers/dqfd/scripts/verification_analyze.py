"""Re-grades from the verification runs.  Reuses scripts/analyze.py definitions.

  h1        H1 (DQfD vs PDD / PDDconst) in the declared 5 % window AND a >= 100 k-step window (the declared floor)
  upsample  C11 up-sample ratio with the chunk-MEAN buffer demo share (cap read from the config) + H2a as the best window of
            the seed-mean curve (paper Table 2 reading)
  long      Breakout, k = 750 k, 2 M steps (runs/full/breakout_long)
  sticky    Breakout on the sticky-action env (runs/full/breakout_sticky, data/demos_sticky/)
  maxp      running-max vs current-max new-sample priority (runs/full/breakout_maxp*)
usage: python scripts/verification_analyze.py {h1|upsample|long|sticky|maxp|all}  -> runs/full/verification/<name>.json
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import yaml
import analyze as G
from rl_jax_bench import stats as S

REPO = G.REPO
OUT = os.path.join(REPO, "runs/full/verification")
CFG = yaml.safe_load(open(os.path.join(REPO, "src/configs/default.yaml")))
os.makedirs(OUT, exist_ok=True)
MAIN = {"Breakout": ["runs/breakout_first5", "runs/breakout_probe", "runs/full/breakout_p1", "runs/full/breakout_p2", "runs/full/breakout_pddconst"],
        "SpaceInvaders": ["runs/full/SpaceInvaders", "runs/full/SpaceInvaders_pddconst"],
        "Asterix": ["runs/full/Asterix", "runs/full/Asterix_pddconst"],
        "Seaquest": ["runs/full/Seaquest", "runs/full/Seaquest_pddconst"]}


def ms(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(S.mean_se(v)[1]) if len(v) > 1 else None]


def h1():
    """Per game: DQfD vs each comparator in two EARLY windows.  Holm over the 4 games within (comparator, window)."""
    out = dict(rule="one-sided paired t, direction DQfD > comparator, Holm over 4 games within each (comparator, window); GO iff Holm p < 0.05 and boot CI lower > 0",
               windows={}, per_game={})
    fam = {}
    for g in G.GAMES:
        G.DIRS[g] = [d for d in MAIN[g] if os.path.isdir(os.path.join(REPO, d))]
        arms, meta, hist = G.load_game(g)
        steps, bw = meta["steps"], meta["bin_width"]
        # mean eps of the PDD arm in the first history chunk (the exploration the comparator really had)
        eps_first = None
        for d, h in hist.items():
            if "PDD" in h and h["PDD"].get("interact"):
                eps_first = float(np.mean(h["PDD"]["interact"][0]["eps"])); break
        rows = {}
        for wname, frac in (("declared_5pct", 0.05), ("floor_100k", 100000 / steps)):
            per, eb, wb = G.per_arm_scores(arms, meta, frac, 0.03)
            rows[wname] = dict(window_steps=eb * bw, arms={a: dict(n=len(p["seeds"]), early=ms(p["early"])) for a, p in per.items()
                                                           if a in ("DQfD", "PDD", "PDDconst", "IMIT")})
            for comp in ("PDD", "PDDconst"):
                t = G.pair(per, "DQfD", comp, "early")
                if t:
                    rows[wname][f"DQfD_gt_{comp}"] = t
                    fam.setdefault((comp, wname), {})[g] = t["p_one_sided"]
        rows["pdd_mean_eps_first_10k"] = eps_first
        out["per_game"][g] = rows
    for (comp, wname), f in fam.items():
        h = S.holm({f"{g}": p for g, p in f.items()})
        out["windows"][f"{comp}/{wname}"] = h
        for g in f:
            t = out["per_game"][g][wname][f"DQfD_gt_{comp}"]
            t["holm_p"] = h[g] if not isinstance(h[g], dict) else h[g].get("p_holm", h[g])
            lo = t["boot95"][0] if t.get("boot95") else None
            t["GO"] = bool(isinstance(t["holm_p"], (int, float)) and t["holm_p"] < 0.05 and lo is not None and lo > 0)
    json.dump(out, open(os.path.join(OUT, "h1.json"), "w"), indent=1, default=float)
    return out


def buffer_share_mean(n_demo, step, chunk, cap):
    s = np.arange(step - chunk + 1, step + 1)
    return float(np.mean(n_demo / (n_demo + np.minimum(s, cap))))


def upsample_and_h2a():
    cap = CFG["per"]["agent_capacity"]
    out = {}
    for g in G.GAMES:
        G.DIRS[g] = [d for d in G.DIRS[g]] if g in G.DIRS else []
        G.DIRS[g] = [d for d in MAIN[g] if os.path.isdir(os.path.join(REPO, d))]
        arms, meta, hist = G.load_game(g)
        info = meta["demonstrator_info"]; nd = info["n_transitions"]
        rows = {}
        for d, h in hist.items():
            if "DQfD" not in h or not h["DQfD"].get("interact"):
                continue
            ir = h["DQfD"]["interact"]
            chunk = ir[1]["step"] - ir[0]["step"] if len(ir) > 1 else ir[0]["step"]
            for r in ir:
                sh = float(np.mean(r["demo_share"])); st = r["step"]
                old = sh / (nd / (nd + min(st, cap)))
                new = sh / buffer_share_mean(nd, st, chunk, cap)
                rows.setdefault(st, []).append((old, new))
        ks = sorted(rows)
        pick = sorted(set(ks[:: max(1, len(ks) // 8)] + [ks[-1]]))
        ups = {int(k): dict(old_end_of_chunk=float(np.mean([a for a, b in rows[k]])), chunk_mean=float(np.mean([b for a, b in rows[k]]))) for k in pick}
        per, eb, wb = G.per_arm_scores(arms, meta, 0.05, 0.03)
        p = per["DQfD"]
        # H2a, Table 2 reading: best window of the SEED-MEAN curve (window mean per seed, then averaged over seeds)
        nb = p["curve"].shape[1]
        rr = {s: arms["DQfD"][s] for s in p["seeds"]}
        wins = []
        for i in range(0, nb - wb + 1):
            v = []
            for s in p["seeds"]:
                bs, bc = np.asarray(rr[s]["bin_sum"]), np.asarray(rr[s]["bin_cnt"])
                v.append(bs[i:i + wb].sum() / max(bc[i:i + wb].sum(), 1))
            wins.append(np.mean(v))
        out[g] = dict(upsample=ups, h2a=dict(mean_of_per_seed_best=float(np.mean(p["best"])), best_window_of_seed_mean=float(np.max(wins)),
                                             best_demo_episode=info["best"], demonstrator_mean=info["true_mean"],
                                             exceeds_best_demo_episode_seed_mean_reading=bool(np.max(wins) > info["best"])))
    json.dump(out, open(os.path.join(OUT, "upsample_h2a.json"), "w"), indent=1, default=float)
    return out


def single_cell(name, dirs, demo_json=None, early_fracs=(0.05,), arms_keep=None, game="Breakout", final_frac=0.03):
    """Generic re-grade of one game's cell from `dirs` (H1 vs PDD/PDDconst, H2b' vs the cell's own demonstrator, IMIT)."""
    G.DIRS[game] = dirs
    arms, meta, hist = G.load_game(game)
    info = meta["demonstrator_info"]
    steps, bw = meta["steps"], meta["bin_width"]
    res = dict(game=game, steps=steps, bin_width=bw, demonstrator=dict(true_mean=info["true_mean"], true_se=info["true_se"], best=info["best"],
               n_transitions=info["n_transitions"], sticky=info.get("sticky_action_prob", meta.get("demonstrator_info", {}).get("sticky_action_prob"))),
               windows={})
    for ef in early_fracs:
        per, eb, wb = G.per_arm_scores(arms, meta, ef, final_frac)
        row = dict(early_steps=eb * bw, final_steps=wb * bw, arms={})
        for a, p in per.items():
            row["arms"][a] = dict(n=len(p["seeds"]), early=ms(p["early"]), final=ms(p["final"]), best=ms(p["best"]),
                                  greedy=(ms(p["greedy"][~np.isnan(p["greedy"])]) if p["greedy"] is not None and (~np.isnan(p["greedy"])).any() else None))
        for comp in ("PDD", "PDDconst"):
            t = G.pair(per, "DQfD", comp, "early")
            if t: row[f"H1_DQfD_gt_{comp}"] = t
        for comp in ("PDD", "PDDconst", "IMIT"):
            t = G.pair(per, "DQfD", comp, "final")
            if t: row[f"FINAL_DQfD_gt_{comp}"] = t
        for a in per:
            if a != "IMIT":
                t = G.vs_const(per, a, "final", info["true_mean"], info["true_se"])
                if t: row[f"H2bprime_{a}_final_gt_demonstrator"] = t
        t = G.vs_const(per, "IMIT", "final", info["true_mean"], info["true_se"], direction=-1) if "IMIT" in per else None
        if t: row["C6_IMIT_lt_demonstrator"] = t
        row["catch_up_step_PDD_over_DQfD"] = G.catch_up(per, "DQfD", "PDD", bw)
        row["catch_up_step_PDDconst_over_DQfD"] = G.catch_up(per, "DQfD", "PDDconst", bw)
        # greedy descriptor vs the demonstrator's own eps = 0 return if known
        res["windows"][f"early_{ef:g}"] = row
        res["_per"] = None
    # best window of the seed-mean curve (Table 2 reading), DQfD
    if "DQfD" in per:
        wb = max(1, int(round(0.03 * steps / bw)))
        c = per["DQfD"]["curve"]
        res["DQfD_curve_mean_every_%d_bins" % max(1, c.shape[1] // 20)] = c.mean(0)[:: max(1, c.shape[1] // 20)].tolist()
    res.pop("_per", None)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1, default=float)
    return res


def long_run():
    return single_cell("long", ["runs/full/breakout_long"], early_fracs=(0.05, 0.05 * 0 + 100000 / 2e6, 0.25))


def sticky():
    return single_cell("sticky", ["runs/full/breakout_sticky"], early_fracs=(0.05, 100000 / 5e5))


def maxp():
    """DQfD with new transitions entering at the CURRENT max priority vs the running-max control (same batch composition, same keys)."""
    import glob
    from src.eval import buffer_demo_share_chunk_mean
    def load(d):
        arms = {}
        for p in glob.glob(os.path.join(REPO, d, "seed_*", "metrics.json")):
            seed = int(p.split("seed_")[1].split("/")[0]); cell = json.load(open(p))["Breakout/subopt"]
            arms[seed] = cell["DQfD"]; meta = cell["meta"]
        h = json.load(open(os.path.join(REPO, d, "history_Breakout_subopt.json")))["DQfD"]["interact"]
        return arms, meta, h
    out = {}
    res = {}
    for tag, d in (("control_running_max", "runs/full/breakout_maxp_ctrl"), ("current_max", "runs/full/breakout_maxp_curmax")):
        arms, meta, h = load(d)
        seeds = sorted(arms)
        fin = np.array([np.asarray(arms[s]["bin_sum"])[-4:].sum() / max(np.asarray(arms[s]["bin_cnt"])[-4:].sum(), 1) for s in seeds])
        early = np.array([np.asarray(arms[s]["bin_sum"])[:5].sum() / max(np.asarray(arms[s]["bin_cnt"])[:5].sum(), 1) for s in seeds])
        gr = np.array([np.mean(arms[s]["final_greedy"]) for s in seeds])
        nd = meta["demonstrator_info"]["n_transitions"]
        chunk = h[1]["step"] - h[0]["step"]
        ups = {r["step"]: float(np.mean(r["demo_share"]) / buffer_demo_share_chunk_mean(nd, r["step"], chunk, CFG["per"]["agent_capacity"])) for r in h if r["step"] in (10000, 50000, 100000, 200000)}
        res[tag] = dict(final=fin, early=early, greedy=gr, ups=ups, demo_share_200k=float(np.mean(h[-1]["demo_share"])))
        out[tag] = dict(n=len(seeds), final=ms(fin), early=ms(early), greedy=ms(gr), upsample_ratio=ups, sampled_demo_share_last_chunk=res[tag]["demo_share_200k"])
    d = res["current_max"]["final"] - res["control_running_max"]["final"]
    out["final_diff_current_minus_running"] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), wins=int((d > 0).sum()), n=len(d))
    out["rule"] = "the priority convention matters iff |FINAL diff| > 3 paired SE with 5/5 same sign, or the up-sample ratio at 200 k moves by > 10 %"
    json.dump(out, open(os.path.join(OUT, "maxp.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("h1", "all"):
        r = h1()
        for g, rows in r["per_game"].items():
            for w in ("declared_5pct", "floor_100k"):
                row = rows[w]
                print(g, w, f"window {row['window_steps']} steps; PDD mean eps first 10k = {rows['pdd_mean_eps_first_10k']}")
                print("   ", {a: (round(v['early'][0], 3), None if v['early'][1] is None else round(v['early'][1], 3), v['n']) for a, v in row["arms"].items()})
                for comp in ("PDD", "PDDconst"):
                    t = row.get(f"DQfD_gt_{comp}")
                    if t: print(f"    DQfD>{comp}: diff {t['diff_mean']:.3f} (SE {t['diff_se']:.3f}) wins {t['wins']}/{t['n']} holm {t.get('holm_p')} GO {t.get('GO')}")
    if what in ("upsample", "all"):
        print(json.dumps(upsample_and_h2a(), indent=1, default=float))
    if what == "maxp":
        print(json.dumps(maxp(), indent=1, default=float))
    if what == "long":
        print(json.dumps(long_run(), indent=1, default=float))
    if what == "sticky":
        print(json.dumps(sticky(), indent=1, default=float))
