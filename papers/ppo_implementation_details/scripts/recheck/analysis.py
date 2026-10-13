"""Recheck read-outs. numpy/scipy only.
Writes runs/recheck/fix/analysis.json and prints the tables pasted into."""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from scripts.full import report as G  # noqa: E402  (PAPER_RAW, n_aware, welch_equiv, oracle loader)
from src.eval import load_runs, summarize  # noqa: E402
from rl_jax_bench.stats import paired_compare  # noqa: E402

R = os.path.join(REPO, "runs")
F = os.path.join(R, "recheck", "fix")
OR = G.OR
out = {}


def fw(tag, env, arm):
    return {s: r["summary"]["final_window"] for s, r in load_runs(os.path.join(R, tag), env, arm).items()}


def ms(v):
    v = np.asarray(list(v), float)
    return f"{v.mean():.1f} ± {v.std(ddof=1) / np.sqrt(len(v)):.1f} (n={len(v)})"


def vs_paper(vals, env, arm):
    raw = G.PAPER_RAW[f"{env}|{arm}"]
    p = raw["final_window_mean"]
    m = float(np.mean(vals))
    rel = (m - p) / abs(p)
    j = "MATCH" if abs(rel) <= 0.05 else ("CLOSE" if abs(rel) <= 0.2 else "DIVERGENT")
    return dict(ours=m, n=len(vals), paper=p, rel=rel, judgment=j, **G.n_aware(list(vals), raw["final_window_per_seed"]))


def post(env, arm, seeds=None):
    sc = "ppo" if arm == "separate" else "ppo_shared"
    fs = glob.glob(f"{G.REF}/post_{sc}__{env}__*.json") + glob.glob(f"{F}/post_s6_10/post_{sc}__{env}__*.json")
    v = {}
    for f in fs:
        d = json.load(open(f))
        if d["returncode"] != 0:
            continue
        e = np.array([[s, r, 1, 0] for s, r in d["episodes"]], float)
        v[d["seed"]] = summarize(e, d["total_timesteps"], env)["final_window"]
    return v


# ----------------------------------------------------------------: window on U*batch instead of total
r1 = {}
for tag, env, arm, total in [("full", "CartPole-v1", "separate", 500000), ("full", "CartPole-v1", "shared", 500000),
                             ("full", "Acrobot-v1", "separate", 500000), ("full", "Acrobot-v1", "shared", 500000),
                             ("full", "Pendulum-v1", "default", 1000000)]:
    old, new = [], []
    for s, d in load_runs(os.path.join(R, tag), env, arm).items():
        e = np.asarray(d["episodes"], float)
        ub = int(d["per_update_global_step"][-1])
        old.append(summarize(e, total, env)["final_window"]); new.append(summarize(e, ub, env)["final_window"])
    r1[f"{env}/{arm}"] = dict(old=float(np.mean(old)), new=float(np.mean(new)), U_batch=ub,
                              rel_pct=float(100 * (np.mean(new) - np.mean(old)) / abs(np.mean(old))))
out["window_check"] = r1
print("\n## window_check: final window on total vs U*batch")
for k, v in r1.items():
    print(f"{k}: {v['old']:.2f} -> {v['new']:.2f} ({v['rel_pct']:+.3f} %), U*batch={v['U_batch']}")

# ----------------------------------------------------------------: before/after on CPU, and the loss claim
f3 = {}
print("\n## before_after: before/after (CPU, same seeds) and Q-A loss half vs the paper's raw W&B losses")
KEYS = {"losses/policy_loss": ("pg_loss_last", "pg_loss"), "losses/value_loss": ("v_loss_last", "v_loss"),
        "losses/entropy": ("entropy_last", "entropy"), "losses/approx_kl": ("k3_last", "k3_last")}
for env in ("CartPole-v1", "Acrobot-v1"):
    for arm in ("separate", "shared"):
        a = load_runs(os.path.join(F, "prefix_cpu"), env, arm)
        b = load_runs(os.path.join(F, "postfix_cpu"), env, arm)
        row = {}
        if a and b:
            common = sorted(set(a) & set(b))
            dfw = [b[s]["summary"]["final_window"] - a[s]["summary"]["final_window"] for s in common]
            dpu = max(float(np.nanmax(np.abs(np.asarray(b[s]["per_update"][k], float) - np.asarray(a[s]["per_update"][k], float))))
                      for s in common for k in ("pg_loss", "v_loss", "entropy", "k3_last", "clipfrac"))
            row.update(n=len(common), fw_before=float(np.mean([a[s]["summary"]["final_window"] for s in common])),
                       fw_after=float(np.mean([b[s]["summary"]["final_window"] for s in common])),
                       max_abs_fw_diff=float(np.max(np.abs(dfw))), max_abs_shared_field_diff=dpu,
                       prefix_has_last=("pg_loss_last" in a[common[0]]["per_update"]))
        exp = "ppo" if arm == "separate" else "ppo_shared"
        P = {}
        for f in glob.glob(f"{R}/recheck/measurement/wb_losses/{env}__{exp}__*.json"):
            rows = json.load(open(f))
            for k in KEYS:
                P.setdefault(k, []).append(float(np.nanmean([r[k] for r in rows if r.get(k) is not None])))
        for k, (new_k, old_k) in KEYS.items():
            o_new = [float(np.nanmean(b[s]["per_update"][new_k])) for s in sorted(b)] if b else []
            o_old = [float(np.nanmean(b[s]["per_update"][old_k])) for s in sorted(b)] if b else []
            p = np.asarray(P.get(k, []), float)
            if not len(o_new) or not len(p):
                continue
            rel_new = (np.mean(o_new) - p.mean()) / abs(p.mean())
            rel_old = (np.mean(o_old) - p.mean()) / abs(p.mean())
            j = "MATCH" if abs(rel_new) <= 0.05 else ("CLOSE" if abs(rel_new) <= 0.2 else "DIVERGENT")
            row[k] = dict(paper=float(p.mean()), paper_se=float(p.std(ddof=1) / np.sqrt(len(p))), n_paper=len(p),
                          ours_new=float(np.mean(o_new)), ours_new_se=float(np.std(o_new, ddof=1) / np.sqrt(len(o_new))),
                          ours_old=float(np.mean(o_old)), ratio_new=float(np.mean(o_new) / p.mean()),
                          ratio_old=float(np.mean(o_old) / p.mean()), rel_new=float(rel_new), rel_old=float(rel_old),
                          welch_p_new=float(st.ttest_ind(o_new, p, equal_var=False).pvalue), judgment=j)
        f3[f"{env}/{arm}"] = row
        print(f"{env} {arm}: before {row.get('fw_before', float('nan')):.2f} after {row.get('fw_after', float('nan')):.2f} "
              f"max|Δfw| {row.get('max_abs_fw_diff', float('nan')):.3g}, max|Δ shared fields| {row.get('max_abs_shared_field_diff', float('nan')):.3g}")
        for k in KEYS:
            if k in row:
                x = row[k]
                print(f"   {k:20s} paper {x['paper']:.4g}±{x['paper_se']:.2g} | ours last-mb {x['ours_new']:.4g}±{x['ours_new_se']:.2g} "
                      f"(ratio {x['ratio_new']:.3f}, {100 * x['rel_new']:+.1f} %, Welch p={x['welch_p_new']:.2f}, {x['judgment']}) | "
                      f"pre-fix all-mb ratio {x['ratio_old']:.3f}")
out["before_after"] = f3

# ---------------------------------------------------------------- vf_grid: symmetric vf_coef grid (CPU)
t1 = {}
print("\n## vf_grid: separate vs shared at matched vf_coef (CPU, seeds 1-10, paired by seed)")
for env in ("CartPole-v1", "Acrobot-v1"):
    for vf, tag in (("0.5", "recheck/fix/postfix_cpu"), ("0.1", "recheck/fix/vf0.1"), ("0.01", "recheck/fix/vf0.01")):
        s, h = fw(tag, env, "separate"), fw(tag, env, "shared")
        if not (s and h):
            continue
        c = sorted(set(s) & set(h))
        pc = paired_compare([s[i] for i in c], [h[i] for i in c])
        t1[f"{env}/vf{vf}"] = dict(separate=[s[i] for i in c], shared=[h[i] for i in c], **pc)
        print(f"{env} vf {vf}: separate {ms(s.values())} shared {ms(h.values())} sep-shared "
              f"{pc['diff_mean']:+.1f} ± {pc['diff_se']:.1f} p={pc['t_p']:.3g} wins {pc['wins']}/{len(c)}")
out["vf_grid"] = t1

# ---------------------------------------------------------------- / / three-way table
print("\n## Three-way: paper raw (n=3) / authors' code on our hardware / ours")
tw = {}
for env in ("CartPole-v1", "Acrobot-v1"):
    for arm in ("separate", "shared"):
        o10 = fw("full", env, arm)
        o50 = dict(o10); o50.update(fw("full_manyseeds", env, arm))
        fr = fw("recheck/fix/fresh_cpu", env, arm)
        po = post(env, arm)
        cl = G.oracle(G.REF + "/" + ("ppo" if arm == "separate" else "cleanrl_ppo_shared") + "__{env}__*.json+" + G.G3 +
                      "/e1_500k_s6_10/" + ("ppo" if arm == "separate" else "cleanrl_ppo_shared") + "__{env}__*.json", env, 500000)
        row = dict(paper=G.PAPER_RAW[f"{env}|{arm}"],
                   ours_n10=vs_paper(list(o10.values()), env, arm),
                   ours_fresh=vs_paper(list(fr.values()), env, arm) if fr else None,
                   post_n=vs_paper(list(po.values()), env, arm) if po else None,
                   post_seeds=po, cleanrl=vs_paper(cl["fw"], env, arm) if cl else None)
        if len(o50) > len(o10):
            row["ours_n50"] = vs_paper(list(o50.values()), env, arm)
        if fr:
            pool = list(o10.values()) + list(fr.values())
            row["ours_pooled_1_10_101_120"] = vs_paper(pool, env, arm)
        if po:
            row["ours_vs_post"] = G.welch_equiv(list(o10.values()), list(po.values()))
        tw[f"{env}/{arm}"] = row
        print(f"{env} {arm}: paper {G.PAPER_RAW[f'{env}|{arm}']['final_window_mean']:.1f}")
        for k in ("ours_n10", "ours_n50", "ours_fresh", "ours_pooled_1_10_101_120", "post_n", "cleanrl"):
            x = row.get(k)
            if x:
                wp = "–" if x["welch_p"] is None else f"{x['welch_p']:.2f}"
                print(f"   {k:28s} {x['ours']:8.1f} (n={x['n']}) {100 * x['rel']:+6.1f} % {x['judgment']:9s} Welch p={wp} "
                      f"q={x['quantile']:.3f} (p2={x['quantile_p2']:.3f}) CI {np.round(x['boot95'], 1).tolist()}")
        if "ours_vs_post" in row:
            e = row["ours_vs_post"]
            print(f"   ours(10) vs post({len(po)}) {100 * e['rel']:+.1f} % p={e['welch_p']:.2f} -> {e['judgment']}")
out["three_way"] = tw

# ---------------------------------------------------------------- larger_n: LSTM row at n=30/20, rows under
print("\n## larger_n re-reads under the one-SD equivalence rule (exploratory, larger n)")
f1r = os.path.join(R, "recheck", "official-oracle")
sys.argv = ["x"]
import runpy  # noqa: E402
cwd = os.getcwd()
os.chdir(f1r)
g = runpy.run_path(os.path.join(f1r, "f1_report.py"))
os.chdir(cwd)
on_o, on_c, off_o, off_c = g["on_o"], g["on_c"], g["off_o"], g["off_c"]
lstm_on = G.welch_equiv(list(on_o.values()), list(on_c.values()))
lstm_off = G.welch_equiv(list(off_o.values()), list(off_c.values()))
print(f"LSTM L3 on: ours {ms(on_o.values())} vs CleanRL {ms(on_c.values())}: {100 * lstm_on['rel']:+.1f} % p={lstm_on['welch_p']:.2f} "
      f"CI {np.round(lstm_on['boot95'], 1).tolist()} margin {lstm_on['margin']:.1f} -> {lstm_on['judgment']}")
print(f"LSTM L3 off: ours {ms(off_o.values())} vs CleanRL {ms(off_c.values())}: {100 * lstm_off['rel']:+.1f} % p={lstm_off['welch_p']:.2f} -> {lstm_off['judgment']}")
f5 = json.load(open(os.path.join(f1r, "f5", "report.json")))["per_seed"]
pend30 = {}
for d in (os.path.join(R, "full", "Pendulum-v1", "default"),
          os.path.join(R, "recheck", "learner-strength", "pendulum", "Pendulum-v1", "v1_more")):
    for f in glob.glob(d + "/seed_*/metrics.json"):
        j = json.load(open(f)); pend30[j["seed"]] = j["summary"]["final_window"]
b0, b1 = list(f5["b0"].values()), list(f5["b1"].values())
e30 = G.welch_equiv(list(pend30.values()), b0)
e10 = G.welch_equiv(list(f5["ours_v1"].values()), b0)
e11 = G.welch_equiv(list(f5["ours_v1"].values()), b1)
for lab, e, a in (("ours v1 n=30 vs baselines v0 n=10", e30, pend30.values()), ("ours v1 n=10 vs baselines v0 n=10", e10, f5["ours_v1"].values()),
                  ("ours v1 n=10 vs baselines v1 n=5 (same dynamics)", e11, f5["ours_v1"].values())):
    print(f"{lab}: ours {ms(a)}; {100 * e['rel']:+.1f} % p={e['welch_p']:.2f} CI {np.round(e['boot95'], 1).tolist()} margin {e['margin']:.1f} -> {e['judgment']}")
out["larger_n"] = dict(lstm_on=lstm_on, lstm_off=lstm_off, pend_ours30_vs_b0=e30, pend_ours10_vs_b0=e10, pend_ours10_vs_b1=e11,
                   n_pend30=len(pend30))
json.dump(out, open(os.path.join(F, "analysis.json"), "w"), indent=1, default=float)
