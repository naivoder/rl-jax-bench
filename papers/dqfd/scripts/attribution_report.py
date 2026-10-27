"""Attribution checks of the DQfD freeze: paired comparison of variant arms against the main-grid
DQfD on the same seeds, using the pre-declared rule  'unfreezes' / 'short pre-training was a cause' iff
   FINAL exceeds the reference FINAL by > 3 paired SE with all seeds positive.
usage: attribution_report.py flags|flags_tau|pre750k
  flags      Breakout variant flags (Huber, update cadence, loss priority, max n-step, lr)
  flags_tau  update cadence with the target period matched in env steps
  pre750k    Asterix with the paper's k = 750 k pre-training updates"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats as sps
from rl_jax_bench import stats as S
import analyze as G


def raw(dirname, game):
    arms = {}
    for p in __import__("glob").glob(os.path.join(G.REPO, dirname, "seed_*", "metrics.json")):
        seed = int(p.split("seed_")[1].split("/")[0]); cell = json.load(open(p)).get(f"{game}/subopt")
        if cell:
            for a, r in cell.items():
                if a != "meta":
                    arms.setdefault(a, {})[seed] = r
            meta = cell["meta"]
    return arms, meta


def final(r, nbins, last):   # pooled mean of the last `last` bins among the first nbins
    s, c = np.asarray(r["bin_sum"])[:nbins], np.asarray(r["bin_cnt"])[:nbins]
    return s[nbins - last:].sum() / max(c[nbins - last:].sum(), 1)


def compare(ref, var, nbins, last, tm, tse):
    seeds = sorted(set(ref) & set(var))
    a = np.array([final(var[s], nbins, last) for s in seeds]); b = np.array([final(ref[s], nbins, last) for s in seeds]); d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    gr = np.array([np.mean(var[s]["final_greedy"]) for s in seeds if "final_greedy" in var[s]])
    out = dict(n=len(seeds), final_mean=float(a.mean()), final_se=float(a.std(ddof=1) / np.sqrt(len(a))), ref_final_mean=float(b.mean()),
               diff=float(d.mean()), diff_se=float(se), wins=int((d > 0).sum()), unfreezes=bool(d.mean() > 3 * se and (d > 0).all()),
               p_one_sided=float(sps.ttest_rel(a, b).pvalue / 2 if d.mean() > 0 else 1 - sps.ttest_rel(a, b).pvalue / 2))
    if len(gr):
        gd = gr - tm
        out.update(greedy_mean=float(gr.mean()), greedy_se=float(gr.std(ddof=1) / np.sqrt(len(gr))), greedy_minus_demonstrator=float(gd.mean()),
                   greedy_exceeds_3se=bool(gd.mean() > 3 * max(gr.std(ddof=1) / np.sqrt(len(gr)), tse)))
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    res = {}
    if which == "flags":
        ref, meta0 = raw("runs/breakout_first5", "Breakout")
        var, meta = raw("runs/full/breakout_flags", "Breakout")
        tm, tse = meta["demonstrator_info"]["true_mean"], meta["demonstrator_info"]["true_se"]
        nb, last = 40, 4     # 200 k steps / 5 k bins; last 20 k steps
        for a in var:
            res[a] = compare(ref["DQfD"], var[a], nb, last, tm, tse)
    elif which == "flags_tau":     # update cadence with tau counted in ENV steps (tau = 2500 updates at 1 update / 4 steps)
        ref, meta0 = raw("runs/breakout_first5", "Breakout")
        var, meta = raw("runs/full/breakout_flags_tau", "Breakout")
        r1, _ = raw("runs/full/breakout_flags", "Breakout")
        tm, tse = meta["demonstrator_info"]["true_mean"], meta["demonstrator_info"]["true_se"]
        for a in var:
            res[a + " (tau=2500 updates)"] = compare(ref["DQfD"], var[a], 40, 4, tm, tse)
        res["flags upd_every=4 (tau=10000 updates)"] = compare(ref["DQfD"], r1["DQfD+upd_every=4"], 40, 4, tm, tse)
        res["matched-tau minus flags upd_every=4"] = compare(r1["DQfD+upd_every=4"], var["DQfD+upd_every=4"], 40, 4, tm, tse)
    else:
        ref, _ = raw("runs/full/Asterix", "Asterix")
        var, meta = raw("runs/full/asterix_pre750k", "Asterix")
        tm, tse = meta["demonstrator_info"]["true_mean"], meta["demonstrator_info"]["true_se"]
        nb = 100; last = 3   # 300 k / 3 k bins; last 9 k steps
        res["DQfD_k750k_vs_k75k"] = compare(ref["DQfD"], var["DQfD"], nb, last, tm, tse)
        d = np.array([final(var["DQfD"][s], nb, last) for s in sorted(var["DQfD"])]) - tm
        res["H2b_prime_k750k"] = dict(diff_mean=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(len(d))), wins=int((d > 0).sum()),
                                      p_one_sided=float(sps.ttest_1samp(d, 0).pvalue / 2 if d.mean() > 0 else 1 - sps.ttest_1samp(d, 0).pvalue / 2))
        if "IMIT" in var:
            v = np.array([var["IMIT"][s]["online_return"] for s in sorted(var["IMIT"])])
            vi = np.array([ref["IMIT"][s]["online_return"] for s in sorted(var["IMIT"])]) if "IMIT" in ref else None
            res["IMIT_k750k"] = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))), k75k_mean=None if vi is None else float(vi.mean()))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(G.REPO, "runs/full", f"attribution_{which}.json"), "w"), indent=1)
