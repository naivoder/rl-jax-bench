"""full-scale report: ours (runs/full, runs/full_lstm) vs the authors' code
(oracle_runs/reference_500k + oracle_runs/full), the declared test families, the property checks T1-T3, compute.
Same summary function (src/eval.summarize) for every implementation. Writes runs/full/report.json and prints
markdown tables. numpy/scipy only (no jax)."""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.eval import load_runs, summarize  # noqa: E402
from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

OR = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "oracle_runs")
REF, G3 = OR + "/reference_500k", OR + "/full"
PAPER = {  # plot reads: (end value, read precision, plotted band)
    ("CartPole-v1", "separate"): (490.0, 10.0, "~470–500"),
    ("CartPole-v1", "shared"): (277.0, 15.0, "~180–370"),
    ("Acrobot-v1", "separate"): (-85.0, 10.0, "end ±10"),
    ("Acrobot-v1", "shared"): (-380.0, 15.0, "−500…−100"),
    ("MountainCar-v0", "separate"): (-200.0, 0.0, "floor"),
    ("MountainCar-v0", "shared"): (-200.0, 0.0, "floor"),
}


# ------------------------------------------------------------------ loading
def ours(tag, env, arm):
    runs = load_runs(os.path.join(REPO, "runs", tag), env, arm)
    if not runs:
        return None
    seeds = sorted(runs)
    R = [runs[s] for s in seeds]
    pu = lambda k: np.array([np.asarray(r["per_update"][k], float) for r in R])  # noqa: E731  (S, U)
    k3l = pu("k3_last")
    t = R[0]["timing"]
    return dict(seeds=seeds, fw=[r["summary"]["final_window"] for r in R], auc=[r["summary"]["auc"] for r in R],
                sts=[r["summary"]["steps_to_threshold"] for r in R],
                greedy=[r["summary"]["greedy_eval_mean"] for r in R],
                k3_last_mean=float(np.nanmean(k3l)), frac_k3_gt_002=float(np.mean(k3l > 0.02)),
                clipfrac=float(np.nanmean(pu("clipfrac"))), expl_var=float(np.nanmean(pu("explained_variance"))),
                ratio_dev_max=float(np.max(pu("ratio_dev_first_mb"))),
                k1_neg_frac=float(np.mean(pu("k1") < 0)) if "k1" in R[0]["per_update"] else None,
                compile_s=t["compile_seconds"], steady_s=t["steady_seconds"], n_vmapped=t["n_seeds_vmapped"],
                total=R[0]["config"]["total_timesteps"], cfg=R[0]["config"])


def oracle(pattern, env, total):
    fs = sorted(f for p in pattern.split("+") for f in glob.glob(p.format(env=env)))
    if not fs:
        return None
    out = dict(seeds=[], fw=[], auc=[], sts=[], k3_last_mean=[], frac_k3_gt_002=[], clipfrac=[], expl_var=[],
               wall_s=[], files=[os.path.relpath(f, OR) for f in fs])
    for f in fs:
        d = json.load(open(f))
        assert d["returncode"] == 0 and d["total_timesteps"] == total, f
        e = np.array([[s, r, 1, 0] for s, r in d["episodes"]], float)
        sm = summarize(e, total, env)
        sc = d.get("scalars", {})
        k3 = np.array([v for _, v in sc.get("losses/approx_kl", [])], float)
        out["seeds"].append(d["seed"]); out["fw"].append(sm["final_window"]); out["auc"].append(sm["auc"])
        out["sts"].append(sm["steps_to_threshold"]); out["wall_s"].append(d["wall_seconds"])
        out["k3_last_mean"].append(float(k3.mean()) if k3.size else np.nan)
        out["frac_k3_gt_002"].append(float((k3 > 0.02).mean()) if k3.size else np.nan)
        out["clipfrac"].append(float(np.mean([v for _, v in sc.get("losses/clipfrac", [[0, np.nan]])])))
        out["expl_var"].append(float(np.nanmean([v for _, v in sc.get("losses/explained_variance", [[0, np.nan]])])))
    for k in ("k3_last_mean", "frac_k3_gt_002", "clipfrac", "expl_var"):
        out[k] = float(np.nanmean(out[k]))
    return out


# ------------------------------------------------------------------ statistics
def welch_equiv(a, b, n_boot=10_000, seed=0):
    """ours a vs reference b (unpaired): Welch t, bootstrap CI of the mean difference, judgment."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    t = st.ttest_ind(a, b, equal_var=False)
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1) - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    ref = abs(b.mean())
    sd = float(b.std(ddof=1)) if len(b) > 1 else 0.0
    rel = d / ref if ref > 0 else np.nan
    if abs(d) <= max(0.05 * ref, sd) and ci[0] >= -0.2 * ref and ci[1] <= 0.2 * ref:
        j = "MATCH"
    elif abs(d) <= 0.2 * ref:
        j = "CLOSE"
    else:
        j = "DIVERGENT"
    return dict(diff=float(d), rel=float(rel), welch_t=float(t.statistic), welch_p=float(t.pvalue), boot95=ci,
                ref_sd=sd, margin=float(max(0.05 * ref, sd)), judgment=j, n_a=len(a), n_b=len(b))


PAPER_RAW = json.load(open(os.path.join(REPO, "data", "paper_raw.json")))["rows"]
PAPER_REF = "raw"   # recheck: the paper's own W&B runs reduced with src/eval.summarize; "plot" = pre-fix


def paper_mean(env, arm):
    if PAPER_REF == "raw" and f"{env}|{arm}" in PAPER_RAW:
        return PAPER_RAW[f"{env}|{arm}"]["final_window_mean"]
    return PAPER[(env, arm)][0]


def n_aware(ours_fw, paper_fw, n_boot=10_000, seed=0):
    """recheck: ours (n) vs the paper's own seeds (n = 3). Primary: Welch t. Also the bootstrap 95 % CI of
    the mean difference and (descriptive) the quantile of the paper's 3-seed mean among our 3-seed means (exact over all
    C(n,3) subsets for n <= 20, else 200k resamples; tails are truncated for small n)."""
    import itertools
    a, b = np.asarray(ours_fw, float), np.asarray(paper_fw, float)
    t = st.ttest_ind(a, b, equal_var=False) if (a.std() + b.std()) > 0 else None
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1) - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    k = len(b)
    if len(a) <= 20:
        m3 = np.array([a[list(c)].mean() for c in itertools.combinations(range(len(a)), k)])
    else:
        m3 = a[rng.integers(0, len(a), (200_000, k))].mean(1)
    q = float((m3 < b.mean()).mean() + 0.5 * (m3 == b.mean()).mean())   # mid-rank for ties (MountainCar floor)
    return dict(welch_p=float(t.pvalue) if t is not None and np.isfinite(t.pvalue) else None,
                boot95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], quantile=q,
                quantile_p2=float(2 * min(q, 1 - q)),
                distinguishable=bool(t is not None and np.isfinite(t.pvalue) and t.pvalue < 0.05))


def vs_paper(m, env, arm, ours_fw=None):
    if (env, arm) not in PAPER:
        return None
    p_plot, prec, band = PAPER[(env, arm)]
    if PAPER_REF == "raw":
        raw = PAPER_RAW[f"{env}|{arm}"]
        p = raw["final_window_mean"]
        rel = (m - p) / abs(p)
        j = "MATCH" if abs(rel) <= 0.05 else ("CLOSE" if abs(rel) <= 0.2 else "DIVERGENT")
        out = dict(paper=p, paper_ref="raw W&B runs (n=3), src/eval.summarize", paper_se=raw["final_window_se"],
                   paper_per_seed=raw["final_window_per_seed"], plot_read=p_plot, rel=float(rel), judgment=j,
                   rel_vs_plot_read=float((m - p_plot) / abs(p_plot)))
        if ours_fw is not None:
            out["n_aware"] = n_aware(ours_fw, raw["final_window_per_seed"])
        return out
    p = p_plot
    rel = (m - p) / abs(p)
    # follow-up: the plot-read precision measures how well WE read the figure, not the spread of the
    # paper's results (the post states no spread), so it can no longer turn a >5 % row into a MATCH.
    # It is reported as a note (within_read_precision). full-scale rule as declared at 12:58 was "MATCH within 5 % or
    # within the read precision"; the only row it changed is Acrobot separate (-8.7 %: MATCH -> CLOSE).
    j = "MATCH" if abs(rel) <= 0.05 else ("CLOSE" if abs(rel) <= 0.2 else "DIVERGENT")
    return dict(paper=p, prec=prec, band=band, rel=float(rel), judgment=j,
                within_read_precision=bool(abs(m - p) <= prec))


def paired(a_runs, b_runs):
    """a - b paired by seed index (the same env/action/permutation keys in ours)."""
    sa = dict(zip(a_runs["seeds"], a_runs["fw"]))
    sb = dict(zip(b_runs["seeds"], b_runs["fw"]))
    common = sorted(set(sa) & set(sb))
    return paired_compare([sa[s] for s in common], [sb[s] for s in common]) | {"seeds": common}


def ms(x):
    m, se = mean_se(x)
    return f"{m:.1f} ± {se:.1f}" if se is not None else f"{m:.1f}"


# ------------------------------------------------------------------ flops (analytic; same network => same FLOPs)
def mlp_macs(dims):
    return sum(a * b for a, b in zip(dims[:-1], dims[1:]))


def flops_per_env_step(cfg, obs_dim, n_act):
    h, K = cfg["hidden"], cfg["update_epochs"]
    if cfg.get("recurrent"):
        H = cfg["lstm_hidden"]
        fwd = obs_dim * h + 4 * H * (h + H) + H * (n_act + 1)
    elif cfg["architecture"] == "shared":
        fwd = mlp_macs([obs_dim, h, h]) + h * n_act + h
    else:
        fwd = mlp_macs([obs_dim, h, h, n_act]) + mlp_macs([obs_dim, h, h, 1])
    return 2 * fwd * (1 + 3 * K)  # rollout forward + K epochs of forward+backward (~3x forward)


OBS = {"CartPole-v1": (4, 2), "Acrobot-v1": (6, 3), "MountainCar-v0": (2, 3), "Pendulum-v1": (3, 1)}


def main():
    rep = dict(ours={}, oracle={}, equivalence={}, primary={}, ablation={}, exploratory={}, properties={}, compute={})
    E1 = [("CartPole-v1", a) for a in ("separate", "shared", "advnorm_batch", "advnorm_ddof0", "no_vclip",
                                       "trunc_bootstrap")]
    E1 += [("Acrobot-v1", a) for a in ("separate", "shared", "advnorm_batch", "advnorm_ddof0", "no_vclip",
                                       "trunc_bootstrap")]
    E1 += [("MountainCar-v0", "separate"), ("MountainCar-v0", "shared")]
    E3 = [("Pendulum-v1", a) for a in ("default", "gym021", "gymnasium029", "trunc_bootstrap", "no_rscale", "no_vclip",
                                       "advnorm_batch")]
    E4 = [("CartPole-v1", a) for a in ("lstm", "gymnasium029", "lstm_noreset", "ff_masked", "ff_full",
                                       "lstm_zero_train", "no_rscale")]
    O = {}
    for tag, rows in (("full", E1 + E3), ("full_lstm", E4)):
        for env, arm in rows:
            r = ours(tag, env, arm)
            if r is not None:
                O[(tag, env, arm)] = r
                rep["ours"][f"{tag}/{env}/{arm}"] = {k: v for k, v in r.items() if k != "cfg"}
    ORC = {
        ("cleanrl", "CartPole-v1", "separate"): (REF + "/ppo__{env}__*.json+" + G3 + "/e1_500k_s6_10/ppo__{env}__*.json", 500000),
        ("cleanrl", "CartPole-v1", "shared"): (REF + "/cleanrl_ppo_shared__{env}__*.json+" + G3 + "/e1_500k_s6_10/cleanrl_ppo_shared__{env}__*.json", 500000),
        ("cleanrl", "Acrobot-v1", "separate"): (REF + "/ppo__{env}__*.json+" + G3 + "/e1_500k_s6_10/ppo__{env}__*.json", 500000),
        ("cleanrl", "Acrobot-v1", "shared"): (REF + "/cleanrl_ppo_shared__{env}__*.json+" + G3 + "/e1_500k_s6_10/cleanrl_ppo_shared__{env}__*.json", 500000),
        ("post", "CartPole-v1", "separate"): (REF + "/post_ppo__{env}__*.json", 500000),
        ("post", "CartPole-v1", "shared"): (REF + "/post_ppo_shared__{env}__*.json", 500000),
        ("post", "Acrobot-v1", "separate"): (REF + "/post_ppo__{env}__*.json", 500000),
        ("post", "Acrobot-v1", "shared"): (REF + "/post_ppo_shared__{env}__*.json", 500000),
        ("post", "MountainCar-v0", "separate"): (REF + "/post_ppo__{env}__*.json", 500000),
        ("post", "MountainCar-v0", "shared"): (REF + "/post_ppo_shared__{env}__*.json", 500000),
        ("baselines", "CartPole-v1", "separate"): (REF + "/baselines_run_separate_networks__{env}__*.json", 500000),
        ("baselines", "CartPole-v1", "shared"): (REF + "/baselines_run__{env}__*.json", 500000),
        ("baselines", "Acrobot-v1", "separate"): (REF + "/baselines_run_separate_networks__{env}__*.json", 500000),
        ("baselines", "Acrobot-v1", "shared"): (REF + "/baselines_run__{env}__*.json", 500000),
        ("cleanrl", "Pendulum-v1", "gymnasium029"): (G3 + "/pendulum_1m/ppo_continuous_action__{env}__*.json", 1000000),
        ("post", "Pendulum-v1", "gym021"): (G3 + "/pendulum_1m/post_ppo_continuous_action__{env}__*.json", 1000000),
        ("cleanrl", "CartPole-v1", "lstm_gymnasium029"): (G3 + "/lstm_500k/cleanrl_ppo_lstm_maskedcartpole__{env}__*.json", 500000),
        ("cleanrl", "CartPole-v1", "no_vclip"): (G3 + "/e1_500k_novclip/ppo__{env}__*.json", 500000),
        ("cleanrl", "Acrobot-v1", "no_vclip"): (G3 + "/e1_500k_novclip/ppo__{env}__*.json", 500000),
    }
    Q = {}
    for k, (pat, total) in ORC.items():
        r = oracle(pat, k[1], total)
        if r is not None:
            Q[k] = r
            rep["oracle"]["/".join(k)] = r

    # ---- headline table (equivalence + vs paper)
    print("\n### Headline: ours vs authors' code vs paper (final-window return, mean ± s.e.)\n")
    print("| Metric | Paper | Ours (mean ± SE, n) | Δ% vs paper | Judgment vs paper | Authors' code: CleanRL / post / baselines | "
          "Δ% vs CleanRL (Welch p; 95 % CI of Δ) | Judgment vs CleanRL (one-SD equivalence rule) | Judgment vs post | Notes |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    head = [("full", "CartPole-v1", "separate", "separate"), ("full", "CartPole-v1", "shared", "shared"),
            ("full", "Acrobot-v1", "separate", "separate"), ("full", "Acrobot-v1", "shared", "shared"),
            ("full", "MountainCar-v0", "separate", "separate"), ("full", "MountainCar-v0", "shared", "shared"),
            ("full", "Pendulum-v1", "gymnasium029", "gymnasium029"), ("full", "Pendulum-v1", "gym021", "gym021"),
            ("full", "Pendulum-v1", "default", None),
            ("full_lstm", "CartPole-v1", "gymnasium029", "lstm_gymnasium029"), ("full_lstm", "CartPole-v1", "lstm", None)]
    for tag, env, arm, okey in head:
        r = O.get((tag, env, arm))
        if r is None:
            print(f"| {env} {arm} | | NOT RUN | | | | | | | |")
            continue
        m = float(np.mean(r["fw"]))
        vp = vs_paper(m, env, arm, r["fw"])
        cl, po, bl = (Q.get((who, env, okey)) if okey else None for who in ("cleanrl", "post", "baselines"))
        eq_c = welch_equiv(r["fw"], cl["fw"]) if cl and len(cl["fw"]) > 1 else None
        eq_p = welch_equiv(r["fw"], po["fw"]) if po and len(po["fw"]) > 1 else None
        if eq_c is None and cl and env == "MountainCar-v0":
            eq_c = None
        key = f"{tag}/{env}/{arm}"
        rep["equivalence"][key] = dict(vs_paper=vp, vs_cleanrl=eq_c, vs_post=eq_p)
        name = f"{env}{' masked LSTM' if tag == 'full_lstm' else ''} `{arm}` @{r['total'] // 1000}k"
        auth = " / ".join(f"{ms(x['fw'])} (n={len(x['fw'])})" if x else "–" for x in (cl, po, bl))
        if eq_c and np.isfinite(eq_c["rel"]):
            dcl = f"{100 * eq_c['rel']:+.1f} % (p={eq_c['welch_p']:.2f}; [{eq_c['boot95'][0]:.1f}, {eq_c['boot95'][1]:.1f}])"
        else:
            dcl = "–"
        jc = eq_c["judgment"] if eq_c else ("MATCH (all −200, identical)" if env == "MountainCar-v0" and cl is None and po and np.allclose(r["fw"], -200) and np.allclose(po["fw"], -200) else "–")
        jp = eq_p["judgment"] if eq_p and np.isfinite(eq_p["rel"]) and eq_p["ref_sd"] > 0 else (
            "MATCH (identical floor)" if po and np.allclose(r["fw"], po["fw"][0]) and np.allclose(po["fw"], po["fw"][0]) else (eq_p["judgment"] if eq_p else "–"))
        if vp and PAPER_REF == "raw":
            na = vp.get("n_aware", {})
            wp = "–" if na.get("welch_p") is None else f"{na['welch_p']:.2f}"
            pcol = (f"{vp['paper']:.1f} ± {vp['paper_se']:.1f} (raw, n=3; plot read ~{vp['plot_read']:.0f}); "
                    f"Welch p={wp}, q={na.get('quantile', float('nan')):.3f}")
        else:
            pcol = f"~{vp['paper']:.0f} (±{vp['prec']:.0f}; band {vp['band']})" if vp else "none (proxy)"
        prel = f"{100 * vp['rel']:+.1f} %" if vp else "–"
        print(f"| {name} | {pcol} | {ms(r['fw'])} (n={len(r['fw'])}) | {prel} | "
              f"{vp['judgment'] if vp else '–'} | {auth} | {dcl} | {jc} | {jp} | |")

    # ---- secondary metrics (AUC, steps-to-threshold), ours vs CleanRL / post
    print("\n### Secondary metrics: AUC (50-bin mean) and steps to threshold (10-bin running mean)\n")
    print("| Setting | ours AUC | CleanRL AUC | post AUC | Δ% AUC vs CleanRL (Welch p) | ours steps-to-thr (median; reached) | CleanRL | post |")
    print("|---|---|---|---|---|---|---|---|")
    for env in ("CartPole-v1", "Acrobot-v1"):
        for arm in ("separate", "shared"):
            r, cl, po = O.get(("full", env, arm)), Q.get(("cleanrl", env, arm)), Q.get(("post", env, arm))
            if not (r and cl and po):
                continue
            w = welch_equiv(r["auc"], cl["auc"])
            rep["equivalence"][f"full/{env}/{arm}"]["auc_vs_cleanrl"] = w

            def sts(x):
                v = [y for y in x if y is not None]
                return f"{np.median(v) / 1e3:.0f}k; {len(v)}/{len(x)}" if v else f"never; 0/{len(x)}"
            print(f"| {env} `{arm}` | {ms(r['auc'])} | {ms(cl['auc'])} | {ms(po['auc'])} | {100 * w['rel']:+.1f} % "
                  f"(p={w['welch_p']:.2f}) → {w['judgment']} | {sts(r['sts'])} | {sts(cl['sts'])} | {sts(po['sts'])} |")

    # ---- primary family Q-B
    print("\n### Primary family: separate > shared, ours, paired by seed, Holm over 2\n")
    pv = {}
    for env in ("CartPole-v1", "Acrobot-v1"):
        a, b = O.get(("full", env, "separate")), O.get(("full", env, "shared"))
        if a and b:
            rep["primary"][env] = paired(a, b)
            pv[env] = rep["primary"][env]["t_p"]
    for env, h in holm(pv).items():
        rep["primary"][env]["holm"] = h
    print("| Claim | Δ (sep − shared) mean ± SE | 95 % t-CI | bootstrap CI | paired t p | Holm p | wins | supported? |")
    print("|---|---|---|---|---|---|---|---|")
    for env, c in rep["primary"].items():
        sup = "YES" if c["holm"]["p_holm"] < 0.05 and c["diff_mean"] > 0 else "NO"
        print(f"| separate > shared, {env} | {c['diff_mean']:+.1f} ± {c['diff_se']:.1f} | [{c['t95'][0]:.1f}, {c['t95'][1]:.1f}] | "
              f"[{c['boot95'][0]:.1f}, {c['boot95'][1]:.1f}] | {c['t_p']:.2g} | {c['holm']['p_holm']:.2g} | {c['wins']}/{c['n']} | {sup} |")

    # ---- ablation family
    AB = {"AB1 adv-norm per-mb vs per-batch, CartPole": (("full", "CartPole-v1", "separate"), ("full", "CartPole-v1", "advnorm_batch")),
          "AB2 adv-norm per-mb vs per-batch, Acrobot": (("full", "Acrobot-v1", "separate"), ("full", "Acrobot-v1", "advnorm_batch")),
          "AB3 value clip on vs off, CartPole": (("full", "CartPole-v1", "separate"), ("full", "CartPole-v1", "no_vclip")),
          "AB4 value clip on vs off, Acrobot": (("full", "Acrobot-v1", "separate"), ("full", "Acrobot-v1", "no_vclip")),
          "AB5 value clip on vs off, Pendulum": (("full", "Pendulum-v1", "default"), ("full", "Pendulum-v1", "no_vclip")),
          "AB6 reward scaling on vs off, Pendulum": (("full", "Pendulum-v1", "default"), ("full", "Pendulum-v1", "no_rscale")),
          "AB7 truncation bootstrap correct vs official, CartPole": (("full", "CartPole-v1", "trunc_bootstrap"), ("full", "CartPole-v1", "separate")),
          "AB8 truncation bootstrap correct vs official, Pendulum": (("full", "Pendulum-v1", "trunc_bootstrap"), ("full", "Pendulum-v1", "default")),
          "AB9 LSTM reset on vs off, masked CartPole": (("full_lstm", "CartPole-v1", "lstm"), ("full_lstm", "CartPole-v1", "lstm_noreset"))}
    EX = {"E-x1 L5 broken (zero-state training) vs correct": (("full_lstm", "CartPole-v1", "lstm_zero_train"), ("full_lstm", "CartPole-v1", "lstm")),
          "E-x2 LSTM vs memoryless ff_masked": (("full_lstm", "CartPole-v1", "lstm"), ("full_lstm", "CartPole-v1", "ff_masked")),
          "E-x3 LSTM with vs without reward scaling": (("full_lstm", "CartPole-v1", "lstm"), ("full_lstm", "CartPole-v1", "no_rscale")),
          "E-x4 truncation bootstrap, Acrobot": (("full", "Acrobot-v1", "trunc_bootstrap"), ("full", "Acrobot-v1", "separate")),
          "E-x5a ddof 0 vs 1, CartPole": (("full", "CartPole-v1", "advnorm_ddof0"), ("full", "CartPole-v1", "separate")),
          "E-x5b ddof 0 vs 1, Acrobot": (("full", "Acrobot-v1", "advnorm_ddof0"), ("full", "Acrobot-v1", "separate")),
          "E-x6a Pendulum gym021 vs baselines wrappers": (("full", "Pendulum-v1", "gym021"), ("full", "Pendulum-v1", "default")),
          "E-x6b Pendulum gymnasium029 vs baselines wrappers": (("full", "Pendulum-v1", "gymnasium029"), ("full", "Pendulum-v1", "default")),
          "E-x6c LSTM gymnasium029 vs baselines wrappers": (("full_lstm", "CartPole-v1", "gymnasium029"), ("full_lstm", "CartPole-v1", "lstm")),
          "E-x8 per-batch adv-norm, Pendulum": (("full", "Pendulum-v1", "default"), ("full", "Pendulum-v1", "advnorm_batch"))}
    for fam, D, out in (("ablation", AB, rep["ablation"]), ("exploratory", EX, rep["exploratory"])):
        pv = {}
        for name, (ka, kb) in D.items():
            if ka in O and kb in O:
                out[name] = paired(O[ka], O[kb]) | dict(a=ms(O[ka]["fw"]), b=ms(O[kb]["fw"]))
                pv[name] = out[name]["t_p"]
        if fam == "ablation":
            for n, h in holm(pv).items():
                out[n]["holm"] = h
        print(f"\n### {fam} (paired by seed{', Holm over the family' if fam == 'ablation' else ', no correction'})\n")
        print("| Test | A | B | Δ (A − B) ± SE | rel. | bootstrap CI | paired t p | Holm p | wins |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name, c in out.items():
            hp = f"{c['holm']['p_holm']:.2g}" if "holm" in c else "–"
            print(f"| {name} | {c['a']} | {c['b']} | {c['diff_mean']:+.1f} ± {c['diff_se']:.1f} | {100 * c['rel_improvement']:+.1f} % | "
                  f"[{c['boot95'][0]:.1f}, {c['boot95'][1]:.1f}] | {c['t_p']:.2g} | {hp} | {c['wins']}/{c['n']} |")
    # E-x7: the value-clip ablation inside CleanRL (oracle), Welch
    for env in ("CartPole-v1", "Acrobot-v1"):
        on, off = Q.get(("cleanrl", env, "separate")), Q.get(("cleanrl", env, "no_vclip"))
        if on and off:
            on5 = [f for s, f in zip(on["seeds"], on["fw"]) if s <= 5]
            w = welch_equiv(off["fw"], on5)
            rep["exploratory"][f"E-x7 CleanRL value clip off − on, {env}"] = dict(off=ms(off["fw"]), on=ms(on5), **w)
            print(f"\nE-x7 CleanRL {env}: no-clip-vloss {ms(off['fw'])} vs clip {ms(on5)} (seeds 1–5): Δ {w['diff']:+.1f} "
                  f"({100 * w['rel']:+.1f} %), Welch p={w['welch_p']:.2g}, CI [{w['boot95'][0]:.1f}, {w['boot95'][1]:.1f}]")

    # ---- properties T1/T2
    print("\n### Properties T1 (max|ρ−1| first minibatch, all updates) and T2 (fraction of updates with k3 > 0.02)\n")
    print("| Setting | ours max\\|ρ−1\\| | ours frac k3>0.02 | ours mean k3 (last mb) | CleanRL frac k3>0.02 | CleanRL mean k3 | ours clipfrac | ours expl. var | ours frac k1<0 |")
    print("|---|---|---|---|---|---|---|---|---|")
    okmap = {("full", "CartPole-v1", "separate"): ("cleanrl", "CartPole-v1", "separate"),
             ("full", "CartPole-v1", "shared"): ("cleanrl", "CartPole-v1", "shared"),
             ("full", "Acrobot-v1", "separate"): ("cleanrl", "Acrobot-v1", "separate"),
             ("full", "Acrobot-v1", "shared"): ("cleanrl", "Acrobot-v1", "shared"),
             ("full", "Pendulum-v1", "gymnasium029"): ("cleanrl", "Pendulum-v1", "gymnasium029"),
             ("full_lstm", "CartPole-v1", "gymnasium029"): ("cleanrl", "CartPole-v1", "lstm_gymnasium029")}
    for k, r in O.items():
        q = Q.get(okmap.get(k, ()), None)
        rep["properties"]["/".join(k)] = dict(ratio_dev_max=r["ratio_dev_max"], frac_k3_gt_002=r["frac_k3_gt_002"],
                                              k3=r["k3_last_mean"], oracle_frac_k3=q["frac_k3_gt_002"] if q else None)
        qf = f"{q['frac_k3_gt_002']:.4f}" if q else "–"
        qk = f"{q['k3_last_mean']:.2e}" if q else "–"
        k1n = "–" if r["k1_neg_frac"] is None else f"{r['k1_neg_frac']:.3f}"
        lab = "/".join(k[1:]) + (" (LSTM)" if k[0] == "full_lstm" else "")
        print(f"| {lab} | {r['ratio_dev_max']:.1e} | {r['frac_k3_gt_002']:.4f} | {r['k3_last_mean']:.2e} | {qf} | {qk} | "
              f"{r['clipfrac']:.3f} | {r['expl_var']:.2f} | {k1n} |")

    # ---- T3 key-mechanism drop
    print("\n### T3: remove the post's one tested mechanism (separate → shared): relative drop\n")
    for env in ("CartPole-v1", "Acrobot-v1"):
        s, h = O.get(("full", env, "separate")), O.get(("full", env, "shared"))
        if s and h:
            ps, ph = paper_mean(env, "separate"), paper_mean(env, "shared")
            ours_drop = (np.mean(h["fw"]) - np.mean(s["fw"])) / abs(np.mean(s["fw"]))
            pap_drop = (ph - ps) / abs(ps)
            rep["properties"][f"T3/{env}"] = dict(ours=float(ours_drop), paper=float(pap_drop))
            print(f"{env}: ours {100 * ours_drop:+.0f} % vs paper-implied {100 * pap_drop:+.0f} %")

    # ---- compute
    print("\n### Compute (per (env, arm); ours: one GPU process, seeds vmapped; oracle: CPU, one process per seed)\n")
    print("| Setting | ours compile s | ours steady s (all seeds) | ours steady env-steps/s/seed | oracle wall s/seed (mean) | analytic FLOPs per env step |")
    print("|---|---|---|---|---|---|")
    for k, r in O.items():
        env = k[1]
        cfg = r["cfg"]
        fl = flops_per_env_step(cfg, *OBS[env])
        q = Q.get(okmap.get(k, ()), None)
        sps = r["total"] / max(r["steady_s"], 1e-9)
        rep["compute"]["/".join(k)] = dict(compile_s=r["compile_s"], steady_s=r["steady_s"], sps_per_seed=sps,
                                           flops_per_step=fl, oracle_wall_s=float(np.mean(q["wall_s"])) if q else None)
        lab = "/".join(k[1:]) + (" (LSTM)" if k[0] == "full_lstm" else "")
        qw = f"{np.mean(q['wall_s']):.0f}" if q else "–"
        print(f"| {lab} | {r['compile_s']:.1f} | {r['steady_s']:.1f} | {sps:,.0f} | {qw} | {fl:,} |")
    # ---- investigations (exploratory)
    print("\n### Investigations I-1..I-4 (exploratory)\n")
    rep["investigation"] = inv = {}
    stuck = lambda fw: int(np.sum(np.asarray(fw) < -400))  # noqa: E731  (a seed that never learns sits at -500)
    for arm in ("separate", "shared"):
        a = O.get(("full", "Acrobot-v1", arm))
        b = ours("full_manyseeds", "Acrobot-v1", arm)
        if a and b:
            fw = list(a["fw"]) + list(b["fw"])
            m, se = mean_se(fw)
            vp = vs_paper(m, "Acrobot-v1", arm, fw)
            inv[f"I-1 ours Acrobot {arm} seeds 1-50"] = dict(n=len(fw), mean=m, se=se, median=float(np.median(fw)),
                                                             stuck=stuck(fw), vs_paper=vp)
            print(f"I-1 ours Acrobot {arm}, seeds 1–50: {m:.1f} ± {se:.1f} (median {np.median(fw):.1f}); stuck "
                  f"{stuck(fw)}/{len(fw)}; vs paper {100 * vp['rel']:+.1f} % → {vp['judgment']}")
    cl_extra = oracle(G3 + "/acrobot_s11_30/ppo__{env}__*.json", "Acrobot-v1", 500000)
    cl = Q.get(("cleanrl", "Acrobot-v1", "separate"))
    if cl:
        clfw = list(cl["fw"]) + (list(cl_extra["fw"]) if cl_extra else [])
        auth = clfw + list(Q[("post", "Acrobot-v1", "separate")]["fw"]) + list(Q[("baselines", "Acrobot-v1", "separate")]["fw"])
        o_fw = list(O[("full", "Acrobot-v1", "separate")]["fw"]) + list(ours("full_manyseeds", "Acrobot-v1", "separate")["fw"])
        tab = [[stuck(o_fw), len(o_fw) - stuck(o_fw)], [stuck(auth), len(auth) - stuck(auth)]]
        fe = st.fisher_exact(tab)
        m, se = mean_se(clfw)
        inv["I-1 authors' code Acrobot separate"] = dict(cleanrl_n=len(clfw), cleanrl_mean=m, cleanrl_se=se,
                                                         cleanrl_stuck=stuck(clfw), all_authors_n=len(auth),
                                                         all_authors_stuck=stuck(auth), fisher_table=tab,
                                                         fisher_p=float(fe.pvalue))
        eq = welch_equiv(o_fw, clfw)
        inv["I-1 ours (50) vs CleanRL (all seeds) Acrobot separate"] = eq
        print(f"I-1 CleanRL Acrobot separate, {len(clfw)} seeds: {m:.1f} ± {se:.1f}; stuck {stuck(clfw)}/{len(clfw)}; all "
              f"authors' code stuck {stuck(auth)}/{len(auth)}; Fisher (ours {tab[0]} vs authors {tab[1]}) p={fe.pvalue:.2f}; "
              f"ours(50) vs CleanRL Welch Δ {eq['diff']:+.1f} ({100 * eq['rel']:+.1f} %) p={eq['welch_p']:.2f} → {eq['judgment']}")
    shp = [Q.get((w, "Acrobot-v1", "shared")) for w in ("cleanrl", "post", "baselines")]
    if all(shp):
        pool = [x for q in shp for x in q["fw"]]
        o50 = list(O[("full", "Acrobot-v1", "shared")]["fw"]) + list(ours("full_manyseeds", "Acrobot-v1", "shared")["fw"])
        eq = welch_equiv(o50, pool)
        inv["I-1 ours shared (50) vs pooled authors' code"] = eq | dict(pool_mean=float(np.mean(pool)), pool_n=len(pool))
        print(f"I-1 Acrobot shared: ours (50) vs pooled authors' code ({len(pool)}: {ms(pool)}): Δ {eq['diff']:+.1f} "
              f"({100 * eq['rel']:+.1f} %), Welch p={eq['welch_p']:.2f}, CI [{eq['boot95'][0]:.1f}, {eq['boot95'][1]:.1f}] → {eq['judgment']}")
    rg = os.path.join(REPO, "runs", "full", "divergent", "random_policy_gymnax.json")
    rq = os.path.join(G3, "divergent", "random_policy_gymnasium.json")
    if os.path.exists(rg) and os.path.exists(rq):
        a, b = json.load(open(rg)), json.load(open(rq))
        z = (a["success_rate"] - b["success_rate"]) / np.hypot(a["success_se"], b["success_se"])
        inv["I-2 random-policy Acrobot"] = dict(gymnax=a, gymnasium=b, z=float(z), p=float(2 * st.norm.sf(abs(z))))
        print(f"I-2 random policy: gymnax {a['success_rate']:.4f} ± {a['success_se']:.4f} (return {a['mean_return']:.2f}) vs "
              f"gymnasium {b['success_rate']:.4f} ± {b['success_se']:.4f} (return {b['mean_return']:.2f}); z={z:.2f}")
    rn = os.path.join(REPO, "runs", "full", "divergent", "ratio_numerics.json")
    if os.path.exists(rn):
        inv["I-3 ratio numerics"] = json.load(open(rn))
        print("I-3 max|rho-1| rollout path vs training path:", json.dumps(inv["I-3 ratio numerics"]))
    on = Q.get(("cleanrl", "CartPole-v1", "lstm_gymnasium029"))
    off = oracle(G3 + "/lstm_noreset_500k/cleanrl_ppo_lstm_maskedcartpole_noreset__{env}__*.json", "CartPole-v1", 500000)
    if on and off:
        w = welch_equiv(off["fw"], on["fw"])
        inv["I-4 CleanRL LSTM L3 off - on"] = dict(off=off["fw"], on=on["fw"], **w)
        print(f"I-4 CleanRL LSTM L3 off {ms(off['fw'])} vs on {ms(on['fw'])} (n={len(off['fw'])}/{len(on['fw'])}): "
              f"Δ {w['diff']:+.1f} ({100 * w['rel']:+.1f} %), Welch p={w['welch_p']:.2g}, CI [{w['boot95'][0]:.1f}, {w['boot95'][1]:.1f}]")
    rep["paper_ref"] = PAPER_REF
    name = "report.json" if PAPER_REF == "raw" else "report_plotref.json"
    json.dump(rep, open(os.path.join(REPO, "runs", "full", name), "w"), indent=1, default=float)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-ref", choices=["raw", "plot"], default="raw",
                    help="raw = the paper's W&B runs reduced with our metric (recheck, default); "
                         "plot = plot reads of the post's figures (pre-fix behavior; writes report_plotref.json)")
    PAPER_REF = ap.parse_args().paper_ref
    main()
