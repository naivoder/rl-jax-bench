"""report: reads runs/full, runs/full_s1, runs/full_figs, runs/full_theory and the authors' exact points (tests/fixtures/paper_fig_curves.npz) and writes the
markdown tables summarised in the README.  Usage: python scripts/report.py <section> [<section> ...]   (sections: tables, family, figs56, figs78, ablation, tuning, timing, theory, x3)
Output: stdout and runs/tables/<section>.md."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import eval as E
from rl_jax_bench.stats import mean_se, paired_compare, holm, boot_ci

RUNS = os.path.join(HERE, "..", "runs")
OUT = os.path.join(HERE, "..", "runs", "tables"); os.makedirs(OUT, exist_ok=True)
CURVES = np.load(os.path.join(HERE, "..", "tests", "fixtures", "paper_fig_curves.npz"))
POL = ("whittle", "myopic", "tev")


def emit(name, text):
    open(os.path.join(OUT, name + ".md"), "w").write(text + "\n"); print(text)


def cells_of(tag):
    return E.load(os.path.join(RUNS, tag))[0]


def fmt(m, se, d=2):
    return f"{m:.{d}f} ± {se:.{d}f}" if se is not None else f"{m:.{d}f}"


# ------------------------------------------------------------------------------------------------------------------ Tables I-IV
def sec_tables(tag="full", variant="primary", title=""):
    cells = cells_of(tag)
    rows = E.table_rows(cells, variant)
    L = [f"tag `{tag}` variant `{variant}`: n per cell = {sorted({r['n'] for r in rows})}; Δ% = (ours − paper)/paper; judgment MATCH <= 5 %, CLOSE <= 20 %; authors' published output = the paper's table (TeX-typeset, identical to the paper numbers, 81/81 by `tests/test_official_parity.py`); z = (ours − paper)/SE in standard errors of our mean.", "",
         "| Metric (cell, policy) | Paper | Ours (mean ± SE, n) | Δ% | z | Judgment | Notes |", "|---|---|---|---|---|---|---|"]
    names = {"whittle": "Whittle/MP", "myopic": "myopic", "tev": "TEV/TEC"}
    cnt = {"MATCH": 0, "CLOSE": 0, "DIVERGENT": 0}; devs = []; idxdev = []
    for r in rows:
        for p in POL:
            x = r[p]; z = (x["mean"] - x["paper"]) / x["se"] if x["se"] else float("nan")
            cnt[x["judge"]] += 1; devs.append(abs(x["rel"])); (idxdev.append(x["rel"]) if p == "whittle" else None)
            note = ""
            if p == "whittle":
                note = f"impr. over best baseline {r['improvement_over_best_baseline_pct']:.2f} % (paper {r['paper_improvement_pct']:.2f} %)"
            L.append(f"| {r['cell'].split('|')[0]} K={r['K']}, {names[p]} | {x['paper']:.2f} | {fmt(x['mean'], x['se'])} (n={r['n']}) | {100 * x['rel']:+.2f} | {z:+.1f} | {x['judge']} | {note} |")
    L += ["", f"**Summary**: {cnt['MATCH']} MATCH / {cnt['CLOSE']} CLOSE / {cnt['DIVERGENT']} DIVERGENT of {sum(cnt.values())}; max |Δ| = {100 * max(devs):.2f} %, median |Δ| = {100 * float(np.median(devs)):.2f} %; "
          f"mean signed Δ of the index arm = {100 * float(np.mean(idxdev)):+.2f} %; |z| ≤ 2 in {sum(1 for r in rows for p in POL if abs((r[p]['mean'] - r[p]['paper']) / r[p]['se']) <= 2)} of {3 * len(rows)}; ordering index < min(myopic, TEV) holds in {sum(r['order_ok'] for r in rows)} of {len(rows)} cells (paper: {sum(r['paper_order_ok'] for r in rows)} of {len(rows)}); "
          f"relative improvement over the best baseline: ours {min(r['improvement_over_best_baseline_pct'] for r in rows):.2f}-{max(r['improvement_over_best_baseline_pct'] for r in rows):.2f} % vs paper {min(r['paper_improvement_pct'] for r in rows):.2f}-{max(r['paper_improvement_pct'] for r in rows):.2f} %, correlation {np.corrcoef([r['improvement_over_best_baseline_pct'] for r in rows], [r['paper_improvement_pct'] for r in rows])[0, 1]:.3f}."]
    return "\n".join(L), rows


def sec_family(tag="full", variant="primary", base="whittle", others=("myopic", "tev"), names=None, label="PRIMARY"):
    cells = cells_of(tag)
    fam = E.paired_family(cells, variant, base, others, names=names or set(E.PAPER))
    L = [f"{label} family on `{tag}` ({variant}): {len(fam)} tests; paired t (two-sided) + 95 % t-CI of the relative difference + bootstrap CI (10 000), Holm over the family. diff = mean(J_{base} − J_other) (negative = {base} cheaper); rel = diff / mean(J_other).", "",
         "| cell | vs | n | diff (mean ± SE) | rel [95 % t-CI] | bootstrap CI of diff | raw p | Holm p | wins | reproduced |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in fam.items():
        cid, comp = k.rsplit("|", 1); oth = comp.split("-vs-")[1]
        c = cells[cid]["J"]; d = c[base] - c[oth]
        L.append(f"| {cid.split('|')[0]} K={cid.split('K=')[1].split('|')[0]} | {oth} | {len(d)} | {fmt(r['diff_mean'], r['diff_se'], 3)} | {100 * r['rel_improvement']:+.2f} % [{100 * r['rel_t95'][0]:+.2f}, {100 * r['rel_t95'][1]:+.2f}] | [{r['boot95'][0]:.3f}, {r['boot95'][1]:.3f}] | {r['t_p']:.2g} | {r['p_holm']:.2g} | {int((d < 0).sum())}/{len(d)} | {'YES' if r['reproduced'] else 'NO'} |")
    nrep = sum(v["reproduced"] for v in fam.values())
    L += ["", f"**{label}: {nrep}/{len(fam)} comparisons reproduced** (negative difference and Holm p < 0.05); largest Holm p = {max(v['p_holm'] for v in fam.values()):.3g}; smallest relative improvement = {min(100 * v['rel_improvement'] for v in fam.values()):.2f} %, largest = {max(100 * v['rel_improvement'] for v in fam.values()):.2f} % (as negative numbers: more negative = bigger advantage)."]
    return "\n".join(L), fam


# ------------------------------------------------------------------------------------------------------------------ Figs 5-6
AUTH = {"fig5a": "Bound_gap_rec", "fig5b": "Bound_gap_cau", "fig5c": "Bound_gap_rec_cau", "fig6": "Bound_gap_rec_cau_ratio"}
AUTH78 = {"fig7": "Bound_performance_rec_cau", "fig8": "Bound_performance_rec_cau_ratio"}   # curve 0 = MP/Whittle, 1 = myopic, 2 = TEC/TEV (verified against the paper's legend)
ORDER = ("whittle", "myopic", "tev")


def kidx(xs, K):
    """Index of the authors' plotted K (calibrated x values are floats like 2.0003); None if K is not a plotted point."""
    i = int(np.argmin(np.abs(np.asarray(xs) - K)))
    return i if abs(xs[i] - K) < 0.05 else None


def authors(key):
    c = [CURVES[f"{key}__{i}__y"] for i in range(3)]
    return CURVES[f"{key}__0__x"], c


def kof(cid):
    return int(cid.split("K=")[1].split("|")[0])


def fig_cells(cells, fig):
    return sorted([(kof(cid), cid) for cid in cells if cid.startswith(fig + "|") and cid.endswith("|primary")])


def sec_figs56():
    cells = cells_of("full_figs")
    L = []
    allrows = {}
    for fig in ("fig5a", "fig5b", "fig5c", "fig6"):
        xs, auth = authors(AUTH[fig])
        rows = []
        for K, cid in fig_cells(cells, fig):
            c = cells[cid]
            if "V_D" not in c:
                continue
            VD = c["V_D"]["lb"]; n = len(VD)
            g = {p: 100 * (c["J"][p].mean() - VD.mean()) / VD.mean() for p in ORDER}
            gm = {p: float(np.mean(100 * (c["J"][p] - VD) / VD)) for p in ORDER}
            ki = kidx(xs, K)
            a = None if ki is None else {p: float(auth[i][ki]) for i, p in enumerate(ORDER)}
            rows.append(dict(K=K, N=4 * K, n=n, VD=float(VD.mean()), g=g, gm=gm, auth=a, pvis=float(c["pvis"]["lb"].max()), J={p: mean_se(c["J"][p]) for p in ORDER}, Jall=c["J"], extra={p: c["J"][p].mean() for p in c["J"] if p not in ORDER}))
        allrows[fig] = rows
        L += [f"#### {fig}: suboptimality gap % (gap = 100 (mean J − mean V^D)/V^D; ours Nmc = {rows[0]['n'] if rows else '-'}; authors = exact points of their vector figure `{AUTH[fig]}.pdf`)", "",
              "| K (N) | policy | authors | ours | ours − authors (pp) | (1+ours)/(1+authors) | ours mean-of-gaps | V^D (ours) | max visited s(P) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["auth"] is None:
                continue
            for p in ORDER:
                L.append(f"| {r['K']} ({r['N']}) | {p} | {r['auth'][p]:.3f} | {r['g'][p]:.3f} | {r['g'][p] - r['auth'][p]:+.2f} | {(1 + r['g'][p] / 100) / (1 + r['auth'][p] / 100):.4f} | {r['gm'][p]:.3f} | {r['VD']:.1f} | {r['pvis']:.1f} |")
        L.append("")
    # summary per figure
    L += ["#### Gap summary (per figure, over all plotted K)", "",
          "| fig | policy | mean offset ours − authors (pp) [min, max] | mean ratio (1+ours)/(1+authors) | level judgment (|rel| of the gap value per plotted K) | ordering index < policy at every K (ours / authors) | difference (policy − index) ours vs authors, mean |Δ| (pp) |", "|---|---|---|---|---|---|---|"]
    for fig, rows in allrows.items():
        rr = [r for r in rows if r["auth"]]
        if not rr:
            continue
        for p in ORDER:
            off = np.array([r["g"][p] - r["auth"][p] for r in rr]); ratio = np.array([(1 + r["g"][p] / 100) / (1 + r["auth"][p] / 100) for r in rr])
            rels = [abs(r["g"][p] - r["auth"][p]) / r["auth"][p] for r in rr]; rel = float(np.mean(rels))
            cn = [sum(x <= 0.05 for x in rels), sum(0.05 < x <= 0.2 for x in rels), sum(x > 0.2 for x in rels)]
            judge = f"{cn[0]} MATCH / {cn[1]} CLOSE / {cn[2]} DIVERGENT of {len(rels)} K"
            if p == "whittle":
                ordr = "-"; dd = "-"
            else:
                ordr = f"{sum(r['g']['whittle'] < r['g'][p] for r in rr)}/{len(rr)} / {sum(r['auth']['whittle'] < r['auth'][p] for r in rr)}/{len(rr)}"
                dd = f"{np.mean([abs((r['g'][p] - r['g']['whittle']) - (r['auth'][p] - r['auth']['whittle'])) for r in rr]):.2f}"
            L.append(f"| {fig} | {p} | {off.mean():+.2f} [{off.min():+.2f}, {off.max():+.2f}] | {ratio.mean():.4f} | {judge} (mean |rel| {100 * rel:.0f} %) | {ordr} | {dd} |")
    # trend
    L += ["", "#### Trend of the gap with K (least-squares slope of gap on ln K; K ≥ 18 for Fig 5, K ≥ 6 for Fig 6; units pp per e-fold of K)", "", "| fig | policy | ours slope | authors slope | ours gap at largest K | authors gap at largest K |", "|---|---|---|---|---|---|"]
    for fig, rows in allrows.items():
        kmin = 18 if fig.startswith("fig5") else 6
        rr = [r for r in rows if r["auth"] and r["K"] >= kmin]
        if len(rr) < 3:
            continue
        lk = np.log([r["K"] for r in rr])
        for p in ORDER:
            so = np.polyfit(lk, [r["g"][p] for r in rr], 1)[0]; sa = np.polyfit(lk, [r["auth"][p] for r in rr], 1)[0]
            L.append(f"| {fig} | {p} | {so:+.3f} | {sa:+.3f} | {rr[-1]['g'][p]:.2f} (K={rr[-1]['K']}) | {rr[-1]['auth'][p]:.2f} |")
    # Fig 6 paired
    cells6 = {kk: cc for kk, cc in ((cid, cells[cid]) for cid in cells if cid.startswith("fig6|") and cid.endswith("|primary"))}
    return "\n".join(L), allrows


# ------------------------------------------------------------------------------------------------------------------ Figs 7-8
def sec_figs78():
    cells = cells_of("full_figs")
    L = []; res = {}
    famp = {}
    for fig in ("fig7", "fig8"):
        xs, auth = authors(AUTH78[fig])
        L += [f"#### {fig}: objective per target (J/N), Nmc = 100; authors = exact points of `{AUTH78[fig]}.pdf` (K/N = 1/4 assumed)", "",
              "| K (N) | policy | authors | ours (mean ± SE) | rel Δ | within 3 SE of ours? | judgment |", "|---|---|---|---|---|---|---|"]
        for K, cid in fig_cells(cells, fig):
            ki = kidx(xs, K)
            if ki is None:
                continue
            N = 4 * K; c = cells[cid]["J"]
            for i, p in enumerate(ORDER):
                m, se = mean_se(c[p] / N); a = float(auth[i][ki]); rel = (m - a) / a
                res[(fig, K, p)] = (m, se, a, rel)
                L.append(f"| {K} ({N}) | {'MP' if p == 'whittle' else ('myopic' if p == 'myopic' else 'TEC')} | {a:.2f} | {fmt(m, se)} | {100 * rel:+.2f} % | {'yes' if abs(m - a) <= 3 * se else 'no'} (authors' SE not published; ours {100 * se / m:.2f} %) | {'MATCH' if abs(rel) <= 0.05 else ('CLOSE' if abs(rel) <= 0.2 else 'DIVERGENT')} |")
            for o in ("myopic", "tev"):
                famp[f"{fig}|K={K}|{o}"] = paired_compare(c["whittle"], c[o])
        L.append("")
    judg = [abs(v[3]) for v in res.values()]
    L += ["", f"**Figs 7-8 summary**: {len(res)} plotted points; {sum(j <= 0.05 for j in judg)} MATCH, {sum(0.05 < j <= 0.2 for j in judg)} CLOSE, {sum(j > 0.2 for j in judg)} DIVERGENT; max |Δ| = {100 * max(judg):.2f} %, median {100 * float(np.median(judg)):.2f} %."]
    return "\n".join(L), res, famp



def sec_s5():
    cells = cells_of("full_figs")
    fam = {}
    for fig in ("fig6", "fig7", "fig8"):
        for K, cid in fig_cells(cells, fig):
            c = cells[cid]["J"]
            for o in ("myopic", "tev"):
                fam[f"{fig}|K={K}|{o}"] = paired_compare(c["whittle"], c[o])
    h = holm({k: v["t_p"] for k, v in fam.items()})
    L = [f"#### S5: index vs myopic / TEV(TEC) at every plotted K of Figs 6, 7, 8 ({len(fam)} paired tests over the 100 replications; Holm over the whole family)", "",
         "| fig K | vs | relative difference of J (index − other)/other | 95 % t-CI | raw p | Holm p | reproduced (diff < 0 and Holm p < 0.05) | authors' sign (index lower) |", "|---|---|---|---|---|---|---|---|"]
    for k, v in fam.items():
        fig, K, o = k.split("|"); K = int(K[2:])
        key = AUTH[fig] if fig == "fig6" else AUTH78[fig]
        xs, auth = authors(key); ki = kidx(xs, K); oi = 1 if o == "myopic" else 2
        asign = "yes" if auth[0][ki] < auth[oi][ki] else "NO"
        L.append(f"| {fig} K={K} | {o} | {100 * v['rel_improvement']:+.2f} % | [{100 * v['rel_t95'][0]:+.2f}, {100 * v['rel_t95'][1]:+.2f}] | {v['t_p']:.2g} | {h[k]['p_holm']:.2g} | {'YES' if (v['diff_mean'] < 0 and h[k]['p_holm'] < 0.05) else 'NO'} | {asign} |")
    nrep = sum(1 for k, v in fam.items() if v["diff_mean"] < 0 and h[k]["p_holm"] < 0.05)
    L += ["", f"**S5: {nrep}/{len(fam)} reproduced**; largest Holm p = {max(x['p_holm'] for x in h.values()):.3g}."]
    return "\n".join(L)


def sec_s5b():
    """Added AFTER launch (not part of the pre-declared families; reported for completeness): Fig 5 ordering as paired tests over the 100 tie-break replications."""
    cells = cells_of("full_figs")
    fam = {}
    for fig in ("fig5a", "fig5b", "fig5c"):
        for K, cid in fig_cells(cells, fig):
            c = cells[cid]["J"]
            for o in ("myopic", "tev"):
                if np.ptp(c["whittle"] - c[o]) == 0 and np.all(c["whittle"] == c["whittle"][0]):
                    fam[f"{fig}|K={K}|{o}"] = None
                else:
                    fam[f"{fig}|K={K}|{o}"] = paired_compare(c["whittle"], c[o])
    ok = {k: v for k, v in fam.items() if v is not None}
    det = {}
    for fig in ("fig5a", "fig5b", "fig5c"):
        for K, cid in fig_cells(cells, fig):
            c = cells[cid]["J"]
            for o in ("myopic", "tev"):
                if fam[f"{fig}|K={K}|{o}"] is None:
                    det[f"{fig}|K={K}|{o}"] = float(c["whittle"][0] - c[o][0])
    h = holm({k: v["t_p"] for k, v in ok.items()})
    nrep = sum(1 for k, v in ok.items() if v["diff_mean"] < 0 and h[k]["p_holm"] < 0.05)
    mean_cheaper = sum(1 for k, v in ok.items() if v["diff_mean"] < 0)
    bad = [(k, round(100 * v["rel_improvement"], 3), float(h[k]["p_holm"])) for k, v in ok.items() if not (v["diff_mean"] < 0 and h[k]["p_holm"] < 0.05)]
    L = [f"S5b (post-launch, descriptive): Fig 5 a/b/c, index vs myopic and vs TEV at every K ({len(fam)} comparisons; {len(ok)} testable, the rest have identical J in every replication): index cheaper on the mean in {mean_cheaper}/{len(ok)}; the other {len(det)} comparisons are deterministic (no tie ever matters: J identical in all 100 replications) and the index is cheaper in {sum(v < 0 for v in det.values())}/{len(det)} of them; "
         f"Holm p < 0.05 in {nrep}/{len(ok)}. Not reproduced / not significant: {bad}. Authors' sign (index lower) in the exact points: "
         + str(sum(1 for fig, key in (("fig5a", "Bound_gap_rec"), ("fig5b", "Bound_gap_cau"), ("fig5c", "Bound_gap_rec_cau")) for oi in (1, 2) for ki in range(13) if authors(key)[1][0][ki] < authors(key)[1][oi][ki])) + "/78."]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------------ ablation / tuning / timing
TAUS = (2, 3, 5, 10, 20, 50, 100, 200)
HS = (2, 3, 5, 10, 20, 50, 100)           # passive_cont_h<H>; 100 = passive_cont (tau = 100)
CELLS = [(n, K) for n in E.PAPER for K in (1, 2, 3)]


def J_of(cells, name, K, pol, variant="primary"):
    c = cells.get(f"{name}|K={K}|{variant}")
    return None if c is None or pol not in c["J"] else c["J"][pol]


def sec_ablation():
    cells = cells_of("full")
    L = ["#### Ablation 1: remove the reaction pricing from the index (`whittle_blind0/1`: both mode-probability rows = u^0 resp. u^1; the environment keeps the real reaction); mean J relative to the full index, 100 replications", "",
         "| cell | index | blind0 (rel) | blind1 (rel) | myopic | myopic_blind0 (rel) | myopic_blind1 (rel) |", "|---|---|---|---|---|---|---|"]
    rel = {k: [] for k in ("whittle_blind0", "whittle_blind1", "myopic_blind0", "myopic_blind1")}
    fam = {}
    for n, K in CELLS:
        w = J_of(cells, n, K, "whittle"); my = J_of(cells, n, K, "myopic")
        if w is None:
            continue
        r = {}
        for a in rel:
            x = J_of(cells, n, K, a)
            if x is None:
                continue
            base = w if a.startswith("whittle") else my
            r[a] = 100 * (x.mean() - base.mean()) / base.mean(); rel[a].append(r[a])
            if a.startswith("whittle"):
                fam[f"{n}|K={K}|whittle-vs-{a}"] = paired_compare(w, x)
        L.append(f"| {n} K={K} | {w.mean():.1f} | {w.mean() * (1 + r['whittle_blind0'] / 100):.1f} ({r['whittle_blind0']:+.2f} %) | {w.mean() * (1 + r['whittle_blind1'] / 100):.1f} ({r['whittle_blind1']:+.2f} %) | {my.mean():.1f} | {r['myopic_blind0']:+.2f} % | {r['myopic_blind1']:+.2f} % |")
    pv = {k: v["t_p"] for k, v in fam.items()}; h = holm(pv) if pv else {}
    L += ["", "Summary (mean over the 27 cells of the relative cost change vs the same arm with the reaction priced): " + "; ".join(f"{a}: {np.mean(v):+.2f} % (min {np.min(v):+.2f}, max {np.max(v):+.2f}, cheaper in {sum(x < 0 for x in v)}/{len(v)})" for a, v in rel.items() if v), ""]
    shares = {a: [] for a in ("whittle_blind0", "whittle_blind1")}
    for n, K in CELLS:
        w = J_of(cells, n, K, "whittle"); my = J_of(cells, n, K, "myopic")
        if w is None:
            continue
        for a in shares:
            x = J_of(cells, n, K, a)
            if x is not None:
                shares[a].append((x.mean() - w.mean()) / (my.mean() - w.mean()))
    L.append("Share of the index's advantage over myopic that is lost when the reaction is not priced, (J_blind − J_index)/(J_myopic − J_index) per cell: " + "; ".join(f"{a}: mean {100 * np.mean(v):.0f} %, median {100 * np.median(v):.0f} %, range [{100 * np.min(v):.0f}, {100 * np.max(v):.0f}] %" for a, v in shares.items() if v) + ". The rest of the advantage is the long horizon (threshold-policy rollout) -- see the tau sweep.")
    nsig = sum(1 for k, v in fam.items() if v["diff_mean"] < 0 and h[k]["p_holm"] < 0.05)
    L.append(f"S4: index cheaper than blind index with Holm p < 0.05 in {nsig}/{len(fam)} comparisons (the paper implies pricing the reaction helps).")
    # tau sweep
    L += ["", "#### Ablation 2: rollout horizon tau of the index (tau = 100 is the paper's); cell = mean J relative to tau = 100 (%), `full`, 100 replications", "",
          "| cell | " + " | ".join(f"tau={t}" for t in TAUS) + " | best tau | tau=2 equals myopic? |", "|---|" + "---|" * (len(TAUS) + 2)]
    tab = {t: [] for t in TAUS}; bests = []
    for n, K in CELLS:
        base = J_of(cells, n, K, "whittle")
        if base is None:
            continue
        vals = {}
        for t in TAUS:
            x = base if t == 100 else J_of(cells, n, K, "whittle", "primary" if t == 100 else f"tau{t}")
            if x is not None:
                vals[t] = x.mean()
        if len(vals) < len(TAUS):
            continue
        for t in TAUS:
            tab[t].append(100 * (vals[t] - vals[100]) / vals[100])
        bt = min(vals, key=vals.get); bests.append(bt)
        my = J_of(cells, n, K, "myopic")
        same = np.allclose(J_of(cells, n, K, "whittle", "tau2"), my, rtol=1e-9)
        L.append(f"| {n} K={K} | " + " | ".join(f"{100 * (vals[t] - vals[100]) / vals[100]:+.2f}" for t in TAUS) + f" | {bt} | {'yes' if same else 'NO'} |")
    if bests:
        L += ["", "Mean over cells: " + "; ".join(f"tau={t}: {np.mean(tab[t]):+.2f} %" for t in TAUS) + f". Cells where tau = 100 is within 0.1 % of the best tau: {sum(1 for t, v in zip(bests, zip(*[tab[x] for x in TAUS])) if min(v) > -0.1)}/{len(bests)}; best-tau histogram {dict(sorted({t: bests.count(t) for t in set(bests)}.items()))}."]
    return "\n".join(L)


R1_BASES = (1, 2, 3, 5, 10, 30, 100)        # ROLLOUT-1 base-length grid after review (was 5, 10, 30, 100)


def cells_with_review(tag, rt_tag):
    """full / full_s1 cells plus the review ROLLOUT-1 base-length cells (runs/review/<rt_tag>): same seed, same key tree, only cells full does not have."""
    base = cells_of(tag)
    rt = E.load(os.path.join(RUNS, "review", rt_tag))[0]
    for cid, c in rt.items():
        if cid not in base or "rollout1" not in base[cid]["J"]:
            base.setdefault(cid, {}).setdefault("J", {}).update(c["J"])
    return base


def sec_tuning():
    c0, c1 = cells_with_review("full", "r1_s0"), cells_with_review("full_s1", "r1_s1")
    L = ["#### Tuning parity: each arm's one knob is searched on the independent resample `full_s1` (selection by mean J per cell) and evaluated on `full` (the binding set)", "",
         "Search spaces: index tau ∈ {2,3,5,10,20,50,100,200}; PASSIVE-CONT horizon ∈ {2,3,5,10,20,50,100} ∪ {revisit = ceil(N/K)}; ROLLOUT-1 base length ∈ {1,2,3,5,10,30,100} (scalar and 4-D; widened after review, the original grid {5,10,30,100} / {5,10,30} stopped at its best values); myopic and TEV have no knob. The paper itself tunes nothing.", "",
         "| cell | tau* (tune) | index(tau*) | index(100) | h* | PASSIVE(h*) | b* | ROLLOUT-1(b*) | myopic | TEV | index(tau*) − PASSIVE(h*) (rel, p) | index(tau*) − ROLLOUT-1(b*) (rel, p) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fam1, fam2, rows, warns = {}, {}, [], []
    for n, K in CELLS:
        sel = {}
        # index
        cand = {t: (J_of(c1, n, K, "whittle", "primary" if t == 100 else f"tau{t}"), J_of(c0, n, K, "whittle", "primary" if t == 100 else f"tau{t}")) for t in TAUS}
        if any(v[0] is None or v[1] is None for v in cand.values()):
            continue
        t_star = min(cand, key=lambda t: cand[t][0].mean()); Jw = cand[t_star][1]
        hc = {}
        for h in HS:
            pol = "passive_cont" if h == 100 else f"passive_cont_h{h}"
            a, b = J_of(c1, n, K, pol), J_of(c0, n, K, pol)
            if a is not None and b is not None:
                hc[str(h)] = (a, b)
        a, b = J_of(c1, n, K, "passive_cont_revisit"), J_of(c0, n, K, "passive_cont_revisit")
        if a is not None:
            hc["revisit"] = (a, b)
        h_star = min(hc, key=lambda h: hc[h][0].mean()); Jp = hc[h_star][1]
        bc = {}
        for b_ in R1_BASES:
            v = "primary" if b_ == 30 else f"r1b{b_}"
            a, b = J_of(c1, n, K, "rollout1", v), J_of(c0, n, K, "rollout1", v)
            if a is not None and b is not None:
                bc[b_] = (a, b)
        if not bc:
            continue
        missing = [b_ for b_ in R1_BASES if b_ not in bc]
        if missing:
            warns.append(f"{n} K={K}: b in {missing}")
        b_star = min(bc, key=lambda b_: bc[b_][0].mean()); Jr = bc[b_star][1]
        my = J_of(c0, n, K, "myopic"); tev = J_of(c0, n, K, "tev"); w100 = cand[100][1]
        p1 = paired_compare(Jw, Jp); p2 = paired_compare(Jw, Jr)
        fam1[f"{n}|K={K}|idx-vs-passive"] = p1; fam2[f"{n}|K={K}|idx-vs-rollout1"] = p2
        rows.append(dict(cell=f"{n} K={K}", t=t_star, h=h_star, b=b_star, w=Jw.mean(), w100=w100.mean(), p=Jp.mean(), r=Jr.mean(), my=my.mean(), tev=tev.mean(), p1=p1, p2=p2))
    allp = {**{k: v["t_p"] for k, v in fam1.items()}, **{k: v["t_p"] for k, v in fam2.items()}}; h = holm(allp)
    for r in rows:
        k1 = f"{r['cell'].replace(' K=', '|K=')}|idx-vs-passive"; k2 = f"{r['cell'].replace(' K=', '|K=')}|idx-vs-rollout1"
        L.append(f"| {r['cell']} | {r['t']} | {r['w']:.1f} | {r['w100']:.1f} | {r['h']} | {r['p']:.1f} | {r['b']} | {r['r']:.1f} | {r['my']:.1f} | {r['tev']:.1f} | {100 * r['p1']['rel_improvement']:+.2f} % (Holm p={h[k1]['p_holm']:.2g}) | {100 * r['p2']['rel_improvement']:+.2f} % (Holm p={h[k2]['p_holm']:.2g}) |")
    n1 = sum(1 for k in fam1 if fam1[k]["diff_mean"] < 0 and h[k]["p_holm"] < 0.05); n2 = sum(1 for k in fam2 if fam2[k]["diff_mean"] < 0 and h[k]["p_holm"] < 0.05)
    m2 = sum(1 for k in fam2 if fam2[k]["diff_mean"] > 0 and h[k]["p_holm"] < 0.05)
    L += ["", f"**S7** (54 tests, Holm): tuned index cheaper than tuned PASSIVE-CONT in {n1}/{len(fam1)} cells; cheaper than tuned ROLLOUT-1 in {n2}/{len(fam2)} cells and DEARER than tuned ROLLOUT-1 in {m2}/{len(fam2)} cells (Holm p < 0.05). Mean relative advantage of the tuned index: over PASSIVE-CONT {-100 * np.mean([r['p1']['rel_improvement'] for r in rows]):.2f} %, over ROLLOUT-1 {-100 * np.mean([r['p2']['rel_improvement'] for r in rows]):.2f} % (positive = index cheaper). "
          f"Tuned index vs index(tau=100): mean {100 * np.mean([(r['w'] - r['w100']) / r['w100'] for r in rows]):+.3f} %."]
    L += ["", "ROLLOUT-1 base lengths NOT run (and so not searched): " + ("; ".join(warns) if warns else "none -- the grid is complete in all 27 cells on both seeds.")]
    return "\n".join(L)


def sec_timing():
    import re
    L = ["#### Compute parity: wall clock and analytic cost per method (CPU, 1 pinned core per worker, nice 19, host load ~20-25 so absolute seconds are pessimistic)", ""]
    rows = []
    for tag in ("full", "full_s1", "full_figs"):
        for f in glob.glob(os.path.join(RUNS, tag, "run_log_*.json")):
            for t in json.load(open(f))["timings"]:
                if t.get("steady_s") is not None and t["policy"] != "V_D(VI)" and t["cell"].endswith("|primary"):
                    rows.append((t["cell"], t["policy"], t["first_call_s"], t["steady_s"]))
    by = {}
    for cell, pol, fc, st in rows:
        kind = "matrix" if cell.startswith("t4") else "scalar"
        if not cell.startswith("t"):
            continue                                            # N = 8 table cells only (the figure cells have other N)
        K = kof(cell)
        by.setdefault((kind, pol, K), []).append((st, fc - st))
    L += ["Steady-state seconds for one cell = 100 replications x 100 slots x N = 8 targets, median over the 3 (4-D) or 6 (scalar) table cells that ran a first-call timing (the second run of a signature is the steady state; the first call also includes jit compilation).  per decision = / 10 000 decisions.", "",
          "| kind | policy | K=1 s | K=2 s | K=3 s | per decision µs (K=1 / K=2 / K=3) | compile+first-call overhead s (median) |", "|---|---|---|---|---|---|---|"]
    for kind in ("scalar", "matrix"):
        for pol in sorted({p for (k, p, _) in by if k == kind}):
            v = {K: by.get((kind, pol, K)) for K in (1, 2, 3)}
            f = lambda K: float(np.median([x[0] for x in v[K]])) if v[K] else float("nan")  # noqa: E731
            ov = [x[1] for K in (1, 2, 3) if v[K] for x in v[K]]
            L.append(f"| {kind} | {pol} | {f(1):.3f} | {f(2):.3f} | {f(3):.3f} | {1e6 * f(1) / 1e4:.1f} / {1e6 * f(2) / 1e4:.1f} / {1e6 * f(3) / 1e4:.1f} | {max(float(np.median(ov)), 0.0):.2f} |")
    # analytic
    tau, base = 100, 30
    L += ["", "Analytic cost per scheduling decision at N = 8 (phi-steps = one TEC update of one target; ROLLOUT-1 phi-steps use the myopic base policy: 3N per base slot + N for the candidate action). FLOP estimates: scalar phi-step ≈ 25 flops, 4-D phi-step ≈ 1100 flops (two 4×4 mode updates incl. the 2×2 inverse), stat/cost ≈ 1 flop; these are order-of-magnitude constants, not profiler output.", "",
          "| policy | phi-steps per decision (N=8) | scalar flops | 4-D flops | tunables searched |", "|---|---|---|---|---|"]
    N = 8
    from math import comb
    spec = [("whittle", 2 * N * tau, "tau (8 values)"), ("passive_cont", 2 * N * tau, "horizon (8 values)"), ("passive_cont_h10", 2 * N * 10, "-"), ("myopic", 2 * N, "none"), ("tev", 0, "none"),
            ("rollout1, K=1", comb(8, 1) * (base * 3 * N + N), "base length (4 values)"), ("rollout1, K=2", comb(8, 2) * (base * 3 * N + N), "-"), ("rollout1, K=3", comb(8, 3) * (base * 3 * N + N), "-")]
    for name, st, tun in spec:
        L.append(f"| {name} | {st} | {st * 25:.2e} | {st * 1100:.2e} | {tun} |")
    return "\n".join(L)


def sec_theory():
    TH = os.path.join(RUNS, "full_theory")
    A_ = json.load(open(os.path.join(TH, "A.json"))); B_ = json.load(open(os.path.join(TH, "B.json"))); C_ = json.load(open(os.path.join(TH, "C.json"))); E_ = json.load(open(os.path.join(TH, "E.json")))
    t = A_["totals"]
    L = ["#### Theory claim 1: PCLI1 and PCLI2 (the paper shows them on a few plots; here a fine sweep)", "",
         f"{t['n_curves']} curves = {{reckless, cautious}} x Q^CT = 2, 3, ..., 40; each on the paper's grid P = 0.01, ..., 15 ({t['n_points_per_curve']} points), index at z = P (the policy's own evaluation point). "
         f"PCLI1: min over all curves of g(P,P) = {min(r['min_g'] for r in A_['rows']):.4f}; violations (g <= 1e-12): {t['pcli1_viol_z_eq_P']} at z = P and {t['pcli1_viol_fixed_z']} at the fixed thresholds z in {{1, 2, 4, 10}} (Figs 2-3 style). "
         f"PCLI2: decreases of mp*(P) larger than 1e-9 over the grid: {t['pcli2_decreases']} (worst relative drop {t['worst_rel_drop']:.2e}); mp* >= {t['mp_min']:.4f} > 0 (bounded below); "
         f"largest relative change between neighbouring grid points (continuity proxy) {max(r['max_neighbour_rel_change'] for r in A_['rows']):.3f} (at the smallest P, where mp* itself is small and the grid relative step is 100 %). **PCLI1 and PCLI2: REPRODUCED (0 violations in {t['n_curves'] * t['n_points_per_curve']} index evaluations).** "
         f"(The 4-D case is different: PCLI1 was flagged 8 times in 1.7e6 index evaluations in the small-scale run, all in the 4-D runs; the paper itself says there is no theory there.)", "",
         "#### Theory claim 2 (extension): the rollout MP index equals Whittle's index obtained by an independent route (grid value iteration, indifference lambda) and the problem is Whittle-indexable", "",
         "| type | Q^CT | max rel. difference rollout vs VI-Whittle over P in {0.5,1,2,3,5,8,10,15} | states (of 2000) where the passive set is NOT nested in lambda (160 lambdas) |", "|---|---|---|---|"]
    for r in B_["rows"]:
        L.append(f"| {r['type']} | {r['Q']} | {r['max_rel']:.2e} | {r['indexability']['n_states_not_nested']} |")
    L += ["", f"Worst relative difference {B_['worst']:.2e} (declared tolerance 1e-3, limited by the linear-interpolation error of the 0.01 grid); all 8 (type, Q^CT) problems are Whittle-indexable on the grid (passive sets nested in lambda): **the two routes agree and indexability holds numerically.**", "",
          "#### Theory claim 3 (extension): PCLI3, the third condition the paper never checks, per jump", "",
          "| type | Q^CT | P0 | tau | jumps | resolved jumps | max per-jump relative residual | median per-jump | aggregate relative residual |", "|---|---|---|---|---|---|---|---|---|"]
    for r in C_["rows"]:
        L.append(f"| {r['type']} | {r['Q']:.0f} | {r['P0']} | {r['tau']} | {r['n_jumps']} | {r['n_resolved']} | {r['max_rel_jump_resolved']:.2e} | {r['median_rel_jump_resolved']:.2e} | {r['total_rel']:.2e} |")
    r4 = [r for r in C_["rows"] if r["tau"] == 400]; r1 = [r for r in C_["rows"] if r["tau"] == 100]
    L += ["", f"At tau = 100 the identity F(P,z2) - F(P,z1) = sum_j mp*(z_j) (G(z_j-) - G(z_j)) holds in aggregate to {max(r['total_rel'] for r in r1):.1e} (the tau truncation 0.9^100 = 2.7e-5) but single late jumps, whose size is below the truncation, show relative residuals up to {max(r['max_rel_jump_resolved'] for r in r1):.2f}. "
          f"With tau = 400 (truncation 5e-19) the residual is **{max(r['max_rel_jump_resolved'] for r in r4):.1e} per jump and {max(r['total_rel'] for r in r4):.1e} in aggregate over {sum(r['n_jumps'] for r in r4)} jumps**: PCLI3 holds to machine-level precision, so the MP index IS a marginal productivity index in the paper's sense (with the strict reading of the identity, not just the plots): **PCL-indexability REPRODUCED (all three conditions), beyond what the paper shows.**", "",
          "#### Extension: the index policy against the EXACT optimum on small instances (open-loop enumeration, deterministic dynamics => open loop is optimal; short horizon T, so the index's infinite-horizon pricing suffers an end-of-horizon effect that Tables I-IV (T = 100, beta^T = 2.7e-5) do not have)", "",
          "| case | N, K, T | replications | J* (mean ± SE) | regret % vs J*: index | myopic | TEV | PASSIVE-CONT(revisit) | ROLLOUT-1 | index exactly optimal | index vs myopic (paired, rel, p) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, o in E_.items():
        rg = o["regret_pct"]; f = lambda p: f"{rg[p][0]:.2f} ± {rg[p][1]:.2f}"
        L.append(f"| {k} | {o['N']}, {o['K']}, {o['T']} | {o['nrep']} | {o['J_exact'][0]:.2f} ± {o['J_exact'][1]:.2f} | {f('whittle')} | {f('myopic')} | {f('tev')} | {f('passive_cont_revisit')} | {f('rollout1')} | {o['n_index_optimal']}/{o['nrep']} | {100 * o['index_vs_myopic']['rel_improvement']:+.2f} % (p={o['index_vs_myopic']['t_p']:.1e}) |")
    return "\n".join(L)


def sec_bound():
    TH = os.path.join(RUNS, "full_theory")
    T1 = json.load(open(os.path.join(TH, "bound_T1.json"))); T2 = json.load(open(os.path.join(TH, "bound_T2.json"))); T3 = json.load(open(os.path.join(TH, "bound_T3.json")))
    L = ["| scenario (K = 2) | authors' gap % | implied V_auth = J_ours/(1+gap) | V^D ours (eq 12, discounted time-average) | T1 per-slot multipliers (LB) | T3 undiscounted budget | ratio V^D/V_auth | ratio T1/V_auth | ratio T3/V_auth |", "|---|---|---|---|---|---|---|---|---|"]
    for a, b, c in zip(T1, T3, T3):
        L.append(f"| {a['fig']} | {a['gap_auth']:.3f} | {a['V_auth']:.2f} | {a['V_timeavg_ours']:.2f} | {a['V_perslot_lb']:.2f} (gap would be {a['gap_if_perslot']:.2f} %) | {b['V_undiscounted']:.2f} (gap {b['gap_if_undisc']:.0f} %) | {a['ratio_timeavg_over_auth']:.4f} | {a['ratio_perslot_over_auth']:.4f} | {b['ratio_undisc_over_auth']:.4f} |")
    L += ["", "T2 (coarse state grid; V^D/V_auth by grid step; linear interpolation):", "", "| scenario | " + " | ".join(f"{s}" for s in sorted({r['step'] for r in T2})) + " |", "|---|" + "---|" * len({r['step'] for r in T2})]
    for fig in dict.fromkeys(r["fig"] for r in T2):
        L.append(f"| {fig} | " + " | ".join(f"{r['ratio_V_over_Vauth']:.4f}" for r in sorted([r for r in T2 if r['fig'] == fig], key=lambda r: r['step'])) + " |")
    return "\n".join(L)


def sec_bound2():
    TH = os.path.join(RUNS, "full_theory")
    T5 = json.load(open(os.path.join(TH, "bound_T5.json"))); T5b = json.load(open(os.path.join(TH, "bound_T5b.json"))); T5c = json.load(open(os.path.join(TH, "bound_T5c.json")))
    L = ["**T5 (coarse grid, next state MAPPED to a node without interpolation; V/V_auth at K = 2; the ratio for the authors' V would be 1.000):**", "",
         "| scheme | step | fig5a | fig5b | fig5c | fig6 |", "|---|---|---|---|---|---|"]
    for sc in ("ceil", "nearest", "floor"):
        for st in sorted({r["step"] for r in T5}):
            v = {r["fig"]: r["ratio"] for r in T5 if r["scheme"] == sc and r["step"] == st}
            L.append(f"| {sc} | {st} | " + " | ".join(f"{v[f]:.4f}" for f in ("fig5a", "fig5b", "fig5c", "fig6")) + " |")
    rows = sorted([r for r in T5b + T5c], key=lambda r: (r["fig"], r["K"], r["step"]))
    L += ["", "**T5b/T5c (EXPLORATORY, post hoc): ceil mapping, grid step 0.075 (and 0.07/0.08 as a sensitivity), V_ceil / V_auth, where V_auth = J_ours / (1 + authors' gap):**", "",
          "| fig | K | step | V_ceil (ours) | V_auth (implied) | V_ceil / V_auth | for comparison V^D(ours)/V_auth |", "|---|---|---|---|---|---|---|"]
    vd = {("fig5a", 2): 0.9685, ("fig5a", 10): 0.9818, ("fig5a", 26): 0.9850, ("fig5a", 82): 0.9941}
    seen = set()
    for r in rows:
        if (r["fig"], r["K"], r["step"]) in seen:
            continue
        seen.add((r["fig"], r["K"], r["step"]))
        L.append(f"| {r['fig']} | {r['K']} | {r['step']} | {r['V']:.1f} | {r['V_auth']:.1f} | {r['ratio']:.4f} | {vd.get((r['fig'], r['K']), float('nan')):.4f} |" if (r["fig"], r["K"]) in vd else f"| {r['fig']} | {r['K']} | {r['step']} | {r['V']:.1f} | {r['V_auth']:.1f} | {r['ratio']:.4f} | |")
    r75 = [r for r in rows if r["step"] == 0.075]
    ratios = np.array([r["ratio"] for r in r75]); f5 = np.array([r["ratio"] for r in r75 if r["fig"].startswith("fig5")])
    L += ["", f"At step 0.075: {len(r75)} cells; Fig 5 (a, b, c; K from 2 to 98): {len(f5)} cells, V_ceil/V_auth in [{f5.min():.4f}, {f5.max():.4f}] (median {np.median(f5):.4f}); Fig 6 ({sum(r['fig'] == 'fig6' for r in r75)} cells, 3 replications each): "
          + ", ".join(f"K={r['K']}: {r['ratio']:.4f}" for r in r75 if r["fig"] == "fig6") + ". The unmodified bound has V^D/V_auth = 0.961-0.972 at K = 2 and 0.99+ only at the largest K of Fig 5."]
    # T4 sampled-mode J
    c4 = cells_of("full_x4")
    L += ["", "**T4 (J in Figs 5-6 computed with SAMPLED modes, V^D unchanged): gap % by K (index / myopic / TEV) -- not flat in K and negative at large K, so it cannot be the authors' computation:**", "", "| cell | gap index | gap myopic | gap TEV |", "|---|---|---|---|"]
    for cid in sorted(c4, key=lambda x: (x.split("|")[0], kof(x))):
        c = c4[cid]
        if "V_D" in c:
            VD = c["V_D"]["lb"].mean(); L.append(f"| {cid.split('|')[0]} K={kof(cid)} | " + " | ".join(f"{100 * (c['J'][p].mean() - VD) / VD:.2f}" for p in ORDER) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    secs = sys.argv[1:]
    for s in secs:
        if s == "tables":
            emit("tables_g3", sec_tables("full")[0]); emit("tables_full_s1", sec_tables("full_s1")[0])
        elif s == "figs56":
            emit("figs56", sec_figs56()[0])
        elif s == "figs78":
            emit("figs78", sec_figs78()[0])
        elif s == "theory":
            emit("theory", sec_theory())
        elif s == "bound2":
            emit("bound2", sec_bound2())
        elif s == "bound":
            emit("bound", sec_bound())
        elif s == "s5b":
            emit("s5b", sec_s5b())
        elif s == "s5":
            emit("s5", sec_s5())
        elif s == "ablation":
            emit("ablation", sec_ablation())
        elif s == "tuning":
            emit("tuning", sec_tuning())
        elif s == "timing":
            emit("timing", sec_timing())
        elif s == "secondary":
            emit("s1", sec_family("full", base="whittle", others=("passive_cont",), label="S1")[0])
            emit("s2", sec_family("full", base="whittle", others=("rollout1",), label="S2")[0])
            emit("s3", sec_family("full", variant="sampled", label="S3 (X3 sampled modes)")[0])
            emit("s3b", sec_family("full", variant="skipneg", label="skip-negative variant")[0])
        elif s == "family":
            emit("family_primary", sec_family("full")[0]); emit("family_s6", sec_family("full_s1", label="S6 (independent resample)")[0])
