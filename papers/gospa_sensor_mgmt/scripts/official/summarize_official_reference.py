"""Three-way table: paper Table III / the authors' UNMODIFIED main.py on our hardware (runs/ref_main) / our JAX 10-run CFG-O runs (runs/small).

Pools the authors' single-run jobs (one MC run each, distinct block seeds; seed index 0 uses their own seed 123546) per (variant, cost, clutter).
Statistic = Table III (mean over steps of sqrt(mean over runs of GOSPA^2)); 95 % run-bootstrap CI; per-run mean GOSPA is the unit for the
authors-vs-ours Welch test (independent RNG streams, so unpaired).  Usage: python scripts/official/summarize_official_reference.py [--json out.json]"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats as sst
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from src import model as M
from src import eval as E

ROOT = os.environ.get("GOSPA_RUNS", "runs")
LAMS = [0.1, 1.0, 2.0]


def load_official(variant):
    pool = {}                                           # (alg, lam) -> list of gospa_sq [200]
    seeds = {}
    for f in sorted(glob.glob(f"{ROOT}/ref_main/{variant}_*_lam*_s*.json")):
        m = re.search(rf"{variant}_(gospa|kld)_lam([0-9.]+)_s(\d+)\.json", f)
        cost, lam, si = m.group(1), float(m.group(2)), int(m.group(3))
        d = json.load(open(f))
        for k, v in d.items():
            if not isinstance(v, dict) or "gospa_sq" not in v:
                continue
            if k == "Myopic-GD" and cost == "kld":
                continue                                # in a --cost kld job this block ran with the KLD cost: not Myopic-GD
            if k == "Myopic-KLD" and cost == "kld" and variant == "def" and False:
                continue
            name = {"MCTS1": "MCTS1", "MCTS2": "MCTS2", "MCTS3": "MCTS3", "MCTS4": "MCTS4"}.get(k, None)
            alg = k if name is None else f"{name}-{'GD' if cost == 'gospa' else 'KLD'}"
            if k == "Myopic-KLD" and cost == "kld":
                continue                                # identical block already present in the gospa job of the same (lam, seed)
            pool.setdefault((alg, lam), []).append(np.asarray(v["gospa_sq"][0], float))
            seeds.setdefault((alg, lam), []).append(si)
    return pool, seeds


def cell_stats(arrs):
    g = np.stack(arrs)
    G, _ = E.table3_stat(g)
    return dict(n=len(arrs), G=G, ci=E.boot_G(g, 2000) if len(arrs) > 1 else [float("nan")] * 2, per_run=E.run_scalar(g))


def main():
    out = {}
    lines = []
    for variant, title, otag, ocfg, onr in (("def", "authors' main.py, shipped parameters (CFG-O-equivalent: r_b 0.05, R = 3 I)", "small", "CFG-O", 10),
                                           ("pp", "authors' main.py with the PAPER's stated parameters (r_b 0.03, R = 2 I); everything else shipped.  Ours = CFG-O with the same two overrides (tag pp_ours, 20 runs)", "pp_ours", "CFG-O+r_b=0.03,R_var=2.0", 20)):
        pool, seeds = load_official(variant)
        if not pool:
            continue
        zs = []
        lines.append(f"\n#### {title}\n")
        lines.append("| algorithm | lambda_C | paper | authors' main.py G [95% CI] (n runs) | grade vs paper | ours G [95% CI] (CFG-O n=10; pp: CFG-O+overrides n=20) | authors vs ours: dG, z (per-run mean GOSPA, SD from our runs) |")
        lines.append("|---|---|---|---|---|---|---|")
        for alg in M.ALG_ORDER[:10]:
            for lam in LAMS:
                if (alg, lam) not in pool:
                    continue
                st = cell_stats(pool[(alg, lam)])
                paper = M.PAPER_TABLE3[alg][M.CLUTTER_RATES.index(lam)]
                rel, gr = E.grade(st["G"], paper)
                try:
                    c = E.load_cell(f"{ROOT}", otag, ocfg, alg, onr)[str(lam)]
                    Go, _ = E.table3_stat(c["gospa_sq"]); cio = E.boot_G(c["gospa_sq"], 2000); po = E.run_scalar(c["gospa_sq"])
                    if po.std(ddof=1) > 1e-6:          # stochastic cell: z of the authors' mean per-run GOSPA against ours, SD taken from our 10 runs (H0: same distribution)
                        zc = float((st["per_run"].mean() - po.mean()) / (po.std(ddof=1) * np.sqrt(1.0 / st["n"] + 1.0 / len(po))))
                        zs.append(zc); dstr = f"{st['G'] - Go:+.2f}, z={zc:+.2f}"
                    else:
                        dstr = f"{st['G'] - Go:+.2f} (deterministic)"
                    ours = f"{Go:.2f} [{cio[0]:.1f}, {cio[1]:.1f}]"
                except Exception as ex:                                 # noqa: BLE001
                    ours, dstr, Go, cio = "n/a", "n/a", None, None
                ci = "-" if st["n"] < 2 else f"[{st['ci'][0]:.1f}, {st['ci'][1]:.1f}]"
                lines.append(f"| {alg} | {lam} | {paper:.2f} | {st['G']:.2f} {ci} (n={st['n']}) | {rel * 100:.1f} % {gr} | {ours} | {dstr} |")
                out[f"{variant}/{alg}/{lam}"] = dict(n=st["n"], G=st["G"], ci=st["ci"], paper=paper, rel=rel, grade=gr, ours_G=Go, ours_ci=cio,
                                                     seeds=seeds[(alg, lam)])
        if zs:
            zs_a = np.array(zs)
            lines.append(f"\nStochastic cells: {len(zs_a)}; mean z = {zs_a.mean():+.2f}, RMS z = {np.sqrt((zs_a ** 2).mean()):.2f}, |z| > 2 in {(np.abs(zs_a) > 2).sum()} cells, |z| > 3 in {(np.abs(zs_a) > 3).sum()} (H0: z ~ N(0,1) up to the SD estimate from 10 runs)")
            out[f"{variant}/z_summary"] = dict(n=len(zs_a), mean=float(zs_a.mean()), rms=float(np.sqrt((zs_a ** 2).mean())), n_gt2=int((np.abs(zs_a) > 2).sum()), n_gt3=int((np.abs(zs_a) > 3).sum()))
    print("\n".join(lines))
    if len(sys.argv) > 2 and sys.argv[1] == "--json":
        json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
