"""Replace the '*' (authors' 1-4-run mean outside the CI of OUR 50-run mean; not a prediction interval) by a z-score
z = (G_authors - G_ours) / SD_n, where SD_n is the bootstrap standard deviation of the Table III statistic G for a set of n_auth runs drawn
(with replacement) from OUR 50 runs, i.e. the run-to-run noise that an n_auth-run mean really has.  |z| > 2 is flagged.
Usage: python scripts/followup/authors_zscores.py  -> runs/report/authors_zscores.json and a printed table (CFG-O and CFG-P v1)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from src import eval as E  # noqa: E402

ROOT = os.environ.get("GOSPA_RUNS", "runs")
rep = json.load(open(os.path.join(HERE, "..", "..", "runs", "report", "e1_report.json")))
rng = np.random.default_rng(0)
out = {}
for cfg in ("CFG-O", "CFG-P"):
    rows = rep[cfg]["table3"]
    off = {(r["alg"], r["clutter"]): r["official"] for r in rep["CFG-O"]["table3"] if "official" in r}     # the authors' column does not depend on our configuration
    n_out = 0
    for r in rows:
        a, lam = r["alg"], r["clutter"]
        if (a, lam) not in off:
            continue
        r = dict(r, official=off[(a, lam)])
        r["official"]["inside_our_ci"] = r["ci"][0] <= r["official"]["G"] <= r["ci"][1]
        n = r["official"]["n"]
        gsq = E.load_cell(ROOT, "e1", cfg, a, 50)[str(lam)]["gospa_sq"]          # [50, 200]
        Gs = np.array([np.mean(np.sqrt(gsq[rng.integers(0, 50, n)].mean(axis=0))) for _ in range(4000)])
        sd = float(Gs.std(ddof=1))
        z = None if sd < 1e-9 else (r["official"]["G"] - r["G"]) / sd
        out[f"{cfg}|{a}|{lam}"] = dict(G_ours=r["G"], G_auth=r["official"]["G"], n_auth=n, sd_n=sd, z=z, star_old=not r["official"]["inside_our_ci"])
        n_out += (z is not None and abs(z) > 2)
    zs = [v["z"] for k, v in out.items() if k.startswith(cfg) and v["z"] is not None]
    print(cfg, "cells with z:", len(zs), "|z|>2:", sum(abs(z) > 2 for z in zs), "|z|>3:", sum(abs(z) > 3 for z in zs), "mean z %.2f rms z %.2f" % (np.mean(zs), np.sqrt(np.mean(np.square(zs)))))
    old = sum(v["star_old"] for k, v in out.items() if k.startswith(cfg))
    print(cfg, "old '*' cells:", old, "of", sum(k.startswith(cfg) for k in out))
json.dump(out, open(os.path.join(HERE, "..", "..", "runs", "report", "authors_zscores.json"), "w"), indent=1)
for k, v in out.items():
    if k.startswith("CFG-O"):
        print(k, "auth %.2f (n=%d) ours %.2f sd_n %.2f z %s" % (v["G_auth"], v["n_auth"], v["G_ours"], v["sd_n"], "n/a" if v["z"] is None else "%+.2f" % v["z"]))
