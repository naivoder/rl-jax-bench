"""Review: the second declared gap definition (mean of per-replication gaps) next to the ratio of means, Fig 6 (and Fig 5), vs the authors' vector points."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import numpy as np, eval as E
cells, _ = E.load(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs'), 'full_figs'))
L = ["| fig | K | policy | ratio of means (primary) | mean of per-replication gaps +- SE | authors' point | n |", "|---|---|---|---|---|---|---|"]
mx = {}
for fig in ("fig6", "fig5a", "fig5b", "fig5c"):
    for cid, c in sorted(cells.items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("K=")[1].split("|")[0]))):
        if not cid.startswith(fig + "|") or not cid.endswith("|primary") or "V_D" not in c: continue
        K = int(cid.split("K=")[1].split("|")[0]); VD = c["V_D"]["lb"]
        try: auth = E.paper_gap(fig, K)
        except KeyError: auth = None
        for pol in E.POL_NAMES:
            if pol not in c["J"]: continue
            J = c["J"][pol]; g = 100 * (J - VD) / VD; rom = 100 * (J.mean() - VD.mean()) / VD.mean()
            mx[(fig, pol)] = max(mx.get((fig, pol), 0), abs(rom - g.mean()))
            if fig == "fig6" or K == 2:
                L.append(f"| {fig} | {K} | {pol} | {rom:.3f} | {g.mean():.3f} +- {g.std(ddof=1) / np.sqrt(len(g)):.3f} | {'' if auth is None else format(auth[pol], '.3f')} | {len(g)} |")
L.append("")
L.append("max |ratio of means - mean of per-replication gaps| over all K (pp): " + "; ".join(f"{f}/{p}: {v:.3f}" for (f, p), v in sorted(mx.items())))
t = "\n".join(L); print(t)
os.makedirs(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs'), 'tables'), exist_ok=True)
open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs'), 'tables', 'review_fig6_meanof.md'), 'w').write(t + "\n")
