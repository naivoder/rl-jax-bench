"""Write tests/fixtures/official/gradstats.npz: official-primitive gradient estimates (official_grad_ref.py output)
and the pairwise / to-true cosines computed with sklearn.metrics.pairwise.cosine_similarity (oracle venv).
Usage: .venv/bin/python -I make_gradstats_fixture.py grad_ref/hopper_ppo_s0.npz <it> <out.npz>"""
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

src, it, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = np.load(src)
sizes = [100, 10000]
res = {"sizes": np.asarray(sizes), "g_ref": d[f"it{it}_ref"]}
for n in sizes:
    G = d[f"it{it}_n{n}"].astype(np.float64)
    C = cosine_similarity(G)
    iu = np.triu_indices(G.shape[0], 1)
    res[f"G_{n}"] = d[f"it{it}_n{n}"]
    res[f"pairwise_{n}"] = C[iu].mean()
    res[f"to_true_{n}"] = cosine_similarity(G, d[f"it{it}_ref"].astype(np.float64)[None])[:, 0].mean()
np.savez_compressed(out, **res)
print({k: float(v) for k, v in res.items() if k.startswith(("pairwise", "to_true"))})
