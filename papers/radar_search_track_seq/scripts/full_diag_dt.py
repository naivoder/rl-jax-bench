"""full-scale DIVERGENT-cause test (diagnostic, NOT adopted; CPU): do the fixed-policy ratios to Random (Coverage/Random and
Static/Random on M_cov; SR ratios) depend on the time step? The companion paper's Table IV gives Ts = 5 ms; this paper's
text gives 50 ms (README, the declared primary). Runs Static/Random/Coverage/hand on the first n small-scale
scenarios (0x6A72, never the final-evaluation set) at the requested dt, everything else unchanged.

    JAX_PLATFORMS=cpu python scripts/full_diag_dt.py --dt 0.005 --n 32
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import EnvConfig, K_SMALL  # noqa: E402
from src import baselines as B, eval as E  # noqa: E402
sys.path.insert(0, os.path.join(C.REPO, "scripts"))
from full_baselines import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--extra", default="", help="k=v,k=v EnvConfig overrides (diagnostic)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--deleter", default="trace", choices=["trace", "frob"],
                    help="frob = this paper's text ('Frobenius norm of the target covariance above a set threshold', "
                         "see README) instead of the library trace; patched at runtime, src/data.py untouched")
    args = ap.parse_args()
    if args.deleter == "frob":
        import types
        from src import data as D
        assert D.__dict__["jnp"].trace is not None
        proxy = types.SimpleNamespace(**{k: getattr(D.jnp, k) for k in dir(D.jnp) if not k.startswith("__")})
        proxy.trace = lambda P, axis1, axis2: jax.numpy.sqrt(jax.numpy.sum(P ** 2, axis=(axis1, axis2)))
        D.jnp = proxy
        C.warn("DIAGNOSTIC: deleter statistic patched to the Frobenius norm (data.py line 292 is the only jnp.trace use)")
    C.check_device()
    kw = dict(dt=args.dt)
    for kv in filter(None, args.extra.split(",")):
        k, v = kv.split("=")
        kw[k] = type(getattr(EnvConfig(), k))(v)
    cfg = replace(EnvConfig(), **kw)
    sk = E.scenario_keys(jax.random.PRNGKey(15), K_SMALL, args.n)
    res = {}
    for name, (pi, po) in dict(static=B.make_static(cfg), random=B.make_random(cfg), coverage=B.make_coverage(cfg),
                               hand=B.make_hand_rule(cfg, 12, 4500.0)).items():
        out = run(pi, po, sk, cfg, 0)
        r = E.reduce_episodes(out)
        res[name] = {k: r[k] for k in ("sr", "m_cov", "gospa_d", "gospa_miss", "mean_tracks", "n_init", "life_median",
                                       "n_episodes_no_track")}
        C.metric("diag", arm=name, **res[name])
    rr = {f"{k}:{a}": res[a][k] / res["random"][k] for a in ("static", "coverage", "hand") for k in ("sr", "m_cov", "gospa_d")}
    C.metric("diag ratios", **rr)
    tag = args.tag or f"dt{args.dt}"
    C.write_json(os.path.join(C.REPO, "runs", "full", f"diag_{tag}.json"), dict(cfg=kw, n=args.n, res=res, ratios=rr))


if __name__ == "__main__":
    main()
