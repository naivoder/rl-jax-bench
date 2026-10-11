"""Recheck: heuristic table with the authors' first-index tie rule, under both observation tie settings,
on the 100 test inputs and the 2,000 sequences of PRNGKey(4e8) (scripts/review_checks.py), plus the authors' own agents
in the authors' env (unpatched) on the same 100 test inputs.

    JAX_PLATFORMS=cpu python scripts/heuristics_recheck.py  # -> runs/recheck/heuristics.json
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import src  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from src.data import full_sha, lb_input_kwargs, sample_lb_batch, test_key  # noqa: E402
from src.envs import lb_heuristic_action, lb_queue_counts, lb_reset, lb_step, make_lb_spec  # noqa: E402
from src.train import load_config  # noqa: E402

NAMES = ("spt_official", "lw_official", "jsq_first", "spt_fastest_tie", "jsq_fastest_tie", "eft", "lrt")
cfg0 = load_config(os.path.join(ROOT, "src/configs/load_balance.yaml"))
kw = lb_input_kwargs(cfg0)
scale = float(cfg0["reward_scale"])


def summarise(x):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return {"mean": float(x.mean()), "se": float(se), "t95": [float(x.mean() - h), float(x.mean() + h)],
            "std": float(x.std(ddof=1)), "n": len(x)}


out = {"paper_heuristic_digitised": -54.03}
sets = {"test_set": sample_lb_batch(test_key(), 100, **kw),
        "fresh_2000_key4e8": sample_lb_batch(jax.random.PRNGKey(400_000_000), 2000, **kw)}
out["test_set_sha256"] = full_sha(sets["test_set"])[:16]
for tie in ("strict", "official"):
    spec = make_lb_spec(dict(cfg0, obs_tie=tie))
    for nm in NAMES:
        def roll(inp, nm=nm):
            def body(st, _):
                now = inp.arrivals[st.k]
                ne = jnp.mean((lb_queue_counts(spec, st, now) > 0).astype(jnp.float32))
                a = lb_heuristic_action(spec, nm, st, inp)
                st2, r, _ = lb_step(spec, st, a, inp)
                return st2, (r, ne)
            _, (r, ne) = jax.lax.scan(body, lb_reset(spec), None, length=spec.n_jobs)
            return jnp.sum(r) * scale, jnp.mean(ne)
        f = jax.jit(jax.vmap(roll))
        for sname, inp in sets.items():
            ret, ne = f(inp)
            row = summarise(np.asarray(ret))
            row["frac_nonempty_servers_at_arrivals"] = float(np.asarray(ne).mean())
            if sname == "test_set":
                row["per_episode"] = [float(v) for v in np.asarray(ret)]
            out[f"{sname}/{tie}/{nm}"] = row
            print(sname, tie, nm, {k: v for k, v in row.items() if k != "per_episode"}, flush=True)

# the authors' agents in the authors' env (float64 durations, unpatched) on the same 100 test inputs
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_official_env as T  # noqa: E402

if T.OFF is not None:
    ti = sets["test_set"]
    for nm, agent in (("official_env/ShortestProcessingTimeAgent", T.OFF[1].ShortestProcessingTimeAgent()),
                      ("official_env/LeastWorkAgent", T.OFF[1].LeastWorkAgent())):
        rets = []
        for i in range(100):
            R = T.run_official(np.asarray(ti.arrivals[i]), np.asarray(ti.sizes[i]), agent.get_action, exact=False)[0]
            rets.append(R.sum() * scale)
        out[f"test_set/{nm}"] = summarise(rets)
        out[f"test_set/{nm}"]["per_episode"] = [float(v) for v in rets]
        print(nm, {k: v for k, v in out[f"test_set/{nm}"].items() if k != "per_episode"}, flush=True)
os.makedirs(os.path.join(ROOT, "runs/recheck"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "runs/recheck/heuristics.json"), "w"), indent=1)
