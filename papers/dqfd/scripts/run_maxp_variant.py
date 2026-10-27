"""Priority-convention sensitivity: new agent transitions enter at the CURRENT buffer maximum priority (what a sum-tree max would give)
instead of the running maximum over all updates (src/replay.py, the convention of openai/baselines and DI-engine PER).
Done by monkeypatching `replay.update_priorities` in THIS process only; src/ is untouched, so the main-run numbers keep their code.
usage: python scripts/run_maxp_variant.py --curmax -- <src.train arguments>     (omit --curmax for the unpatched control)"""
import sys
args = sys.argv[1:]
cur = "--curmax" in args
args = [a for a in args if a not in ("--curmax", "--")]
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from src import replay as R  # noqa: E402  (imports jax lazily; gpu bootstrap runs in src.train)
import src.train as T  # noqa: E402,F401

if cur:
    import jax.numpy as jnp
    _orig = R.update_priorities

    def update_priorities_curmax(L, buf, idx, td_abs, valid, hp):
        b = _orig(L, buf, idx, td_abs, valid, hp)
        b["maxp"] = jnp.max(b["pa"]) ** (1.0 / L.alpha)     # current maximum over the stored p^alpha, back to p
        return b

    R.update_priorities = update_priorities_curmax
    print("[INFO] priority variant: new transitions enter at the CURRENT max priority", flush=True)
else:
    print("[INFO] priority control: running-max priority (the shipped code)", flush=True)
T.main(args)
