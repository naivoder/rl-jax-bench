"""Loader for the authors' code (read-only copy, commit b5959f6) used ONLY as a differential-test oracle; never imported by src/.

The official modules import each other by bare names (Parameters, utils, ...) so the repo directory is put on sys.path for the
test process.  Returns None when the copy is absent (tests then skip with SKIP).
"""
import os, sys, importlib

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
REPO = os.path.join(OFFICIAL_ROOT, "repo")
SKIP = "authors' code absent: run `sh scripts/official/setup.sh` (or set OFFICIAL_ROOT)"
_MODS = None


def load():
    global _MODS
    if _MODS is not None:
        return _MODS
    if not os.path.isdir(REPO):
        return None
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import matplotlib
    matplotlib.use("Agg")
    names = ["Parameters", "obstacles", "utils", "lbp", "tomb", "predict", "update", "MCTS", "MCTS_joint", "sensor", "gospa"]
    mods = {n: importlib.import_module(n) for n in names}
    P = mods["Parameters"]
    P.obstacles_list = [mods["obstacles"].obstacle_three(P.coordD3, P.coordB3)]
    _MODS = mods
    return mods
