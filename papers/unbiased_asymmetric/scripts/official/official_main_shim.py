"""Run the official main_a2c.py unmodified, but (1) stub gym's headless-incompatible pyglet renderer (imported at
module level by car_flag.py, used only for rendering) and (2) also print every wandb.log payload as a JSON line
(wandb runs offline) so our reference curves can be read without the wandb service."""
import json
import os
import runpy
import sys
import types

PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
sys.modules["gym.envs.classic_control.rendering"] = types.ModuleType("rendering")
from wandb.sdk import wandb_run  # noqa: E402

_orig = wandb_run.Run.log   # wandb.init() rebinds the module-level wandb.log to run.log -> patch the method


def _log(self, d, *a, **k):
    print("WANDB " + json.dumps({kk: (float(v) if hasattr(v, "__float__") else v) for kk, v in d.items()}), flush=True)
    return _orig(self, d, *a, **k)


wandb_run.Run.log = _log
sys.argv = ["main_a2c.py"] + sys.argv[1:]
runpy.run_path(os.path.join(OFFICIAL_ROOT, "asym-rlpo-aamas2022", "main_a2c.py"), run_name="__main__")
