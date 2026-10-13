"""Run the UNMODIFIED official pomdp-baselines main.py (repo @ e7c19c3) in this isolated venv (torch 1.13.1 CPU,
gym 0.18.0).  Shim (labelled): the repo calls gym.make("Pendulum-v1"), which gym 0.18 does not register; we register
"Pendulum-v1" -> gym 0.18's PendulumEnv, i.e. exactly the Pendulum-v0 dynamics the paper's runs used (README).
Usage: python scripts/official/run_official.py --cfg <yml> --seed N --cuda -1 [other main.py flags]
"""
import os, runpy, sys
ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
REPO = os.path.join(ROOT, "repo")
import gym
from gym.envs.registration import register, registry
if "Pendulum-v1" not in [s.id for s in registry.all()]:
    register(id="Pendulum-v1", entry_point="gym.envs.classic_control:PendulumEnv", max_episode_steps=200)
# Shim 2 (labelled): envs/pomdp/__init__.py builds LunarLander / PyBullet envs at import time (Box2D and pybullet are
# not installed here and are not used by the toy experiments): a stub ``pybullet_envs`` module and a gym.make that
# returns None for those ids only.  The Pendulum / CartPole / meta envs are untouched.
import types
sys.modules.setdefault("pybullet_envs", types.ModuleType("pybullet_envs"))
_orig_make = gym.make
def _make(id, **kw):
    if id.startswith(("LunarLander", "Hopper", "Ant", "Walker", "HalfCheetah", "Humanoid")) and "BLT" not in id:
        try:
            return _orig_make(id, **kw)
        except Exception:
            return None
    return _orig_make(id, **kw)
gym.make = _make
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.argv = [os.path.join(REPO, "policies", "main.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
