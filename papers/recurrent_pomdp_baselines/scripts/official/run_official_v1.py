"""Variant of run_official.py: the repo's CURRENT default env, gym 0.21 ``Pendulum-v1`` (speed clipped BEFORE the
angle update, sin(th) sign convention), loaded from the gym-0.21.0 sdist source in ../gym021 and registered as
"Pendulum-v1" inside this gym 0.18 venv.  Everything else as run_official.py (unmodified main.py).  Labelled shim."""
import importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
p = os.path.join(ROOT, "gym021", "gym-0.21.0", "gym", "envs", "classic_control", "pendulum.py")
spec = importlib.util.spec_from_file_location("pendulum_v1_gym021", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from gym.envs.registration import register, registry
register(id="Pendulum-v1", entry_point=m.PendulumEnv, max_episode_steps=200)
sys.argv = [os.path.join(HERE, "run_official.py")] + sys.argv[1:]
import runpy
runpy.run_path(sys.argv[0], run_name="__main__")
