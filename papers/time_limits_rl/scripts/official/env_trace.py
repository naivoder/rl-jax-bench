"""Oracle harness: step the env the authors' library builds -- Tonic @0e20c89
environments.Gym(<name>, terminal_timeouts=False) = gym.make(<name>) with the TimeLimit stripped + Tonic's
ActionRescaler ([-1, 1] -> true action range, with clipping) -- and record obs / rewards / dones.

Run with .venv-ref (Python 3.10, gym 0.24.1, mujoco 2.2.0, numpy 1.23.5, torch 1.13.1+cpu):
  .venv-ref/bin/python -I scripts/official/env_trace.py <in.npz> <out.npz>
Input keys: names (gym ids); for case i:
  free-run : q0_<i>, qd0_<i>, act_<i>        -> fr_obs_<i>, fr_rew_<i>, fr_done_<i>   (set state once, step N actions)
  resync   : qpre_<i>, qdpre_<i>, act_<i>    -> rs_obs_<i>, rs_rew_<i>, rs_done_<i>  (set the GIVEN pre-state before
                                                                                   every step: one-step comparison)
Positions are given in the gym model's coordinates (callers convert, e.g. Brax hopper rootz is offset by 1.25).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("OFFICIAL_ROOT", "external/official"), "tonic"))  # Tonic clone
import gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import tonic  # noqa: E402


def main(inp, out):
    z = dict(np.load(inp))
    res = {}
    for i, name in enumerate(z["names"]):
        env = tonic.environments.Gym(str(name), terminal_timeouts=False)
        env.reset()
        acts = z[f"act_{i}"]
        env.unwrapped.set_state(z[f"q0_{i}"].astype(np.float64), z[f"qd0_{i}"].astype(np.float64))
        fr = [env.step(np.asarray(a, np.float32))[:3] for a in acts]
        res[f"fr_obs_{i}"] = np.array([f[0] for f in fr])
        res[f"fr_rew_{i}"] = np.array([f[1] for f in fr])
        res[f"fr_done_{i}"] = np.array([f[2] for f in fr])
        rs = []
        for k, a in enumerate(acts):
            env.unwrapped.set_state(z[f"qpre_{i}"][k].astype(np.float64), z[f"qdpre_{i}"][k].astype(np.float64))
            rs.append(env.step(np.asarray(a, np.float32))[:3])
        res[f"rs_obs_{i}"] = np.array([f[0] for f in rs])
        res[f"rs_rew_{i}"] = np.array([f[1] for f in rs])
        res[f"rs_done_{i}"] = np.array([f[2] for f in rs])
        res[f"init_qpos_{i}"] = np.array(env.unwrapped.init_qpos)
        res[f"init_qvel_{i}"] = np.array(env.unwrapped.init_qvel)
        res[f"dt_{i}"] = np.array(env.unwrapped.dt)
        res[f"timestep_{i}"] = np.array(env.unwrapped.model.opt.timestep)
        res[f"integrator_{i}"] = np.array(int(env.unwrapped.model.opt.integrator))
        res[f"frame_skip_{i}"] = np.array(env.unwrapped.frame_skip)
    res["meta"] = np.array(f"gym {gym.__version__} mujoco {mujoco.__version__} numpy {np.__version__}")
    np.savez(out, **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
