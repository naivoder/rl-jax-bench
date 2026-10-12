"""Env-level differential: our brax/MJX Hopper/Walker2d proxy vs gymnasium MuJoCo v4 on IDENTICAL states and actions.

The official code runs gym *-v2 on mujoco-py, which cannot be installed here. Gymnasium's v4 envs (same XML lineage,
same reward and termination definitions, RK4 integrator, MuJoCo 3) are the closest runnable stand-in, and the
official training code itself was run on them (official_ref_run.py). Transitions are recorded in the oracle venv by
scripts/official/record_env.py (random uniform actions; 2000 transitions; resets on
termination).

Two checks per recorded transition (s_t, a_t, s_{t+1}^gym):
  (1) DEFINITION parity: brax's own step() is run with its physics replaced by the recorded gym next state, so
      obs/reward/done are computed by OUR env code on gym's states -> must equal gym's obs/reward/terminated.
      Tolerance: obs 1e-5 abs, reward 1e-4 abs, done exact.
  (2) DYNAMICS gap (measured, not a pass/fail): brax's real MJX step from the same state and action; the
      one-step next-obs error and reward error are reported (Euler + 4 solver iterations + brax contact
      settings vs RK4 + gym XML; PROXY). Under --env_kinematics legacy the reward error also contains the artefact (brax State.x is the pre-integration position of the last substep); post_step removes it.
gym qpos[1] (rootz with ref=1.25) = brax q[1] + 1.25.

Usage: JAX_PLATFORMS=cpu python scripts/env_parity.py [--fixture N]   (writes runs/oracle/env_parity.json;
       --fixture N also writes tests/fixtures/official/env_<name>.npz with the first N transitions)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (bootstrap first)
from src.common import info, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data as D  # noqa: E402

REC = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "env_ref")
FIX = os.path.join(common.REPO, "tests", "fixtures", "official")
NAMES = {"hopper": "Hopper-v4", "walker2d": "Walker2d-v4"}


def to_brax_q(qpos):
    q = np.array(qpos, np.float32)
    q[..., 1] -= 1.25
    return q


KIN = [False]


def replay(env_name, d):
    env, _ = D.make_env(env_name, "mjx", KIN[0])
    st_template = env.reset(jax.random.PRNGKey(0))
    real_step = env.pipeline_step

    def one(q, qd, a, q1, qd1):
        ps = env.pipeline_init(q, qd)
        st = st_template.replace(pipeline_state=ps, obs=env._get_obs(ps))
        obs0 = st.obs
        # (1) definition parity: physics replaced by the recorded gym next state
        env.pipeline_step = lambda ps0, act: env.pipeline_init(q1, qd1)
        st_def = env.step(st, a)
        # (2) real MJX dynamics
        env.pipeline_step = real_step
        st_dyn = env.step(st, a)
        return obs0, st_def.obs, st_def.reward, st_def.done, st_dyn.obs, st_dyn.reward, st_dyn.done

    f = jax.jit(jax.vmap(one))
    out = f(jnp.asarray(to_brax_q(d["qpos"])), jnp.asarray(d["qvel"], jnp.float32), jnp.asarray(d["act"]),
            jnp.asarray(to_brax_q(d["qpos1"])), jnp.asarray(d["qvel1"], jnp.float32))
    return [np.asarray(x, np.float64) for x in out]


def compare(env_name, d):
    obs0, o_def, r_def, dn_def, o_dyn, r_dyn, dn_dyn = replay(env_name, d)
    term = d["term"].astype(bool)
    res = {
        "n_transitions": int(len(term)), "n_terminations_gym": int(term.sum()),
        "def_obs0_max_abs": float(np.max(np.abs(obs0 - d["obs"]))),
        "def_obs1_max_abs": float(np.max(np.abs(o_def - d["obs1"]))),
        "def_reward_max_abs": float(np.max(np.abs(r_def - d["rew"]))),
        "def_done_mismatches": int(np.sum((dn_def > 0.5) != term)),
        "dyn_obs1_abs_err_median": float(np.median(np.abs(o_dyn - d["obs1"]))),
        "dyn_obs1_abs_err_p95": float(np.percentile(np.abs(o_dyn - d["obs1"]), 95)),
        "dyn_obs1_rel_err_median": float(np.median(np.linalg.norm(o_dyn - d["obs1"], axis=1)
                                                   / np.linalg.norm(d["obs1"], axis=1))),
        "dyn_obs1_err_pos_vs_vel_median": [float(np.median(np.abs(o_dyn - d["obs1"])[:, : o_dyn.shape[1] // 2])),
                                           float(np.median(np.abs(o_dyn - d["obs1"])[:, o_dyn.shape[1] // 2:]))],
        "dyn_reward_abs_err_median": float(np.median(np.abs(r_dyn - d["rew"]))),
        "dyn_reward_abs_err_p95": float(np.percentile(np.abs(r_dyn - d["rew"]), 95)),
        "dyn_done_mismatches": int(np.sum((dn_dyn > 0.5) != term)),
    }
    res["definition_pass"] = bool(res["def_obs0_max_abs"] < 1e-5 and res["def_obs1_max_abs"] < 1e-5
                                  and res["def_reward_max_abs"] < 1e-4 and res["def_done_mismatches"] == 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=int, default=0)
    ap.add_argument("--env_kinematics", default="legacy", choices=["legacy", "post_step"],
                    help="recheck: legacy = pre-fix brax State.x (one substep behind); post_step = fixed")
    ap.add_argument("--out", default=os.path.join("runs", "oracle", "env_parity.json"))
    args = ap.parse_args()
    KIN[0] = args.env_kinematics == "post_step"
    rows = {}
    for env_name, gname in NAMES.items():
        d = dict(np.load(os.path.join(REC, f"{gname}.npz")))
        rows[env_name] = compare(env_name, d)
        rows[env_name]["gym_env"] = gname
        rows[env_name]["mujoco_gym_side"] = str(d["mujoco_version"])
        info("env_parity", env=env_name, **{k: v for k, v in rows[env_name].items() if not isinstance(v, list)})
        if args.fixture:
            os.makedirs(FIX, exist_ok=True)
            keep = ["qpos", "qvel", "obs", "act", "qpos1", "qvel1", "obs1", "rew", "term"]
            np.savez_compressed(os.path.join(FIX, f"env_{env_name}.npz"),
                                **{k: d[k][: args.fixture] for k in keep})
    ok = all(r["definition_pass"] for r in rows.values())
    write_json(os.path.join(common.REPO, args.out),
               {"definition_pass": ok, "env_kinematics": args.env_kinematics, "rows": rows, **common.git_info()})
    print(json.dumps(rows, indent=1))
    print("DEFINITION PASS" if ok else "DEFINITION FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
