"""Build tests/fixtures/gym_v4/*.npz: OUR env's initial states + action sequences, and the outputs of the env the
authors' library builds (Tonic @0e20c89 environments.Gym -> gym 0.24.1 InvertedPendulum-v4 / Hopper-v4 on mujoco
2.2.0, Tonic's ActionRescaler) on them. Runs the oracle in its isolated venv
($OFFICIAL_ROOT/.venv-ref) through scripts/official/env_trace.py; only NUMBERS are committed.

Two of our envs are traced against the same oracle: the MJX gym-v4 port (src.data.gymv4_env) and the small-scale Brax
proxy (src.data.brax_env, generalized backend) -- the latter to quantify the proxy gap, not as a parity claim.
Usage: uv run python scripts/oracle/make_env_fixtures.py
"""
import json
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data as D  # noqa: E402

OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official"))  # authors' clones + oracle venvs
HARNESS = os.path.join(REPO, "scripts", "official")
FIX = os.path.join(REPO, "tests", "fixtures", "gym_v4")
CASES = [("inverted_pendulum", "InvertedPendulum-v4", 50, 0.0), ("hopper", "Hopper-v4", 50, 1.25)]
# (our name, gym id, steps, Brax-hopper rootz offset: Brax re-places rootz at the torso, gym z is absolute)


def trace(env, key, acts, qoff):
    st, _ = env.reset(key)
    step = jax.jit(env.step)
    qpre, qdpre, obs, rew, term = [], [], [], [], []
    for k, a in enumerate(acts):
        ps = st.pipeline_state if hasattr(st, "pipeline_state") else st
        qpre.append(np.asarray(ps.q if hasattr(ps, "q") else ps.qpos, np.float64) + qoff)
        qdpre.append(np.asarray(ps.qd if hasattr(ps, "qd") else ps.qvel, np.float64))
        st, o, r, t = step(jax.random.PRNGKey(k), st, jnp.asarray(a))
        obs.append(np.asarray(o)); rew.append(float(r)); term.append(bool(t))
    return dict(qpre=np.array(qpre), qdpre=np.array(qdpre), obs=np.array(obs), rew=np.array(rew),
                term=np.array(term))


def main():
    os.makedirs(FIX, exist_ok=True)
    rng = np.random.default_rng(20261014)
    for kind in ("mjx_port", "brax_proxy"):
        inp = {"names": np.array([c[1] for c in CASES])}
        ours = {}
        for i, (name, gid, n, zoff) in enumerate(CASES):
            env = D.gymv4_env(name) if kind == "mjx_port" else D.brax_env(name, "generalized")
            acts = np.clip(rng.normal(0.0, 0.5, (n, env.n_out)), -1.3, 1.3).astype(np.float32)  # incl. clipping
            qoff = np.zeros(1)
            st0, _ = env.reset(jax.random.PRNGKey(100 + i))
            ps0 = st0.pipeline_state if hasattr(st0, "pipeline_state") else st0
            nq = (ps0.q if hasattr(ps0, "q") else ps0.qpos).shape[0]
            qoff = np.zeros(nq); qoff[1] = zoff if kind == "brax_proxy" else 0.0
            tr = trace(env, jax.random.PRNGKey(100 + i), acts, qoff)
            inp[f"q0_{i}"], inp[f"qd0_{i}"], inp[f"act_{i}"] = tr["qpre"][0], tr["qdpre"][0], acts
            inp[f"qpre_{i}"], inp[f"qdpre_{i}"] = tr["qpre"], tr["qdpre"]
            for k, v in tr.items():
                ours[f"{k}_{i}"] = v
        ip = os.path.join(FIX, f"{kind}_in.npz"); op = os.path.join(FIX, f"{kind}_oracle.npz")
        np.savez(ip, **inp)
        np.savez(os.path.join(FIX, f"{kind}_ours_at_record.npz"), **ours)
        subprocess.run([os.path.join(OFF, ".venv-ref", "bin", "python"), "-I", "-W", "ignore",
                        os.path.join(HARNESS, "env_trace.py"), ip, op], check=True)
        print("[INFO] wrote", ip, op)
    commit = subprocess.check_output(["git", "-C", os.path.join(OFF, "tonic"), "rev-parse", "HEAD"]).decode().strip()
    with open(os.path.join(FIX, "provenance.json"), "w") as f:
        json.dump(dict(oracle="Tonic environments.Gym over gym 0.24.1 MuJoCo-v4 (mujoco 2.2.0)", commit=commit,
                       harness=os.path.join(HARNESS, "env_trace.py"), venv=os.path.join(OFF, ".venv-ref"),
                       cases=[c[1] for c in CASES], ours=["src.data.gymv4_env", "src.data.brax_env(generalized)"]),
                  f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
