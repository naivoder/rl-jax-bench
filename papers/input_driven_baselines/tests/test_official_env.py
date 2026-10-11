"""Author-code oracle (recheck): replay identical job sequences and actions through the AUTHORS' event-driven
load-balancing env (hongzimao/input_driven_rl_example, commit 6c9cc23, located by $OFFICIAL_ROOT, see the README) and through
our closed-form JAX simulator. Skipped when the official checkout is absent.

Known, documented differences:
  * durations: the authors use float64 ``int(size / rate)``, which is 1 below the exact floor at some exact multiples;
    the test patches the authors' Worker to the exact floor so rewards must then agree EXACTLY;
  * observation at same-time events: the authors pop an arrival before a same-time completion, so the job about to
    start is still queued; ``obs_tie: official`` mirrors that, ``obs_tie: strict`` (pre-fix) does not.

    python -m tests.test_official_env # full measurement -> runs/recheck/official_env_diff.json
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

OFFICIAL = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "input_driven_rl_example")
RATES = [float(r) for r in np.round(np.linspace(0.15, 1.05, 10), 2)]
PAPER_ARGS = ["--num_workers", "10", "--service_rates", *map(str, RATES), "--job_distribution", "pareto",
              "--job_size_pareto_shape", "1.5", "--job_size_pareto_scale", "100", "--job_interval", "55",
              "--num_stream_jobs", "500"]


def _import_official():
    if not os.path.isdir(os.path.join(OFFICIAL, "load_balance")):
        return None
    argv, path = sys.argv, list(sys.path)
    try:
        sys.argv = ["official"] + PAPER_ARGS
        sys.path.insert(0, OFFICIAL)
        import load_balance.env as offenv  # noqa: E402
        import load_balance.heuristic_agents as offheur  # noqa: E402
        import load_balance.worker as offworker  # noqa: E402
    finally:
        sys.argv, sys.path[:] = argv, path
    return offenv, offheur, offworker


OFF = _import_official()
pytestmark = pytest.mark.skipif(OFF is None, reason="needs the authors' clone at $OFFICIAL_ROOT/input_driven_rl_example: run sh scripts/official/setup.sh")


def _exact_process(self):
    """The authors' Worker.process with the exact integer floor instead of float64 int(size/rate)."""
    if (self.curr_job is None or self.curr_job.finish_time <= self.wall_time.curr_time) and len(self.queue) > 0:
        self.curr_job = self.queue.pop(0)
        duration = (int(self.curr_job.size) * 100) // int(round(self.service_rate * 100))
        self.curr_job.start_time = self.wall_time.curr_time
        self.curr_job.finish_time = self.wall_time.curr_time + duration
        return self.curr_job
    return None


def official_sequence(seed):
    offenv = OFF[0]
    np.random.seed(seed)
    t, s = offenv.generate_jobs(500)
    return np.asarray(t, np.int64), np.asarray(s, np.int64)


def run_official(t, s, policy, exact=True):
    """policy(state) -> action.  Returns (rewards, actions, queued work per server at each decision, done flags)."""
    offenv, _, offworker = OFF
    orig = offworker.Worker.process
    if exact:
        offworker.Worker.process = _exact_process
    try:
        class SeqEnv(offenv.LoadBalanceEnvironment):
            def generate_jobs(self):
                for tt, ss in zip(t.tolist(), s.tolist()):
                    self.timeline.push(int(tt), int(ss))

        env = SeqEnv()
        env.reset()
        st, done = env.observe(), False
        R, A, Q, D = [], [], [], []
        while not done:
            Q.append([float(sum(j.size for j in w.queue)) for w in st[0]])
            a = int(policy(st))
            st, r, done = env.step(a)
            R.append(r)
            A.append(a)
            D.append(done)
    finally:
        offworker.Worker.process = orig
    return np.asarray(R, np.int64), np.asarray(A), np.asarray(Q), np.asarray(D)


def _ours(obs_tie):
    import jax
    import jax.numpy as jnp

    import src  # noqa: F401
    from src.data import LBInputs
    from src.envs import lb_heuristic_rollout, lb_now, lb_queued_work, lb_reset, lb_step, make_lb_spec

    spec = make_lb_spec(dict(service_rates=RATES, jobs_per_episode=500, obs_tie=obs_tie))

    @jax.jit
    def replay(t, s, acts):
        inp = LBInputs(t, s)

        def body(st, a):
            q = lb_queued_work(spec, st, inp, lb_now(st, inp))
            st2, r, d = lb_step(spec, st, a, inp)
            return st2, (r, q, d)

        _, out = jax.lax.scan(body, lb_reset(spec), acts)
        return out

    heur = jax.jit(lambda t, s, nm: lb_heuristic_rollout(spec, nm, LBInputs(t, s)), static_argnums=2)
    to = lambda x: jnp.asarray(x, jnp.int32)  # noqa: E731
    return (lambda t, s, a: [np.asarray(v) for v in replay(to(t), to(s), to(a))],
            lambda t, s, nm: np.asarray(heur(to(t), to(s), nm)).astype(np.int64))


def measure(n_eps=10, seed0=100_000_000):
    rng = np.random.default_rng(0)
    spt = OFF[1].ShortestProcessingTimeAgent()
    out = {"episodes": n_eps}
    for tie in ("strict", "official"):
        replay, heur = _ours(tie)
        c = dict(steps=0, reward_mismatch_steps=0, done_mismatch_eps=0, obs_mismatch_steps=0, obs_max_abs_size=0.0,
                 spt_return_equal_eps=0, spt_max_abs_return_diff_raw=0)
        for ep in range(n_eps):
            t, s = official_sequence(seed0 + ep)
            for kind in ("random", "spt"):
                pol = (lambda st: rng.integers(10)) if kind == "random" else spt.get_action
                R, A, Q, D = run_official(t, s, pol)
                r, q, d = replay(t, s, A)
                c["steps"] += len(A)
                c["reward_mismatch_steps"] += int((r.astype(np.int64) != R).sum())
                c["done_mismatch_eps"] += int(not np.array_equal(d, D))
                diff = np.abs(q - Q)
                c["obs_mismatch_steps"] += int((diff.max(1) > 0).sum())
                c["obs_max_abs_size"] = max(c["obs_max_abs_size"], float(diff.max()))
                if kind == "spt":
                    rh = heur(t, s, "spt_official")
                    c["spt_return_equal_eps"] += int(rh.sum() == R.sum())
                    c["spt_max_abs_return_diff_raw"] = max(c["spt_max_abs_return_diff_raw"], int(abs(rh.sum() - R.sum())))
        out[tie] = c
    # unpatched authors' durations (float64 int(size/rate)) against ours: the only reward difference
    replay, _ = _ours("official")
    tot = mis = 0
    rel = []
    for ep in range(n_eps):
        t, s = official_sequence(seed0 + ep)
        R, A, _, _ = run_official(t, s, lambda st: rng.integers(10), exact=False)
        r = replay(t, s, A)[0].astype(np.int64)
        tot += len(R)
        mis += int((r != R).sum())
        rel.append(abs(r.sum() - R.sum()) / abs(R.sum()))
    out["float64_durations"] = {"steps": tot, "reward_mismatch_steps": mis, "max_rel_return_diff": float(max(rel))}
    return out


def test_rewards_dones_exact_and_official_obs_tie():
    m = measure(n_eps=4)
    for tie in ("strict", "official"):
        assert m[tie]["reward_mismatch_steps"] == 0 and m[tie]["done_mismatch_eps"] == 0
    assert m["official"]["obs_mismatch_steps"] <= 0.002 * m["official"]["steps"]
    assert m["official"]["obs_mismatch_steps"] < m["strict"]["obs_mismatch_steps"]
    assert m["official"]["spt_return_equal_eps"] >= 3  # the authors' SPT agent == our spt_official
    assert m["float64_durations"]["max_rel_return_diff"] < 1e-3


if __name__ == "__main__":
    import json

    res = measure(n_eps=int(os.environ.get("N_EPS", 50)))
    print(json.dumps(res, indent=1))
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "recheck")
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(os.path.join(d, "official_env_diff.json"), "w"), indent=1)
