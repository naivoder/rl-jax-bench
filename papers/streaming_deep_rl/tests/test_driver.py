"""Full-scale driver features: per-arm seed lists, schedule_steps (truncated view of a longer run) and the 'mark'
NaN policy (a diverging lane is recorded and isolated instead of aborting the program)."""
import json
import os
import shutil

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

from src import agents as ag  # noqa: E402
from src import data as dt  # noqa: E402
from src import train as tr  # noqa: E402


def test_schedule_steps_sets_eps_horizon():
    import jax
    import jax.numpy as jnp
    env = dt.minatar_env("Breakout")
    a = ag.QAgent(env, 5_000_000, overshoot_diag=False)
    l0 = jax.tree.map(lambda x: x[0], tr.stack_lanes("q", [tr.lane_cfg("q", "stream_q")]))
    # 20 % of 5e6 = 1e6 steps: eps after 5e5 steps is half-way between 1 and 0.01; floor after 1e6
    assert abs(float(a.eps(jnp.int32(499_999), l0)) - (1 - 0.99 * 0.5)) < 1e-4
    assert abs(float(a.eps(jnp.int32(1_500_000), l0)) - 0.01) < 1e-6
    # the driver passes schedule_steps (not total_steps) to the agent
    _, agent = tr.build(dict(kind="q", env="Breakout", total_steps=1000, schedule_steps=5_000_000))
    assert agent.T == 5_000_000


def test_mark_policy_isolates_a_diverging_lane():
    tag = "_pytest_full"
    cfg = dict(kind="q", exp="t3", env="Breakout", total_steps=200, chunk_steps=100, bin_stride=100,
               schedule_steps=1000, arms=["stream_q", "classic_q@lr=.nan"], seeds=[1, 2],
               arm_seeds={"classic_q@lr=.nan": [1]}, master_seed=0, overshoot_diag=False, nan_policy="mark")
    try:
        r = tr.run_rl(cfg, tag, ckpt=False)
        assert set(r["final"]["stream_q"]) == {1, 2} and set(r["final"]["classic_q@lr=.nan"]) == {1}
        base = os.path.join(tr.REPO, "runs", tag, "t3_Breakout")
        ok = json.load(open(os.path.join(base, "stream_q", "seed_1", "metrics.json")))
        bad = json.load(open(os.path.join(base, "classic_q@lr=.nan", "seed_1", "metrics.json")))
        assert ok["diverged_by_step"] is None and bad["diverged_by_step"] == 100
        assert np.isfinite(ok["summary"]["auc_bins"]) or ok["summary"]["n_episodes"] >= 0
    finally:
        shutil.rmtree(os.path.join(tr.REPO, "runs", tag), ignore_errors=True)
