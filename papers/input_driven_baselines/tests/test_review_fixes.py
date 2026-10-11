"""Tests added when fixing issues found in review.

    JAX_PLATFORMS=cpu python -m pytest tests/test_review_fixes.py -q
"""
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401  (bootstrap before jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import src.train as T  # noqa: E402
from src.data import lb_input_kwargs, sample_lb_batch  # noqa: E402
from src.envs import lb_heuristic_rollout, lb_rollout, make_lb_spec  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "configs")
TINY = ["jobs_per_episode=20", "n_envs=4", "mv_groups=2", "mv_k=2", "meta_k=4", "n_value_nets=3",
        "test_episodes=3", "hidden=[8,4]", "pg_var_every=1", "eval_every=1000"]


def _cfg(extra=()):
    return T.load_config(os.path.join(CFG_DIR, "load_balance.yaml"), TINY + list(extra))


def test_resume_refuses_a_different_config(tmp_path):
    """a checkpoint written under one config must not be resumed under another; extending iterations is fine."""
    T.run(_cfg(["iterations=2"]), "state", 0, str(tmp_path))
    with pytest.raises(RuntimeError, match="refusing to resume"):
        T.run(_cfg(["iterations=4", "gamma=0.9"]), "state", 0, str(tmp_path))
    blob = T.run(_cfg(["iterations=4"]), "state", 0, str(tmp_path))  # same config, longer run: resumes
    assert blob["final"]["it"] == 4 and len(blob["history"]) == 4


def test_metrics_json_is_written_at_every_checkpoint(tmp_path, monkeypatch):
    """after a crash right after a checkpoint, metrics.json holds every history row the checkpoint covers."""
    calls = {"n": 0}
    real = T.save_ckpt

    def save_then_crash(mgr, ts):
        real(mgr, ts)
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated crash")

    monkeypatch.setattr(T, "save_ckpt", save_then_crash)
    with pytest.raises(KeyboardInterrupt):
        T.run(_cfg(["iterations=6", "ckpt_minutes=0"]), "state", 0, str(tmp_path))  # checkpoint every iteration
    with open(os.path.join(tmp_path, "state", "seed_0", "metrics.json")) as f:
        m = json.load(f)
    assert [h["it"] for h in m["history"]] == [0, 1]  # checkpoint at it = 2 covers iterations 0 and 1
    monkeypatch.setattr(T, "save_ckpt", real)
    blob = T.run(_cfg(["iterations=6", "ckpt_minutes=0"]), "state", 0, str(tmp_path))
    assert [h["it"] for h in blob["history"]] == list(range(6))


def test_two_server_obs_has_no_job_size():
    """: paper §3 observes s_t = (q1, q2); the load-balancing obs keeps (q_1..q_10, j) (App. J)."""
    two = T.load_config(os.path.join(CFG_DIR, "two_server.yaml"), ["jobs_per_episode=20"])
    lb = _cfg()
    for cfg, dim in ((two, 2), (lb, 11)):
        spec = make_lb_spec(cfg)
        assert spec.obs_dim == dim and spec.critic_dim == dim + 1
        inp = sample_lb_batch(jax.random.PRNGKey(0), 1, **lb_input_kwargs(cfg))
        tr = lb_rollout(spec, lambda o: jnp.zeros(spec.n_servers), jax.tree_util.tree_map(lambda x: x[0], inp),
                        jax.random.PRNGKey(1))
        assert tr.obs.shape == (20, dim) and tr.cobs.shape == (20, dim + 1)


def test_fastest_heuristic_always_picks_the_fastest_server():
    """: the rule formerly called "random" is deterministic and always picks the fastest server."""
    cfg = _cfg()
    spec = make_lb_spec(cfg)
    inp = jax.tree_util.tree_map(lambda x: x[0], sample_lb_batch(jax.random.PRNGKey(3), 1, **lb_input_kwargs(cfg)))
    from src.envs import lb_heuristic_action, lb_reset, lb_step

    st = lb_reset(spec)
    for _ in range(10):
        a = lb_heuristic_action(spec, "fastest", st, inp)
        assert int(a) == int(np.argmax(np.asarray(spec.rates)))
        st, _, _ = lb_step(spec, st, a, inp)
    assert lb_heuristic_rollout(spec, "fastest", inp).shape == (20,)
    with pytest.raises(ValueError):
        lb_heuristic_action(spec, "random", lb_reset(spec), inp)


def test_startup_overlap_check_uses_the_arms_own_group_count(monkeypatch, tmp_path):
    """: the start-up hash check draws the iteration-0 inputs with the arm's own G (maml: 1, state: 16)."""
    seen = []
    real = T.sample_lb_batch

    def spy(key, n, **kw):
        seen.append(n)
        return real(key, n, **kw)

    monkeypatch.setattr(T, "sample_lb_batch", spy)
    T.run(_cfg(["iterations=1"]), "maml", 0, str(tmp_path))
    G = T.batch_shape(_cfg(), "maml")[0]
    # order: the 3 fixed sequences (build), the 3 test sequences, then the iteration-0 spot check with G groups
    assert G == 1 and seen[:3] == [3, 3, G]


def test_maml_probe_at_training_k_reports_both_adaptation_set_sizes():
    """a k = 16 probe adapts each half's baseline on the other half's 8 rollouts ("maml") and, on the same
    rollouts, on only 4 of them ("maml_adapt_on_4"), so the effect of the adaptation-set size is isolated."""
    from src.eval import grad_variance_probe

    cfg = _cfg(["meta_k=16"])
    ts, consts, models = T.build(cfg, "maml", 0)
    out = grad_variance_probe(cfg, models["actor"], ts["actor"], jax.random.PRNGKey(0), n_z=2, k=16, n_fit_z=4,
                              fit_steps=5, meta=(models["critic"], ts["critic"], cfg))
    for b in ("maml", "maml_unadapted", "maml_adapt_on_4"):
        assert np.isfinite(out[b]["var_pg"]) and np.isfinite(out[b]["var_ratio_vs_state"])
    assert out["maml"]["var_pg"] != out["maml_adapt_on_4"]["var_pg"]
