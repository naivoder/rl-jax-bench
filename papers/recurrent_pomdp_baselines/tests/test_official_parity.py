"""Official-code official-code parity .  Fixture: tests/fixtures/parity.npz, recorded by
scripts/official/dump_parity.py from the UNMODIFIED official code
(twni2016/pomdp-baselines @ e7c19c3, torch 1.13.1 CPU, gym 0.18; no official code is imported here).

1. Evaluation path, end to end.  The official ``Learner.evaluate`` ran an official agent (paper sizes: obs/act/rew
   embeddings 32/8/8 or 32/0/8, RNN 128, MLP [128,128]; policy output layer re-drawn U(+-0.5) so deterministic actions
   are not ~0) on the official Pendulum-V-v0 and PointRobotSparse-v0 (VariBAD, 2 episodes/task) eval envs, recording
   every state / action / observation / reward.  We load the same weights and check
   (a) teacher-forced: our ``Agent.act`` on the official (prev action, reward, observation) history reproduces every
       official deterministic action (|diff| <= 1e-5 over 120-200 recurrent steps; measured <= 1e-6; a
       one-step shift of the reward input moves actions by 1e-2..2e-1, so the check is sensitive);
   (b) closed loop through OUR eval code (``Runner.run_episodes`` in "det" mode + our env, started from the official
       reset states): the per-episode returns and the official metric ``metrics/return_eval_total`` =
       np.mean(np.sum(returns, -1)) agree (Pendulum: float32 vs gym's float64 on a chaotic system, rel <= 1e-4 per
       episode, measured <= 9e-6; Semi-Circle: sparse 0/1 reward, exact).  The Semi-Circle internal VariBAD reset draws a random start,
       so the official post-boundary start is injected (the only intervention).
2. Replay sampler.  Official ``SeqReplayBuffer`` (vanilla) with episodes of 3/9/7/12 steps and L = 7: the valid-start
   weights per episode equal ours (max(1, n-L+1), uniform within an episode), and the official ``random_episodes``
   at every valid start, mapped to our layout, equals ``data.gather_sequences`` (mask exact; data exact where
   mask = 1, the official buffer reads the NEXT episode's data under mask 0, ours zero-pads).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import common  # noqa: F401
from src import data as D
from src.algorithm import Agent
from src.envs import Pendulum, SemiCircle
from src.model import Nets
from src.train import Runner

Z = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "parity.npz"))

# name -> (algo, cell, arch, inputs, env)
CASES = {"pend_sac_lstm_separate": ("sac", "lstm", "separate", "oar", "Pendulum-V"),
         "pend_sac_lstm_shared": ("sac", "lstm", "shared", "oar", "Pendulum-V"),
         "pend_td3_lstm_separate": ("td3", "lstm", "separate", "oar", "Pendulum-V"),
         "pend_td3_gru_shared": ("td3", "gru", "shared", "oar", "Pendulum-V"),
         "pend_sac_gru_separate": ("sac", "gru", "separate", "oar", "Pendulum-V"),
         "semi_td3_lstm_separate_or": ("td3", "lstm", "separate", "or", "SemiCircle"),
         "semi_td3_lstm_shared_or": ("td3", "lstm", "shared", "or", "SemiCircle"),
         "semi_sac_gru_separate_oar": ("sac", "gru", "separate", "oar", "SemiCircle")}


def _cfg(name):
    algo, cell, arch, inputs, env = CASES[name]
    cfg = common.load_config("pendulum_v.yaml" if env == "Pendulum-V" else "semicircle.yaml")
    cfg.update(algo=algo, cell=cell, arch=arch, inputs=inputs, action_embed=8, matmul_precision="highest")
    return cfg


def _lin(sd, k):
    return {"weight": jnp.asarray(sd[k + ".weight"]), "bias": jnp.asarray(sd[k + ".bias"])}


def _actor_params(name):
    """official actor-side state_dict -> our learner['params'] (only what Agent.actor_parts reads)."""
    algo, _, arch, _, _ = CASES[name]
    pre = f"ev/{name}/sd/"
    sd = {k[len(pre):]: Z[k] for k in Z.files if k.startswith(pre)}
    p = "actor." if arch == "separate" else ""
    enc = {e: _lin(sd, p + e + ".fc") for e in ("action_embedder", "reward_embedder", "observ_embedder")
           if p + e + ".fc.weight" in sd}
    enc["rnn"] = {"w_ih": jnp.asarray(sd[p + "rnn.weight_ih_l0"]), "w_hh": jnp.asarray(sd[p + "rnn.weight_hh_l0"]),
                  "b_ih": jnp.asarray(sd[p + "rnn.bias_ih_l0"]), "b_hh": jnp.asarray(sd[p + "rnn.bias_hh_l0"])}
    pi = {k: _lin(sd, p + "policy." + k) for k in ("fc0", "fc1", "last_fc")}
    if algo == "sac":
        pi["last_fc_log_std"] = _lin(sd, p + "policy.last_fc_log_std")
    short = {"fc": _lin(sd, p + "current_observ_embedder.fc")}
    if arch == "separate":
        return {"params": {"actor": {"enc": enc, "short": short, "pi": pi}}}
    return {"params": {"main": {"enc": enc, "short_o": short, "pi": pi}}}


def _episodes(name):
    p = f"ev/{name}/"
    return [{k: Z[f"{p}e{e}/{k}"] for k in ("s0", "goal", "obs", "act", "rew", "done", "state")}
            for e in range(int(Z[p + "n_eps"]))]


def _check_structure(name, learner):
    """our freshly initialised parameter tree has exactly the shapes of the loaded official one."""
    cfg = _cfg(name)
    env = Runner(cfg).env
    nets = Nets(cfg, env.obs_dim, env.act_dim, False)
    ref = Agent(cfg, nets).init(jax.random.PRNGKey(0))["params"]
    ref = {"actor": {k: ref["actor"][k] for k in ("enc", "short", "pi")}} if "actor" in learner["params"] else \
        {"main": {k: ref["main"][k] for k in ("enc", "short_o", "pi")}}
    s1 = jax.tree_util.tree_map(lambda x: x.shape, ref)
    s2 = jax.tree_util.tree_map(lambda x: x.shape, learner["params"])
    assert s1 == s2, (s1, s2)


@pytest.mark.parametrize("name", list(CASES))
def test_eval_actions_teacher_forced(name):
    cfg = _cfg(name)
    nets_env = Runner(cfg).env
    ag = Agent(cfg, Nets(cfg, nets_env.obs_dim, nets_env.act_dim, False))
    learner = _actor_params(name)
    _check_structure(name, learner)
    worst = 0.0
    for E in _episodes(name):
        T, A = E["act"].shape
        obs = jnp.asarray(E["obs"][:T], jnp.float32)[:, None]                       # (T, 1, O)
        prev_a = jnp.concatenate([jnp.zeros((1, 1, A)), jnp.asarray(E["act"][:-1], jnp.float32)[:, None]])
        prev_r = jnp.concatenate([jnp.zeros((1, 1, 1)), jnp.asarray(E["rew"][:-1], jnp.float32)[:, None, None]])

        def step(carry, x):
            a, carry = ag.act(learner, carry, x[0], x[1], x[2], jax.random.PRNGKey(0), deterministic=True)
            return carry, a[0]
        _, acts = jax.lax.scan(step, ag.init_carry(1), (prev_a, prev_r, obs))
        worst = max(worst, float(np.max(np.abs(np.asarray(acts) - E["act"]))))
    assert worst <= 1e-5, worst
    assert np.std(np.concatenate([E["act"].ravel() for E in _episodes(name)])) > 0.05  # non-trivial policy


class _FixedPendulum(Pendulum):
    """our Pendulum with reset(key, task) := the official reset state carried in ``task`` (test-only)."""
    def reset(self, key, task):
        state = {"th": task[0], "thdot": task[1], "t": jnp.zeros((), jnp.int32)}
        return state, self._obs(task[0], task[1])


class _FixedSemiCircle(SemiCircle):
    """our SemiCircle with task = (goal, official start, official post-boundary start) (test-only)."""
    def reset(self, key, task):
        state = {"pos": task[2:4], "t": jnp.zeros((), jnp.int32), "ep": jnp.zeros((), jnp.int32)}
        return state, self._obs(task[2:4], task[:2], jnp.zeros((), bool))

    def step(self, key, state, task, action):
        st, obs, r, done, term = super().step(key, state, task[:2], action)
        boundary = (st["ep"] == 1) & (st["t"] == 0) & (state["ep"] == 0)
        pos = jnp.where(boundary, task[4:6], st["pos"])
        term = jnp.where(boundary, (jnp.sqrt(jnp.sum((pos - task[:2]) ** 2)) <= self.goal_radius).astype(jnp.float32),
                         term)
        return {**st, "pos": pos}, obs, r, done, term


@pytest.mark.parametrize("name", list(CASES))
def test_eval_metric_closed_loop(name):
    cfg = _cfg(name)
    eps = _episodes(name)
    R = Runner(cfg)
    if cfg["env"] == "Pendulum-V":
        R.env = _FixedPendulum(obs_dims=(2,), version="v0")
        tasks = jnp.asarray(np.stack([E["s0"] for E in eps]), jnp.float32)
    else:
        R.env = _FixedSemiCircle()
        H = R.env.inner_H
        tasks = jnp.asarray(np.stack([np.concatenate([E["goal"], E["s0"], E["state"][H - 1]]) for E in eps]),
                            jnp.float32)
    R.H = R.env.H
    learner = _actor_params(name)
    n = tasks.shape[0]
    ekeys = jax.random.split(jax.random.PRNGKey(0), n)
    ep = R.run_episodes(learner, tasks, ekeys, jax.random.PRNGKey(1), "det")
    ours = np.asarray(jnp.sum(ep["rew"][..., 0], -1))                  # Runner._evaluate's per-episode return
    ref_eps = np.asarray(Z[f"ev/{name}/returns"])                       # official (n_tasks, episodes_per_task)
    ref_tot = ref_eps.sum(-1)
    assert np.allclose(ref_tot, [E["rew"].sum() for E in eps])        # recorder == evaluate()'s own bookkeeping
    official_metric = float(Z[f"ev/{name}/metric"])
    assert abs(official_metric - ref_tot.mean()) < 1e-9
    if cfg["env"] == "Pendulum-V":
        assert np.max(np.abs(ours - ref_tot) / np.abs(ref_tot)) < 1e-4, (ours, ref_tot)
        assert abs(ours.mean() - official_metric) / abs(official_metric) < 1e-4
    else:
        assert np.array_equal(ours, ref_tot), (ours, ref_tot)
        # per inner episode as the official logger splits it (return_eval_episode_1/2)
        H = R.env.inner_H
        per = np.asarray(jnp.stack([jnp.sum(ep["rew"][:, :H, 0], -1), jnp.sum(ep["rew"][:, H:, 0], -1)], -1))
        assert np.array_equal(per, ref_eps), (per, ref_eps)
        assert np.max(np.abs(np.asarray(ep["obs"]) - np.stack([E["obs"] for E in eps]))) < 1e-5


def _buffer():
    lens = [int(x) for x in Z["buf/lens"]]
    H, O, A = max(lens), 2, 1
    buf = D.init_buffer(len(lens), H, O, A)
    for e, n in enumerate(lens):
        pad = lambda x, m: np.concatenate([x, np.zeros((m - x.shape[0],) + x.shape[1:])])  # noqa: E731
        ep = {"obs": pad(Z[f"buf/ep{e}/obs"], H + 1)[None], "act": pad(Z[f"buf/ep{e}/act"], H)[None],
              "rew": pad(Z[f"buf/ep{e}/rew"], H)[None], "term": pad(Z[f"buf/ep{e}/term"], H)[None],
              "length": np.array([n])}
        buf = D.add_episodes(buf, {k: jnp.asarray(v) for k, v in ep.items()})
    return buf, lens


def test_replay_valid_start_weights():
    buf, lens = _buffer()
    L = int(Z["buf/L"])
    vs = Z["buf/valid_starts"]
    off = np.cumsum([0] + lens)
    ours_n = np.maximum(1, np.asarray(buf["length"]) - L + 1)        # = sample_sequences' per-episode weight
    for e, n in enumerate(lens):
        seg = vs[off[e]:off[e + 1]]
        assert np.sum(seg > 0) == ours_n[e]                            # same number of valid starts
        assert np.all(seg[: ours_n[e]] > 0) and np.allclose(seg[seg > 0], seg[0])  # uniform, first starts
        assert np.isclose(seg.sum(), ours_n[e])                        # episode weight = its start count


def test_replay_forced_starts_layout():
    buf, lens = _buffer()
    L = int(Z["buf/L"])
    off = np.cumsum([0] + lens)
    g = Z["buf/starts"]
    ep = np.searchsorted(off, g, side="right") - 1
    st = g - off[ep]
    ours = D.gather_sequences(buf, jnp.asarray(ep), jnp.asarray(st), L)
    b = {k[len("buf/batch/"):]: Z[k] for k in Z.files if k.startswith("buf/batch/")}
    B = len(g)
    ref = {"obs": np.concatenate([b["obs"][:1], b["obs2"]]),               # official policy.update() layout
           "act": np.concatenate([np.zeros((1, B, 1)), b["act"]]),
           "rew": np.concatenate([np.zeros((1, B, 1)), b["rew"]]),
           "term": np.concatenate([np.zeros((1, B, 1)), b["term"]]), "mask": b["mask"]}
    m = ref["mask"]
    assert np.array_equal(np.asarray(ours["mask"]), m)
    m1 = np.concatenate([np.ones((1, B, 1)), m])                            # slot 0 + every valid transition
    for k in ("act", "rew", "term"):
        assert np.allclose(np.asarray(ours[k]) * m1, ref[k] * m1, atol=1e-6), k
    mo = np.concatenate([np.ones((1, B, 1)), m])                            # obs[0] and obs[t+1] for valid t
    assert np.allclose(np.asarray(ours["obs"]) * mo, ref["obs"] * mo, atol=1e-6)
