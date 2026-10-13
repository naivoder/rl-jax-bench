"""Parity with the AUTHORS' CODE, from recorded fixtures.

tests/fixtures/official/<case>.npz holds inputs we fed both sides and the outputs of the authors' code on them
(key prefix "oracle/"): CleanRL @fe8d8a0 `ppo.py` / `ppo_continuous_action.py` / `ppo_atari_lstm.py` Agent classes,
the post's `ppo_shared.py` @fbef824 Agent, and the GAE / minibatch-loss / clip_grad_norm_ / Adam / lr-annealing blocks
extracted verbatim from those files; baselines @ea25b9e `VecNormalize`; gymnasium 0.29.1 wrappers and classic-control
dynamics; gym 0.21.0 wrappers (the post's own ppo_continuous_action.py stack) and classic-control dynamics. Provenance: tests/fixtures/official/provenance.json; the harness that produced them lives outside this repo
(scripts/official/). Rebuild: scripts/small/run_difftests.sh then
scripts/small/make_parity_fixtures.py.

This file recomputes OUR side from the fixture inputs only (weights copied from the torch state_dict layout), so it
does not depend on PRNG streams. Tolerance: |ours - theirs| <= 1e-5 + 1e-5 |theirs| (the loop brief's 1e-5), float32,
matmul precision "highest" (set by src.train).

Run: uv run python -m pytest tests/test_oracle_parity.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pytest  # noqa: E402

from src import algorithm as alg  # noqa: E402
from src import data as dt  # noqa: E402
from src import model as mdl  # noqa: E402
from src.train import CONFIG_DIR, PPO, load_config  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "official")
TOL = 1e-5


class MB(dict):
    __getattr__ = dict.__getitem__


def load(case):
    z = np.load(os.path.join(FIX, f"{case}.npz"))
    return {k: z[k] for k in z.files}


def check(ours: dict, z: dict, skip=()):
    """Compare every oracle array with ours; returns the number of arrays compared (must be > 0)."""
    n, bad = 0, []
    for k, b in z.items():
        if not k.startswith("oracle/"):
            continue
        key = k[len("oracle/"):]
        if key.startswith("meta/") or any(key.startswith(s) for s in skip):
            continue
        key = "lr/ours" if key == "lr/oracle" else key
        assert key in ours, f"ours is missing {key}"
        a = np.asarray(ours[key], np.float64)
        b = np.asarray(b, np.float64).reshape(a.shape)
        err = np.abs(a - b)
        if not np.all(err <= TOL + TOL * np.abs(b)):
            bad.append((key, float(err.max())))
        n += 1
    assert not bad, f"parity failures (key, max abs err): {bad}"
    return n


# ----- torch state_dict <-> flax params -------------------------------------------------------------------------------

def from_torch_sd(template, z, arch, cont):
    sd = {k[3:]: jnp.asarray(v) for k, v in z.items() if k.startswith("sd/")}
    p = jax.tree.map(lambda x: x, template)["params"]

    def lin(prefix):
        return {"kernel": sd[prefix + ".weight"].T, "bias": sd[prefix + ".bias"]}

    if arch == "separate":
        a = "actor_mean" if cont else "actor"
        new = {"actor": {"l0": lin(f"{a}.0"), "l1": lin(f"{a}.2")}, "actor_head": lin(f"{a}.4"),
               "critic": {"l0": lin("critic.0"), "l1": lin("critic.2")}, "critic_head": lin("critic.4")}
        if cont:
            new["log_std"] = sd["actor_logstd"][0]
    elif arch == "shared":
        new = {"torso": {"l0": lin("network.0"), "l1": lin("network.2")}, "actor_head": lin("actor"),
               "critic_head": lin("critic")}
    else:  # lstm: torch layout (4H, in), gate order (i, f, g, o), two biases, as in our cell
        new = {"encoder": lin("network.0"), "actor_head": lin("actor"), "critic_head": lin("critic"),
               "lstm": {"cell": {"weight_ih": sd["lstm.weight_ih_l0"], "weight_hh": sd["lstm.weight_hh_l0"],
                                 "bias_ih": sd["lstm.bias_ih_l0"], "bias_hh": sd["lstm.bias_hh_l0"]}}}
    assert jax.tree.structure(new) == jax.tree.structure(p), "fixture weights do not fit our parameter tree"
    for x, y in zip(jax.tree.leaves(new), jax.tree.leaves(p)):
        assert x.shape == y.shape
    return {"params": new}


def to_torch_names(tree, arch, cont, tag):
    p = jax.tree.map(np.asarray, tree)["params"]
    out = {}

    def lin(prefix, d):
        out[f"{tag}/{prefix}.weight"] = d["kernel"].T
        out[f"{tag}/{prefix}.bias"] = d["bias"]

    if arch == "separate":
        a = "actor_mean" if cont else "actor"
        lin(f"{a}.0", p["actor"]["l0"]); lin(f"{a}.2", p["actor"]["l1"]); lin(f"{a}.4", p["actor_head"])
        lin("critic.0", p["critic"]["l0"]); lin("critic.2", p["critic"]["l1"]); lin("critic.4", p["critic_head"])
        if cont:
            out[f"{tag}/actor_logstd"] = p["log_std"][None]
    elif arch == "shared":
        lin("network.0", p["torso"]["l0"]); lin("network.2", p["torso"]["l1"])
        lin("actor", p["actor_head"]); lin("critic", p["critic_head"])
    else:
        lin("network.0", p["encoder"])
        c = p["lstm"]["cell"]
        out[f"{tag}/lstm.weight_ih_l0"], out[f"{tag}/lstm.weight_hh_l0"] = c["weight_ih"], c["weight_hh"]
        out[f"{tag}/lstm.bias_ih_l0"], out[f"{tag}/lstm.bias_hh_l0"] = c["bias_ih"], c["bias_hh"]
        lin("actor", p["actor_head"]); lin("critic", p["critic_head"])
    return out


# ----- feed-forward cases: CleanRL ppo.py (separate, +/- value clip), post ppo_shared.py, ppo_continuous_action.py -----

FF = {"separate": ("classic_control.yaml", {"architecture": "separate"}),
      "separate_novclip": ("classic_control.yaml", {"architecture": "separate", "clip_vloss": False}),
      "shared": ("classic_control.yaml", {"architecture": "shared"}),
      "gaussian": ("pendulum.yaml", {"ent_coef": 0.01})}


@pytest.mark.parametrize("case", list(FF))
def test_ff_parity(case):
    z = load(case)
    config, over = FF[case]
    cfg = load_config(os.path.join(CONFIG_DIR, config))
    cfg.update(over)
    cfg.update(num_envs=4, num_steps=128, num_minibatches=4)
    ppo = PPO(cfg)
    cont, arch = ppo.cont, cfg["architecture"]
    for k in ("clip_coef", "ent_coef", "vf_coef", "clip_vloss"):   # the fixture was made with these coefficients
        assert np.asarray(z[f"meta/{k}"]).item() == cfg[k], k
    runner, _ = ppo.init(jnp.int32(0))
    params = from_torch_sd(runner.params, z, arch, cont)
    ours = {}
    # (1) forward: logits / Gaussian mean, value, log-prob, entropy on 256 observations
    dist, v = ppo.net.apply(params, jnp.asarray(z["fwd/obs"]))
    act = jnp.asarray(z["fwd/act"])
    ours.update({"fwd/value": v, "fwd/logp": mdl.log_prob(dist, act, cont), "fwd/entropy": mdl.entropy(dist, cont),
                 "fwd/out0": dist[0] if cont else dist})
    # (2) GAE (official no-bootstrap-on-truncation semantics), M=128, N=4, mid-segment dones, next_done=1 for 2 envs
    next_value = ppo.net.apply(params, jnp.asarray(z["gae/next_obs"]))[1]
    adv, ret = alg.gae(jnp.asarray(z["gae/rewards"]), jnp.asarray(z["gae/values"]), jnp.asarray(z["gae/dones"]),
                       next_value, jnp.asarray(z["gae/next_done"]), 0.99, 0.95)
    ours.update({"gae/adv": adv, "gae/ret": ret})
    # (3) two consecutive minibatch steps: loss terms, debug variables, raw + clipped grads, params after Adam
    perm = np.asarray(z["b/perm"])
    opt_state = alg.set_lr(ppo.tx.init(params), float(z["b/lr"]))
    p = params
    for step in range(2):
        ids = perm[step * 128:(step + 1) * 128]
        mb = MB(obs=jnp.asarray(z["b/obs"][ids]), action=jnp.asarray(z["b/actions"][ids]),
                logp=jnp.asarray(z["b/logprobs"][ids]), value=jnp.asarray(z["b/values"][ids]),
                ret=jnp.asarray(z["b/returns"][ids]), adv=jnp.asarray(z["b/advantages"][ids]), done=None)
        (loss, aux), g = jax.value_and_grad(ppo._loss, has_aux=True)(p, mb)
        gc, _ = optax.clip_by_global_norm(cfg["max_grad_norm"]).update(g, None)
        p, opt_state, _ = ppo.sgd_step(p, opt_state, mb)
        pre = f"step{step}/"
        ours.update({pre + k: np.asarray(val) for k, val in aux.items()})
        ours[pre + "loss"] = loss
        ours.update(to_torch_names(g, arch, cont, pre + "grad"))
        ours.update(to_torch_names(gc, arch, cont, pre + "gradclip"))
        ours.update(to_torch_names(p, arch, cont, pre + "params_after"))
    # (4) lr annealing over 10 updates
    ours["lr/ours"] = np.array([float(alg.lr_at(jnp.int32(u), 10, 2.5e-4, True)) for u in range(10)])
    n = check(ours, z)
    assert n >= 60, n
    # power: the batch really exercises the clip, the value clip and the global-norm clip
    assert 0.2 < float(z["oracle/step0/clipfrac"]) < 0.8
    raw = np.concatenate([np.ravel(v) for k, v in ours.items() if k.startswith("step0/grad/")])
    clp = np.concatenate([np.ravel(v) for k, v in ours.items() if k.startswith("step0/gradclip/")])
    assert np.abs(raw - clp).max() > 1e-4, "global-norm clip inactive: the test would not see a clip bug"


# ----- LSTM: CleanRL ppo_atari_lstm.py get_states (reset at done from a saved state) + env-column minibatch ----------

def test_lstm_parity():
    z = load("lstm")
    cfg = load_config(os.path.join(CONFIG_DIR, "lstm_masked_cartpole.yaml"))
    cfg.update(num_envs=4, num_steps=16, num_minibatches=2)
    ppo = PPO(cfg)
    runner, _ = ppo.init(jnp.int32(0))
    params = from_torch_sd(runner.params, z, "lstm", False)
    obs, dones = jnp.asarray(z["seq/obs"]), jnp.asarray(z["seq/dones"])
    h0 = (jnp.asarray(z["seq/h0"]), jnp.asarray(z["seq/c0"]))
    assert float(dones.sum()) == 4.0 and float(jnp.abs(h0[0]).sum()) > 0  # in-segment resets from a non-zero state
    logits, v, (hT, cT) = ppo.net.apply(params, obs, dones, h0)
    ours = {"seq/logits": logits, "seq/value": v, "seq/hT": hT, "seq/cT": cT}
    e = jnp.asarray(z["b/mbenvinds"])
    mb = MB(obs=obs[:, e], action=jnp.asarray(z["b/actions"])[:, e], logp=jnp.asarray(z["b/logprobs"])[:, e],
            value=jnp.asarray(z["b/values"])[:, e], ret=jnp.asarray(z["b/returns"])[:, e],
            adv=jnp.asarray(z["b/advantages"])[:, e], done=dones[:, e])
    ic = (h0[0][e], h0[1][e])
    (loss, aux), g = jax.value_and_grad(ppo._loss, has_aux=True)(params, mb, ic)
    p2, _, _ = ppo.sgd_step(params, alg.set_lr(ppo.tx.init(params), float(z["b/lr"])), mb, ic)
    ours["step0/loss"] = loss
    ours.update({"step0/" + k: np.asarray(val) for k, val in aux.items()})
    ours.update(to_torch_names(g, "lstm", False, "step0/grad"))
    ours.update(to_torch_names(p2, "lstm", False, "step0/params_after"))
    assert check(ours, z) >= 20


# ----- wrappers: baselines VecNormalize (default mode) and gymnasium 0.29.1 NormalizeObservation/Reward --------

def _replay(mode, s, N=2, T=1000):
    cfg = load_config(os.path.join(CONFIG_DIR, "pendulum.yaml"))
    cfg.update(num_envs=N, obs_norm=True, reward_scaling=True, reward_scaling_mode=mode)
    env = dt.VecEnv(cfg)
    S = {k: jnp.asarray(v) for k, v in s.items()}

    def reset_one(key):
        i = key[0]
        return S["init"][i], (i, jnp.int32(0))

    def step_one(key, state, action):
        i, t = state
        done = jnp.logical_or(S["term"][i, t], S["trunc"][i, t])
        o_next = S["nxt"][i, t]
        return (jnp.where(done, S["rst"][i, t], o_next), (i, t + 1), S["rew"][i, t], S["term"][i, t],
                S["trunc"][i, t], o_next)

    env.reset_one, env.step_one = reset_one, step_one
    keys = jnp.stack([jnp.array([i, 0], jnp.uint32) for i in range(N)])
    obs0, es = jax.vmap(reset_one)(keys)
    vs, obs0 = env.wrap_reset(obs0, es)
    step = jax.jit(lambda vs: env.step(jnp.zeros(2, jnp.uint32), vs, jnp.zeros((N, 1))))
    obs, rews, fins = [np.asarray(obs0)], [], []
    for _ in range(T):
        vs, so = step(vs)
        obs.append(np.asarray(so.obs)); rews.append(np.asarray(so.reward)); fins.append(np.asarray(so.final_obs))
    return np.stack(obs, 1), np.stack(rews, 1), np.stack(fins, 1)


def test_wrapper_parity():
    z = load("wrappers")
    z21 = load("wrappers_gym021")   # the post's own ppo_continuous_action.py wrappers under gym 0.21.0
    s = {k[len("stream/"):]: v for k, v in z.items() if k.startswith("stream/")}
    assert s["term"].any() and s["trunc"].any()
    ours = {}
    for mode in ("baselines", "gymnasium029", "gym021"):
        o, r, f = _replay(mode, s)
        ours.update({f"{mode}/obs": o, f"{mode}/rew": r, f"{mode}/final": f})
    assert check(ours, z) == 5
    assert check(ours, z21) == 2
    # power: the three modes differ by far more than the tolerance, so the test can tell them apart
    assert np.abs(ours["baselines/rew"] - ours["gymnasium029/rew"]).max() > 0.1
    assert np.abs(ours["gym021/rew"] - ours["gymnasium029/rew"]).max() > 0.1   #: reward order differs
    assert np.abs(ours["gym021/obs"] - ours["baselines/obs"]).max() > 0.1      #: per-env, double obs update


# ----- env dynamics: gymnax vs gymnasium 0.29.1, teacher-forced one-step transitions ----------------------------------

FIELDS = {"CartPole-v1": ["x", "x_dot", "theta", "theta_dot"],
          "Acrobot-v1": ["joint_angle1", "joint_angle2", "velocity_1", "velocity_2"],
          "MountainCar-v0": ["position", "velocity"],
          "Pendulum-v1": ["theta", "theta_dot"]}


@pytest.mark.parametrize("oracle", ["envs", "envs_gym021"])   # gymnasium 0.29.1 (CleanRL) and gym 0.21.0 (the post)
@pytest.mark.parametrize("env_id", list(FIELDS))
def test_env_parity(env_id, oracle):
    import gymnax
    z = load(oracle)
    env, prm = gymnax.make(env_id)
    _, s0 = env.reset(jax.random.PRNGKey(0), prm)
    S, A = jnp.asarray(z[f"{env_id}/tf/state"]), jnp.asarray(z[f"{env_id}/tf/action"])

    def one(sv, a):
        st = s0.replace(**{f: sv[i] for i, f in enumerate(FIELDS[env_id])})
        o, _, r, term, _ = env.step_env(jax.random.PRNGKey(0), st, a, prm)
        return o, r, term

    o, r, term = jax.jit(jax.vmap(one))(S, A)
    ours = {f"{env_id}/tf/obs": o, f"{env_id}/tf/reward": r, f"{env_id}/tf/terminated": term.astype(np.float64)}
    sub = {k: v for k, v in z.items() if k.startswith(f"oracle/{env_id}/")}
    sub = {k: (v.astype(np.float64) if v.dtype == bool else v) for k, v in sub.items()}
    assert check(ours, sub) == 3


# ----- metric: the paper's episodic return (RecordEpisodeStatistics innermost) on the same replayed rollouts ----------

def test_metric_parity():
    """Our per-step raw episodic returns/lengths (what train.py turns into the curve and the final-window metric) equal
    those the authors' env stacks report on the same stream, in every wrapper mode (the metric is raw, so the mode
    must not matter). The post's ppo.py logs only the first finishing env per step: it is a subset of ours."""
    from scripts.official_check.metric_export import episodes
    from src.eval import summarize
    z = load("metric")
    s = {k[len("stream/"):]: v for k, v in z.items() if k.startswith("stream/")}
    for mode in ("baselines", "gymnasium029", "gym021"):
        r, ln = episodes(mode, s)
        for tag in ("gymnasium029", "gym021"):
            np.testing.assert_allclose(r, z[f"oracle/{tag}/ep_ret"], atol=1e-4, rtol=1e-5)
            np.testing.assert_array_equal(ln, z[f"oracle/{tag}/ep_len"])
    assert (ln > 0).sum() > 50   # power: many episodes, terminations and truncations both end some
    # the final-window metric, computed by the same function from both sides' episode lists, agrees
    N, T = r.shape
    gs = (np.arange(T) + 1) * N

    def ep_array(R, L):
        cnt = (L > 0).sum(0)
        nz = cnt > 0
        return np.stack([gs[nz], R.sum(0)[nz], cnt[nz], L.sum(0)[nz]], 1)

    a = summarize(ep_array(r, ln), N * T, "Pendulum-v1")["final_window"]
    b = summarize(ep_array(z["oracle/gymnasium029/ep_ret"], z["oracle/gymnasium029/ep_len"]), N * T,
                  "Pendulum-v1")["final_window"]
    assert abs(a - b) <= 1e-5 * max(1.0, abs(b))
    first = z["oracle/gym021/ep_ret_first_only"]
    assert np.all((first == 0) | np.isclose(first, z["oracle/gym021/ep_ret"]))
