"""Official-code parity (planned check 2), replayed from committed fixtures.

The authors released no code (README 'Official-code verification'). The fixtures in tests/fixtures/official/ were
recorded from the AUTHORS' OWN FORKS of their named toolchain -- stable-baselines3 2.4.0a8 @ e46a6d5 and imitation
@ f5cb8a4, installed from github.com/iwishiwasaneagle -- plus the torch 2.3.1 modules an SB3 features extractor is
built from, and from Stone Soup 1.9.1 (the metric pipeline). Generators: scripts/official/sb3_ref.py and
scripts/official/ss_metric_ref.py (oracle venvs only). This file needs neither torch nor Stone Soup: it loads the
recorded inputs, runs OUR JAX code on them and compares with the recorded outputs.

  P1  SB3 collect_rollouts timeout bootstrap + RolloutBuffer GAE           vs alg.gae            (<= 2e-6 abs, f32)
  P2  SB3 PPO.train: per-step gradients (3 Adam steps, clipping active)    vs alg.ppo_loss_terms (<= 1e-5 of max|g|)
      logged pg / value / entropy losses, clip fraction, approx-KL         (<= 1e-5 relative)
  P3  SB3 optimiser (clip_grad_norm_ 0.5 + Adam eps 1e-5) on SB3's grads   vs alg.ppo_optimizer  (<= 2e-6 abs)
  P4  imitation BehaviorCloningLossCalculator loss / grads, BC trainer     vs alg.bc_ce_loss + Adam(1e-3)
  P5  torch MultiheadAttention + residual + LayerNorm                       vs model.MHSA         (<= 2e-5 abs)
  P6  torch bidirectional GRU, packed (masked) and zero-padded               vs model.BiGRU        (<= 2e-5 abs)
  P7  SB3 NatureCNN on a 1x48x48 raster                                     vs model.NatureCNN    (<= 2e-5 abs)
  P8  SB3 init semantics: Linear layers inside a features extractor are re-initialised orthogonal(sqrt 2), zero
      bias; GRU and MHA in-projection keep torch defaults                   vs model init_style="torch"
  P9  Stone Soup MetricManager (SumofCovarianceNorms, GOSPA) on OUR rollouts' detections           (see the test)
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from src.configs.config import EnvConfig, NetConfig  # noqa: E402
from src import algorithm as alg  # noqa: E402
from src.model import ActorCritic, MHSA, BiGRU, NatureCNN, TrackEncoder  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "official")
ECFG = replace(EnvConfig(), n_a=19)
NET = ActorCritic(NetConfig(mlp=(64, 64)), ECFG)


def load(name):
    return dict(np.load(os.path.join(FIX, name)))


def heads_params(d, prefix):
    """SB3 MlpPolicy state_dict -> our ActorCritic head params (torch Linear [out, in] -> flax kernel [in, out])."""
    m = {"pi_0": "mlp_extractor.policy_net.0", "pi_1": "mlp_extractor.policy_net.2",
         "v_0": "mlp_extractor.value_net.0", "v_1": "mlp_extractor.value_net.2",
         "action_net": "action_net", "value_net": "value_net"}
    return {k: {"kernel": jnp.asarray(d[f"{prefix}{v}.weight"].T, jnp.float64),
                "bias": jnp.asarray(d[f"{prefix}{v}.bias"], jnp.float64)} for k, v in m.items()}


def to_torch_names(g):
    m = {"pi_0": "mlp_extractor.policy_net.0", "pi_1": "mlp_extractor.policy_net.2",
         "v_0": "mlp_extractor.value_net.0", "v_1": "mlp_extractor.value_net.2",
         "action_net": "action_net", "value_net": "value_net"}
    out = {}
    for k, v in m.items():
        out[f"{v}.weight"] = np.asarray(g[k]["kernel"]).T
        out[f"{v}.bias"] = np.asarray(g[k]["bias"])
    return out


def heads(P, obs):
    return NET.apply({"params": P}, obs, method=ActorCritic.heads)


def unflat(x, E, T):
    """SB3 swap_and_flatten (env-major [E*T, ...]) -> [T, E, ...]."""
    return np.asarray(x).reshape((E, T) + x.shape[1:]).swapaxes(0, 1)


# ----------------------------------------------------------------------------- P1
def test_P1_timeout_bootstrap_and_gae_vs_sb3():
    d = load("sb3_ppo.npz")
    T, E = d["raw_r"].shape
    lr, ent, vf, mgn, gamma, lam, clip, eps = d["hp"]
    assert d["raw_trunc"].sum() > 0, "the fixture must contain truncations"
    # SB3 collect_rollouts: reward += gamma * V(terminal obs) on TimeLimit truncation
    r_boot = d["raw_r"] + gamma * d["v_term"] * d["raw_trunc"]
    np.testing.assert_allclose(r_boot, d["rewards_buf"], atol=2e-6)
    # SB3 stores episode_starts[t+1] = done[t]
    np.testing.assert_array_equal(d["episode_starts"][1:], d["raw_done"][:-1])
    vals = unflat(d["values"].reshape(-1), E, T)
    adv, ret = alg.gae(jnp.asarray(d["rewards_buf"], jnp.float64), jnp.asarray(vals, jnp.float64),
                       jnp.asarray(d["raw_done"], jnp.float64), jnp.asarray(d["last_values"], jnp.float64),
                       float(gamma), float(lam))
    np.testing.assert_allclose(np.asarray(adv), unflat(d["advantages"].reshape(-1), E, T), atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ret), unflat(d["returns"].reshape(-1), E, T), atol=2e-5, rtol=1e-6)


# ----------------------------------------------------------------------------- P2, P3
def _ppo_batch(d):
    return dict(obs=jnp.asarray(d["obs"], jnp.float64), a=jnp.asarray(d["actions"], jnp.int32),
                logp=jnp.asarray(d["log_probs"].reshape(-1), jnp.float64),
                adv=jnp.asarray(d["advantages"].reshape(-1), jnp.float64),
                ret=jnp.asarray(d["returns"].reshape(-1), jnp.float64))


def _ppo_loss(P, B, hp):
    lr, ent, vf, mgn, gamma, lam, clip, eps = hp
    la, le, v = heads(P, B["obs"])
    pg, vl, el, aux = alg.ppo_loss_terms(la, le, v, B["a"], B["logp"], B["adv"], B["ret"], float(clip))
    return pg + ent * el + vf * vl, (pg, vl, el, aux)


def test_P2_ppo_gradients_and_logged_losses_vs_sb3():
    d = load("sb3_ppo.npz")
    hp = d["hp"]
    B = _ppo_batch(d)
    n = int(d["n_opt_steps"])
    assert n == 3
    pgs, vls, els, cfs, kls, losses = [], [], [], [], [], []
    for i in range(n):
        P = heads_params(d, f"s{i}/before/")
        (l, (pg, vl, el, aux)), g = jax.value_and_grad(_ppo_loss, has_aux=True)(P, B, hp)
        gt = to_torch_names(g)
        gmax = max(np.max(np.abs(d[f"s{i}/pre/{k}"])) for k in gt)
        for k, v in gt.items():
            err = np.max(np.abs(v - d[f"s{i}/pre/{k}"])) / gmax
            assert err < 1e-5, (i, k, err)
        gn = float(optax.global_norm(g))
        assert abs(gn - float(d[f"s{i}/norm"])) / gn < 1e-5
        pgs.append(float(pg)), vls.append(float(vl)), els.append(float(el))
        cfs.append(float(aux["clip_frac"])), kls.append(float(aux["approx_kl"])), losses.append(float(l))
    log = lambda k: float(d["log/train/" + k])
    assert abs(np.mean(pgs) - log("policy_gradient_loss")) <= 1e-5 * abs(log("policy_gradient_loss")) + 1e-7
    assert abs(np.mean(vls) - log("value_loss")) <= 1e-5 * abs(log("value_loss"))
    assert abs(np.mean(els) - log("entropy_loss")) <= 1e-5 * abs(log("entropy_loss"))
    assert abs(np.mean(cfs) - log("clip_fraction")) < 1e-6 and np.mean(cfs) > 0.1      # clipping exercised (f32 cast)
    assert abs(kls[-1] - log("approx_kl")) <= 1e-4 * abs(log("approx_kl"))           # SB3 logs the last epoch's KL
    assert abs(losses[-1] - log("loss")) <= 1e-5 * abs(log("loss"))


def test_P3_ppo_optimizer_vs_sb3():
    d = load("sb3_ppo.npz")
    lr, ent, vf, mgn, gamma, lam, clip, eps = d["hp"]
    opt = alg.ppo_optimizer(float(lr), float(mgn), float(eps))
    P = heads_params(d, "p0/")
    st = opt.init(P)
    n = int(d["n_opt_steps"])
    for i in range(n):
        g = {k: {kk: jnp.asarray(vv) for kk, vv in v.items()} for k, v in heads_params(d, f"s{i}/pre/").items()}
        u, st = opt.update(g, st, P)
        P = optax.apply_updates(P, u)
        ref = f"s{i + 1}/before/" if i + 1 < n else "pend/"
        for k, v in to_torch_names(P).items():
            assert np.max(np.abs(v - d[ref + k])) < 2e-6, (i, k, np.max(np.abs(v - d[ref + k])))
        post_ours = to_torch_names(jax.tree.map(lambda x: x * min(1.0, float(mgn) / float(optax.global_norm(g))), g))
        for k, v in post_ours.items():                                   # clipped gradient (torch adds 1e-6)
            np.testing.assert_allclose(v, d[f"s{i}/post/{k}"], rtol=2e-6, atol=1e-9)


# ----------------------------------------------------------------------------- P4
def test_P4_bc_loss_and_trainer_vs_imitation():
    d = load("imitation_bc.npz")
    P0 = heads_params(d, "p0/")
    obs, acts = jnp.asarray(d["obs"][:32], jnp.float64), jnp.asarray(d["acts"][:32], jnp.int32)

    def loss(P, o, a):
        la, le, _ = heads(P, o)
        return alg.bc_ce_loss(la, le, a, 1e-3)
    l, g = jax.value_and_grad(loss)(P0, obs, acts)
    assert abs(float(l) - float(d["m/loss"])) < 2e-6 * abs(float(d["m/loss"]))
    la, le, _ = heads(P0, obs)
    lp, en = alg.logp_entropy(la, le, acts)
    assert abs(-float(lp.mean()) - float(d["m/neglogp"])) < 2e-6 * float(d["m/neglogp"])
    assert abs(float(en.mean()) - float(d["m/entropy"])) < 2e-6 * float(d["m/entropy"])
    gt = to_torch_names(g)
    gmax = max(np.max(np.abs(d["g0/" + k])) for k in gt)
    for k, v in gt.items():
        assert np.max(np.abs(v - d["g0/" + k])) / gmax < 1e-5, k
    # trainer: imitation's own shuffled batches, torch Adam(lr 1e-3, eps 1e-8) == optax.adam(1e-3)
    lr_, eps_ = d["adam"]
    assert lr_ == 1e-3 and eps_ == 1e-8
    opt = optax.adam(1e-3)
    P, st = P0, opt.init(P0)
    for i in range(int(d["n_batches"])):
        g = jax.grad(loss)(P, jnp.asarray(d[f"b{i}/obs"], jnp.float64), jnp.asarray(d[f"b{i}/acts"], jnp.int32))
        u, st = opt.update(g, st, P)
        P = optax.apply_updates(P, u)
    for k, v in to_torch_names(P).items():
        assert np.max(np.abs(v - d["pend/" + k])) < 2e-6, (k, np.max(np.abs(v - d["pend/" + k])))


# ----------------------------------------------------------------------------- P5
def mha_params_from_torch(d):
    W, b = d["mha/in_proj_weight"], d["mha/in_proj_bias"]
    dm = W.shape[1]
    h = 4
    hd = dm // h
    p = {}
    for j, nm in enumerate(("query", "key", "value")):
        p[nm] = {"kernel": jnp.asarray(W[j * dm:(j + 1) * dm].T.reshape(dm, h, hd)),
                 "bias": jnp.asarray(b[j * dm:(j + 1) * dm].reshape(h, hd))}
    p["out"] = {"kernel": jnp.asarray(d["mha/out_proj.weight"].T.reshape(h, hd, dm)),
                "bias": jnp.asarray(d["mha/out_proj.bias"])}
    return {"mha": p, "ln": {"scale": jnp.asarray(d["ln/weight"]), "bias": jnp.asarray(d["ln/bias"])}}


def test_P5_mhsa_vs_torch_multiheadattention():
    d = load("torch_mha.npz")
    P = mha_params_from_torch(d)
    assert float(d["ln_eps"]) == 1e-5
    m = MHSA(64, 4)                                   # ln_eps defaults to torch's 1e-5 (fixed in this phase)
    y = np.asarray(m.apply({"params": P}, jnp.asarray(d["e"]), jnp.asarray(d["valid"])))
    v = d["valid"]
    assert np.max(np.abs(y[v] - d["y"][v])) < 2e-5
    assert np.max(np.abs(y - d["y"])) < 2e-5          # padded query rows also agree (they attend to valid keys)
    m6 = MHSA(64, 4, ln_eps=1e-6)                     # the small-scale setting: measurably different, but small
    y6 = np.asarray(m6.apply({"params": P}, jnp.asarray(d["e"]), jnp.asarray(d["valid"])))
    assert 1e-6 < np.max(np.abs(y6 - d["y"])) < 1e-3


# ----------------------------------------------------------------------------- P6
def gru_params_from_torch(d):
    p = {}
    for nm, suf in (("fwd", ""), ("bwd", "_reverse")):
        wi, wh = d[f"gru/weight_ih_l0{suf}"], d[f"gru/weight_hh_l0{suf}"]
        bi, bh = d[f"gru/bias_ih_l0{suf}"], d[f"gru/bias_hh_l0{suf}"]
        H = wh.shape[1]
        p[f"{nm}_wi"] = jnp.asarray(wi.T)
        p[f"{nm}_wh"] = jnp.asarray(wh.T)
        p[f"{nm}_bi"] = jnp.asarray(bi + np.concatenate([bh[:2 * H], np.zeros(H, np.float32)]))
        p[f"{nm}_bhn"] = jnp.asarray(bh[2 * H:])
    return p


def test_P6_bigru_vs_torch_gru():
    d = load("torch_gru.npz")
    P = gru_params_from_torch(d)
    g = BiGRU(64)
    x = jnp.asarray(d["x"])
    h_masked = np.asarray(g.apply({"params": P}, x, jnp.asarray(d["valid"])))
    assert np.max(np.abs(h_masked - d["h_packed"])) < 2e-5           # masked carry-through == packed sequence
    h_all = np.asarray(g.apply({"params": P}, x, jnp.ones(d["valid"].shape, bool)))
    assert np.max(np.abs(h_all - d["h_unpacked"])) < 2e-5            # NetConfig(masked=False) == zero rows fed in
    assert np.max(np.abs(h_all - h_masked)) > 1e-3                   # and the two readings differ


# ----------------------------------------------------------------------------- P7
def test_P7_naturecnn_vs_sb3():
    d = load("sb3_naturecnn.npz")
    conv = lambda w: jnp.asarray(w.transpose(2, 3, 1, 0))
    Wl = d["cnn/linear.0.weight"]                                    # [128, 64*2*2], torch (C, H, W) flatten
    P = {"Conv_0": {"kernel": conv(d["cnn/cnn.0.weight"]), "bias": jnp.asarray(d["cnn/cnn.0.bias"])},
         "Conv_1": {"kernel": conv(d["cnn/cnn.2.weight"]), "bias": jnp.asarray(d["cnn/cnn.2.bias"])},
         "Conv_2": {"kernel": conv(d["cnn/cnn.4.weight"]), "bias": jnp.asarray(d["cnn/cnn.4.bias"])},
         "Dense_0": {"kernel": jnp.asarray(Wl.reshape(128, 64, 2, 2).transpose(2, 3, 1, 0).reshape(256, 128)),
                     "bias": jnp.asarray(d["cnn/linear.0.bias"])}}
    y = np.asarray(NatureCNN(128).apply({"params": P}, jnp.asarray(d["img"][:, 0])))
    assert np.max(np.abs(y - d["y"])) < 2e-5 * max(1.0, np.max(np.abs(d["y"])))


# ----------------------------------------------------------------------------- P8
def _is_ortho(w, gain):
    w = np.asarray(w, np.float64).reshape(w.shape[0], -1) if w.ndim == 2 else np.asarray(w, np.float64)
    g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
    return np.allclose(g, np.eye(len(g)) * gain ** 2, atol=1e-4)


def test_P8_init_semantics_match_sb3():
    s = json.load(open(os.path.join(FIX, "sb3_ref_summary.json")))["init_semantics"]
    # what SB3 does (recorded): Linear (incl. MHA out_proj) -> orthogonal sqrt 2, zero bias; MHA in_proj and GRU kept
    assert s["embed.weight"]["orthogonal"] and abs(s["embed.weight"]["gain"] - 2 ** 0.5) < 1e-6
    assert s["mha.out_proj.weight"]["orthogonal"] and s["out.weight"]["orthogonal"]
    assert not s["mha.in_proj_weight"]["orthogonal"] and not s["gru.weight_ih_l0"]["orthogonal"]
    rows, mask = jnp.zeros((1, 17, 7), jnp.float32), jnp.ones((1, 17), bool)
    for enc in ("bigru", "mhsa"):
        p = TrackEncoder(NetConfig(encoder=enc))
        prm = p.init(jax.random.PRNGKey(0), rows, mask)["params"]
        if enc == "bigru":
            assert _is_ortho(prm["out"]["kernel"], 2 ** 0.5) and not np.any(prm["out"]["bias"])
            k = 1 / np.sqrt(64)                                         # torch GRU: U(-1/sqrt H, 1/sqrt H)
            assert np.max(np.abs(prm["bigru"]["fwd_wi"])) <= k and np.max(np.abs(prm["bigru"]["fwd_wh"])) <= k
            assert np.max(np.abs(prm["bigru"]["fwd_bhn"])) <= k and np.max(np.abs(prm["bigru"]["fwd_bi"])) <= 2 * k
            assert np.std(prm["bigru"]["fwd_wi"]) > 0.5 * k / np.sqrt(3)
        else:
            assert _is_ortho(prm["embed"]["kernel"], 2 ** 0.5) and not np.any(prm["embed"]["bias"])
            ko = np.asarray(prm["mhsa"]["mha"]["out"]["kernel"]).reshape(64, 64)
            assert _is_ortho(ko, 2 ** 0.5) and not np.any(prm["mhsa"]["mha"]["out"]["bias"])
            kq = np.asarray(prm["mhsa"]["mha"]["query"]["kernel"])
            assert np.max(np.abs(kq)) <= np.sqrt(6 / (64 + 192)) + 1e-7     # xavier_uniform on in_proj [192, 64]
            assert not np.any(prm["mhsa"]["mha"]["query"]["bias"])
    # the small-scale initialisation stays selectable
    p = TrackEncoder(NetConfig(encoder="mhsa", init_style="flax"))
    prm = p.init(jax.random.PRNGKey(0), rows, mask)["params"]
    assert not _is_ortho(prm["embed"]["kernel"], 2 ** 0.5)


# ----------------------------------------------------------------------------- P9
def test_P9_metrics_vs_stonesoup_metric_manager():
    """Our per-episode M_cov and summed GOSPA, computed online from the live track list, vs Stone Soup's
    MetricManager (SumofCovarianceNormsMetric, GOSPAMetric) run by scripts/official/ss_metric_ref.py on the SAME
    detections and truths (the B6-verified Stone Soup tracker, analytic Jacobian on both sides).
    Reading (i) 'live' = the tracker's current tracks at each step (what the online reward sees): must match.
    Reading (ii) 'library' = all tracks ever, as the Stone Soup tutorial pipeline accumulates them, so a deleted
    track's last (predicted, trace > 5000) state counts at its deletion step: recorded, not asserted equal."""
    path = os.path.join(FIX, "ss_metrics.npz")
    if not os.path.exists(path):
        pytest.skip("ss_metrics.npz not generated")
    d = load("ss_metrics.npz")
    ours = d["ours_m_cov"], d["ours_gospa"]
    np.testing.assert_allclose(ours[0], d["ss_live_m_cov"], rtol=1e-6)
    np.testing.assert_allclose(ours[1], d["ss_live_gospa"], rtol=1e-6, atol=1e-3)
    # reading (ii): our descriptive M_cov_lib (killed tracks' last state included) equals the library pipeline
    np.testing.assert_allclose(d["ours_m_cov_lib"], d["ss_lib_m_cov"], rtol=1e-6)
    assert np.all(d["ss_lib_m_cov"] >= d["ss_live_m_cov"] - 1e-9)


# ----------------------------------------------------------------------------- P10
def flat_net_params(d, prefix):
    """SB3 MultiInputPolicy (CombinedExtractor: NatureCNN on 'raster', Flatten on 'tl'; features ordered [raster,
    tl] by the sorted Dict keys) -> our ActorCritic(encoder='flat'), whose features are [tl, cnn]."""
    g = lambda k: d[prefix + k]
    conv = lambda w: jnp.asarray(w.transpose(2, 3, 1, 0))
    cnn = "features_extractor.extractors.raster."
    Wl = g(cnn + "linear.0.weight")
    P = {"cnn": {"Conv_0": {"kernel": conv(g(cnn + "cnn.0.weight")), "bias": jnp.asarray(g(cnn + "cnn.0.bias"))},
                 "Conv_1": {"kernel": conv(g(cnn + "cnn.2.weight")), "bias": jnp.asarray(g(cnn + "cnn.2.bias"))},
                 "Conv_2": {"kernel": conv(g(cnn + "cnn.4.weight")), "bias": jnp.asarray(g(cnn + "cnn.4.bias"))},
                 "Dense_0": {"kernel": jnp.asarray(Wl.reshape(128, 64, 2, 2).transpose(2, 3, 1, 0).reshape(256, 128)),
                             "bias": jnp.asarray(g(cnn + "linear.0.bias"))}}}
    for ours, th_ in (("pi", "policy_net"), ("v", "value_net")):
        W0 = g(f"mlp_extractor.{th_}.0.weight")                                   # [128, 128 (cnn) + 119 (tl)]
        P[f"{ours}_0"] = {"kernel": jnp.asarray(np.concatenate([W0[:, 128:].T, W0[:, :128].T], 0)),
                          "bias": jnp.asarray(g(f"mlp_extractor.{th_}.0.bias"))}
        P[f"{ours}_1"] = {"kernel": jnp.asarray(g(f"mlp_extractor.{th_}.2.weight").T),
                          "bias": jnp.asarray(g(f"mlp_extractor.{th_}.2.bias"))}
    for k in ("action_net", "value_net"):
        P[k] = {"kernel": jnp.asarray(g(k + ".weight").T), "bias": jnp.asarray(g(k + ".bias"))}
    return P


def flat_grads_to_torch(gr):
    cnn = "features_extractor.extractors.raster."
    out = {}
    for i, j in ((0, 0), (1, 2), (2, 4)):
        out[cnn + f"cnn.{j}.weight"] = np.asarray(gr["cnn"][f"Conv_{i}"]["kernel"]).transpose(3, 2, 0, 1)
        out[cnn + f"cnn.{j}.bias"] = np.asarray(gr["cnn"][f"Conv_{i}"]["bias"])
    out[cnn + "linear.0.weight"] = np.asarray(gr["cnn"]["Dense_0"]["kernel"]).reshape(2, 2, 64, 128) \
        .transpose(3, 2, 0, 1).reshape(128, 256)
    out[cnn + "linear.0.bias"] = np.asarray(gr["cnn"]["Dense_0"]["bias"])
    for ours, th_ in (("pi", "policy_net"), ("v", "value_net")):
        k0 = np.asarray(gr[f"{ours}_0"]["kernel"])
        out[f"mlp_extractor.{th_}.0.weight"] = np.concatenate([k0[119:].T, k0[:119].T], 1)
        out[f"mlp_extractor.{th_}.0.bias"] = np.asarray(gr[f"{ours}_0"]["bias"])
        out[f"mlp_extractor.{th_}.2.weight"] = np.asarray(gr[f"{ours}_1"]["kernel"]).T
        out[f"mlp_extractor.{th_}.2.bias"] = np.asarray(gr[f"{ours}_1"]["bias"])
    for k in ("action_net", "value_net"):
        out[k + ".weight"] = np.asarray(gr[k]["kernel"]).T
        out[k + ".bias"] = np.asarray(gr[k]["bias"])
    return out


def test_P10_full_flat_network_ppo_update_vs_sb3():
    """End to end on the paper-setting Flat policy: a real rollout of OUR env collected by SB3 (authors' fork),
    then SB3's own PPO minibatch steps. Our ActorCritic receives the COMPACT observation and renders the raster
    itself. Checks: raster == SB3's buffer image; forward logp / V at the rollout parameters == SB3's buffer;
    per optimizer step, our PPO-loss gradient on SB3's minibatch indices at SB3's parameters == SB3's pre-clip
    gradient (float32 on both sides)."""
    from src.model import raster_of
    from src.data import Obs
    path = os.path.join(FIX, "sb3_flat_update.npz")
    if not os.path.exists(path):
        pytest.skip("sb3_flat_update.npz not generated")
    d = load("sb3_flat_update.npz")
    ecfg = EnvConfig()
    net = ActorCritic(NetConfig(encoder="flat"), ecfg)
    obs = Obs(*(jnp.asarray(d["obs/" + k]) for k in Obs._fields))
    img = np.asarray(raster_of(obs, ecfg))
    n_img = d["raster"].shape[0]
    assert np.max(np.abs(img[:n_img] - d["raster"][:, 0])) < 1e-6 * max(1.0, np.max(np.abs(d["raster"])))
    tl = np.asarray(obs.tl * obs.tl_mask[..., None])
    np.testing.assert_allclose(tl, d["tl_sb3"], atol=1e-7)
    P0 = {"params": flat_net_params(d, "s0/before/")}
    la, le, v = net.apply(P0, obs)
    a = jnp.asarray(d["actions"], jnp.int32)
    lp, _ = alg.logp_entropy(la, le, a)
    assert np.max(np.abs(np.asarray(lp) - d["log_probs"].reshape(-1))) < 1e-4
    assert np.max(np.abs(np.asarray(v) - d["values"].reshape(-1))) < 1e-4
    B = int(d["batch"])
    n_mb = d["perms"].shape[1] // B
    steps = sorted({int(k.split("/")[0][1:]) for k in d if k.startswith("s") and "/pre/" in k})
    assert len(steps) >= 2
    adv, ret, old = (jnp.asarray(d[k].reshape(-1)) for k in ("advantages", "returns", "log_probs"))

    def loss_fn(p, idx):
        o = jax.tree.map(lambda x: x[idx], obs)
        la, le, v = net.apply(p, o)
        pg, vl, el, _ = alg.ppo_loss_terms(la, le, v, a[idx], old[idx], adv[idx], ret[idx], 0.2)
        return pg + 0.0 * el + 0.5 * vl
    from src.model import NatureCNN

    def cnn_preact(pc, img):                         # pre-activation of the CNN's final Dense (before its ReLU)
        x = img[..., None]
        for j, s_ in enumerate((4, 2, 1)):
            x = jax.nn.relu(jax.lax.conv_general_dilated(x, pc[f"Conv_{j}"]["kernel"], (s_, s_), "VALID",
                                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
                            + pc[f"Conv_{j}"]["bias"])
        return x.reshape(x.shape[0], -1) @ pc["Dense_0"]["kernel"] + pc["Dense_0"]["bias"]

    worst, kinks = 0.0, []
    lin = "features_extractor.extractors.raster.linear.0."
    for i in steps:
        if f"s{i}/pre/action_net.bias" not in d:
            continue
        idx = jnp.asarray(d["perms"][i // n_mb][(i % n_mb) * B:(i % n_mb + 1) * B])
        Pi = flat_net_params(d, f"s{i}/before/")
        # a ReLU kink: a sample whose final-CNN-Dense pre-activation sits at float32 round-off (|z| < 1e-7) can be
        # gated differently by torch and XLA, which changes that sample's whole backward path. Measured in the
        # recorded fixture: step 5, unit 1, |z| = 1.05e-9 (raw error 7.9e-3 of max|g|). Causal check: shifting that
        # unit's bias by +-2e-8 (far below every other |z|) must reproduce SB3's gradient.
        z = np.asarray(cnn_preact(Pi["cnn"], jnp.asarray(img)[np.asarray(idx)]))
        ku = np.where(np.min(np.abs(z), 0) < 1e-7)[0]
        variants = [Pi]
        for sgn in (1.0, -1.0):
            if len(ku):
                b = np.asarray(Pi["cnn"]["Dense_0"]["bias"]).copy()
                b[ku] += sgn * 2e-8
                variants.append({**Pi, "cnn": {**Pi["cnn"], "Dense_0": {**Pi["cnn"]["Dense_0"], "bias": jnp.asarray(b)}}})
        errs = []
        for Pv in variants:
            g = flat_grads_to_torch(jax.grad(loss_fn)({"params": Pv}, idx)["params"])
            gmax = max(np.max(np.abs(d[f"s{i}/pre/{k}"])) for k in g)
            errs.append(max(np.max(np.abs(v_ - d[f"s{i}/pre/{k}"])) for k, v_ in g.items()) / gmax)
        err = min(errs)
        if len(ku):
            kinks.append(dict(step=i, units=ku.tolist(), raw_err=float(errs[0]), resolved_err=float(err)))
        worst = max(worst, err)
        assert err < 1e-4, (i, errs)
    print(f"[P10] steps {steps}: worst gradient error {worst:.2e} of max|g|; ReLU kinks excluded {kinks}")


def test_P11_full_mhsa_network_ppo_update_vs_sb3():
    """Recheck: the headline BiGRU+MHSA network end to end. A torch extractor in the authors' SB3 fork
    (scripts/official/mhsa_sb3.py --record-update) collected a real rollout of OUR env and ran SB3's own
    PPO minibatch steps; the fixture keeps steps 1, 4, 7 (step 0 has every bias exactly 0 and subnormal raster pixels,
    the ReLU-kink round-off already documented for P10). Checks: raster and track rows == SB3's buffer; forward logp / V
    at the rollout parameters == SB3's buffer; per kept step, our PPO-loss gradient on SB3's minibatch at SB3's
    parameters == SB3's pre-clip gradient, all 36 parameter tensors (float32 on both sides)."""
    from src.model import raster_of
    from src.data import Obs
    from scripts.official import mhsa_map as M
    path = os.path.join(FIX, "sb3_mhsa_update.npz")
    if not os.path.exists(path):
        pytest.skip("sb3_mhsa_update.npz not generated")
    d = load("sb3_mhsa_update.npz")
    ecfg = EnvConfig()
    net = ActorCritic(NetConfig(encoder="mhsa"), ecfg)
    obs = Obs(*(jnp.asarray(d["obs/" + k]) for k in Obs._fields))
    img = np.asarray(raster_of(obs, ecfg))
    assert np.max(np.abs(img - d["raster"][:, 0])) < 1e-6 * max(1.0, np.max(np.abs(d["raster"])))
    np.testing.assert_allclose(np.asarray(obs.tl * obs.tl_mask[..., None]), d["tl_sb3"], atol=1e-7)
    la, le, v = net.apply({"params": M.params(d, "p0/")}, obs)
    a = jnp.asarray(d["actions"], jnp.int32)
    lp, _ = alg.logp_entropy(la, le, a)
    assert np.max(np.abs(np.asarray(lp) - d["log_probs"].reshape(-1))) < 1e-5
    assert np.max(np.abs(np.asarray(v) - d["values"].reshape(-1))) < 1e-5
    B = int(d["batch"])
    n_mb = d["perms"].shape[1] // B
    adv, ret, old = (jnp.asarray(d[k].reshape(-1)) for k in ("advantages", "returns", "log_probs"))

    def loss_fn(p, idx):
        o = jax.tree.map(lambda x: x[idx], obs)
        la, le, v = net.apply(p, o)
        pg, vl, el, _ = alg.ppo_loss_terms(la, le, v, a[idx], old[idx], adv[idx], ret[idx], 0.2)
        return pg + 0.0 * el + 0.5 * vl

    steps = [int(i) for i in d["kept_steps"]]
    assert steps == [1, 4, 7]
    for i in steps:
        idx = jnp.asarray(d["perms"][i // n_mb][(i % n_mb) * B:(i % n_mb + 1) * B])
        g = M.grads_to_torch(jax.grad(loss_fn)({"params": M.params(d, f"s{i}/before/")}, idx)["params"])
        ref = {k: d[f"s{i}/pre/{k}"] for k in g}
        assert len(ref) == 36
        gmax = max(np.max(np.abs(r)) for r in ref.values())
        worst = max(float(np.max(np.abs(g[k] - ref[k]))) / gmax for k in g)
        assert worst < 1e-5, (i, worst)
