# STUDY-ERA differential-test fixtures (official-code check): identical to difftest_official.py (HEAD seed_rl@0e1e0ac modules)
# except that the action distribution is seed_rl@68c15aa's NormalSquashedDistribution through scripts/official/studyera_dist.py
# (stored action = PRE-squash x; log-prob at x minus the postprocessor log-det), plus per-case unroll size and
# behaviour-logit perturbation (the larger 'band' case puts samples inside the PPO clip band so epsilon is pinned),
# plus standalone log-prob / gradient fixtures on a grid of raw x incl. deep saturation.
#.venv/bin/python scripts/official/difftest_studyera.py <out.npz>
# Builds the official ContinuousControlAgent + GeneralizedOnPolicyLoss on a small random unroll, runs
# learner.training_step's sequence (obs-norm update; compute_advantages_and_split; loss on ALL transitions as one
# minibatch; gradients; clip_by_global_norm; keras Adam with PolynomialDecay x2 steps) and dumps every input,
# weight and output so the JAX re-implementation can be compared on identical numbers.
import math
import sys

import numpy as np
import tensorflow as tf

import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
sys.path.insert(0, OFFICIAL_ROOT)
from seed_rl.agents.policy_gradient.modules import (advantages, continuous_control_agent, generalized_onpolicy_loss,  # noqa: E402
                                                    input_normalization, policy_losses, policy_regularizers, popart,
                                                    running_statistics)
from seed_rl.agents.policy_gradient.modules import ppo_training_step_utils as ptu  # noqa: E402
from seed_rl.common import parametric_distribution as pdist, utils  # noqa: E402
from studyera_dist import StudyEraDist, OLD  # noqa: E402

D_OBS, A = 7, 3
CASES = {
    "rec_pre":  dict(tanh=True, softplus=True, init_std=0.5, value_norm=True, vclip=None, adv_norm=False, ent=0.0,
                     shared=False, obs_norm=True, last_gain=0.5, wp=8, wv=16, eps=0.25, lam=0.9, T=5, B=6, pert=0.2),
    "base_pre": dict(tanh=False, softplus=False, init_std=1.0, value_norm=True, vclip=0.2, adv_norm=True, ent=1e-2,
                     shared=False, obs_norm=True, last_gain=0.5, wp=8, wv=8, eps=0.2, lam=0.95, T=5, B=6, pert=0.2),
    "band_pre": dict(tanh=True, softplus=True, init_std=0.5, value_norm=True, vclip=None, adv_norm=True, ent=0.0,
                     shared=False, obs_norm=True, last_gain=0.5, wp=8, wv=16, eps=0.25, lam=0.9, T=16, B=32, pert=0.25),
    "bandclip_pre": dict(tanh=False, softplus=False, init_std=1.0, value_norm=False, vclip=0.2, adv_norm=False,
                         ent=0.0, shared=False, obs_norm=True, last_gain=0.5, wp=8, wv=8, eps=0.2, lam=0.95, T=16,
                         B=32, pert=0.25),
}


def run_case(name, c, seed):
    T, B = c["T"], c["B"]
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    dist = StudyEraDist(A, c["tanh"], c["softplus"], c["init_std"])
    obs_normalizer = (input_normalization.InputNormalization(running_statistics.AverageMeanStd())
                      if c["obs_norm"] else None)
    agent = continuous_control_agent.ContinuousControlAgent(
        dist, observation_normalizer=obs_normalizer, num_layers_policy=2, num_layers_value=2,
        num_units_policy=c["wp"], num_units_value=c["wv"], shared=c["shared"], activation=tf.nn.tanh,
        kernel_init=tf.keras.initializers.Orthogonal(math.sqrt(2)),
        last_kernel_init_policy=tf.keras.initializers.Orthogonal(c["last_gain"]),
        last_kernel_init_value=tf.keras.initializers.Orthogonal(1.0),
        std_independent_of_input=True, input_clipping=10.0 if c["obs_norm"] else None)
    obs = (rng.normal(size=(T + 1, B, D_OBS)) * 3 + 1).astype(np.float32)
    zeros_b = np.zeros((T + 1, B), np.float32)
    # behaviour outputs: run the agent (before any normaliser update), then perturb the logits so rho != 1
    env0 = utils.EnvOutput(tf.constant(zeros_b), tf.constant(zeros_b > 0), tf.constant(obs),
                           tf.constant(zeros_b > 0), tf.constant(zeros_b.astype(np.int32)))
    prev = tf.zeros((T + 1, B, A))
    out0, _ = agent((prev, env0), (), unroll=True)
    logits_b = out0.policy_logits.numpy() + rng.normal(size=(T + 1, B, 2 * A)).astype(np.float32) * c["pert"]
    base_b = out0.baseline.numpy() + rng.normal(size=(T + 1, B)).astype(np.float32) * 0.3
    act = dist(tf.constant(logits_b)).sample().numpy()
    # pre-squash samples; force deep saturation on a few entries (tanh: |x| > atanh(.999) = 3.8; clip: |x| > 1)
    if T <= 5:
        act[0, 0, 0], act[1, 1, 1], act[2, 2, 2] = 4.5, -5.0, (2.5 if not c["tanh"] else 3.9)
    rew = rng.normal(size=(T + 1, B)).astype(np.float32)
    done = rng.random((T + 1, B)) < 0.15
    done[0] = False
    env = utils.EnvOutput(tf.constant(rew), tf.constant(done), tf.constant(obs), tf.constant(zeros_b > 0),
                          tf.constant(zeros_b.astype(np.int32)))
    agent_out = continuous_control_agent.AgentOutput(tf.constant(act), tf.constant(logits_b), tf.constant(base_b))
    # prior obs statistics, then the training-step update with observation[:-1]
    prior_obs = (rng.normal(size=(40, D_OBS)) * 2).astype(np.float32)
    if c["obs_norm"]:
        agent.update_observation_normalization_statistics(tf.constant(prior_obs))
        agent.update_observation_normalization_statistics(env.observation[:-1])
    reward_normalizer = None
    prior_targets = (rng.normal(size=(50,)) * 4 + 2).astype(np.float32)
    if c["value_norm"]:
        reward_normalizer = popart.PopArt(running_statistics.AverageMeanStd(), compensate=False)
        reward_normalizer.init()
        reward_normalizer.update_normalization_statistics(tf.constant(prior_targets[:, None]))
    reg = policy_regularizers.KLPolicyRegularizer(entropy=c["ent"]) if c["ent"] > 0 else None
    loss_fn = generalized_onpolicy_loss.GeneralizedOnPolicyLoss(
        agent, reward_normalizer, dist, advantages.GAE(lambda_=c["lam"]),
        policy_losses.ppo(c["eps"], normalize_advantages=c["adv_norm"]), 0.99, reg,
        value_ppo_style_clip_eps=c["vclip"], baseline_cost=1.0)
    loss_fn.init()
    unroll = ptu.Unroll if hasattr(ptu, "Unroll") else None
    import collections
    Unroll = collections.namedtuple("Unroll", "agent_state prev_actions env_outputs agent_outputs")
    args = Unroll((), prev, env, agent_out)
    pre_stats = reward_normalizer.mean_std_tracker.get_mean_std() if c["value_norm"] else (0.0, 1.0)
    split = ptu.compute_advantages_and_split(loss_fn, args)
    post_stats = reward_normalizer.mean_std_tracker.get_mean_std() if c["value_norm"] else (0.0, 1.0)
    tvars = loss_fn.trainable_variables
    with tf.GradientTape() as tape:
        loss = loss_fn(*split)
    grads = tape.gradient(loss, tvars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    gnorm = tf.linalg.global_norm(grads)
    cgrads, _ = tf.clip_by_global_norm(grads, 0.5, use_norm=gnorm)
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, 4, 0.0)
    opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    w0 = [v.numpy().copy() for v in tvars]
    opt.apply_gradients(zip(cgrads, tvars))
    opt.apply_gradients(zip([g * 0.5 for g in cgrads], tvars))
    w2 = [v.numpy().copy() for v in tvars]
    res = {"obs": obs, "act": act, "env_act": dist.postprocess(tf.constant(act)).numpy(), "logits_b": logits_b, "base_b": base_b, "rew": rew, "done": done,
           "prior_obs": prior_obs, "prior_targets": prior_targets,
           "n_targets": split[-2].numpy().reshape(T, B), "n_adv": split[-1].numpy().reshape(T, B),
           "loss": loss.numpy(), "gnorm": gnorm.numpy(),
           "pre_mean": np.asarray(pre_stats[0]), "pre_std": np.asarray(pre_stats[1]),
           "post_mean": np.asarray(post_stats[0]), "post_std": np.asarray(post_stats[1])}
    if c["obs_norm"]:
        m, s = agent.observation_normalizer.get_mean_std()
        res["obs_mean"], res["obs_std"] = m.numpy(), s.numpy()
    names = []
    for i, (v, g, cg, a0, a2) in enumerate(zip(tvars, grads, cgrads, w0, w2)):
        names.append(v.name)
        res[f"var{i}"], res[f"grad{i}"], res[f"cgrad{i}"], res[f"w2_{i}"] = a0, g.numpy(), cg.numpy(), a2
    res["names"] = np.array(names)
    return {f"{name}/{k}": v for k, v in res.items()}


def logprob_grid():
    """Standalone fixtures: log-prob and its gradient w.r.t. (mu, rho) on raw x incl. deep saturation."""
    rng = np.random.default_rng(7)
    out = {}
    for name, tanh, softplus, init_std in [("tanh_sp05", True, True, 0.5), ("clip_se1", False, False, 1.0),
                                           ("tanh_se1", True, False, 1.0), ("clip_sp05", False, True, 0.5)]:
        dist = StudyEraDist(A, tanh, softplus, init_std)
        mu = (rng.normal(size=(64, A)) * 2).astype(np.float32)
        rho = (rng.normal(size=(64, A)) * 0.7).astype(np.float32)
        x = (mu + rng.normal(size=(64, A)) * 1.5).astype(np.float32)
        x[:4] = np.array([[4.5, -5.0, 8.0], [-3.9, 3.81, 0.0], [12.0, -12.0, 1.0], [1.0, -1.0, 0.999]], np.float32)
        p = tf.Variable(np.concatenate([mu, rho], -1))
        with tf.GradientTape() as tape:
            lp = dist(p).log_prob(tf.constant(x))
            s = tf.reduce_sum(lp)
        g = tape.gradient(s, p).numpy()
        out.update({f"lp/{name}/mu": mu, f"lp/{name}/rho": rho, f"lp/{name}/x": x, f"lp/{name}/logp": lp.numpy(),
                    f"lp/{name}/grad": g, f"lp/{name}/post": dist.postprocess(tf.constant(x)).numpy(),
                    f"lp/{name}/std": dist.std_fn(tf.constant(rho)).numpy(),
                    f"lp/{name}/fldj": dist.old._postprocessor.forward_log_det_jacobian(tf.constant(x), 0).numpy()})
    return out


if __name__ == "__main__":
    out = logprob_grid()
    for i, (n, c) in enumerate(CASES.items()):
        out.update(run_case(n, c, 200 + i))
        print("[INFO] case", n, "loss", float(out[f"{n}/loss"]), "vars", len([k for k in out if k.startswith(f"{n}/var")]))
    out["meta/seed_rl_dist_commit"] = np.array("68c15aa")
    out["meta/seed_rl_modules_commit"] = np.array("0e1e0ac9178a670ad1e1463baed92020e88905ec")
    np.savez(sys.argv[1], **out)
    print("[INFO] wrote", sys.argv[1])
