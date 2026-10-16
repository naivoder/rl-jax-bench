# Differential-test fixtures from the OFFICIAL seed_rl modules (commit 0e1e0ac), run in this folder's TF venv:
#.venv/bin/python scripts/official/difftest_official.py <out.npz>
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

T, B, D_OBS, A = 5, 6, 7, 3
CASES = {
    # name: tanh, softplus, init_std, value_norm, vclip, adv_norm, ent, shared, obs_norm, last_gain, wp, wv
    "rec":    dict(tanh=True, softplus=True, init_std=0.5, value_norm=True, vclip=None, adv_norm=False, ent=0.0,
                   shared=False, obs_norm=True, last_gain=0.5, wp=8, wv=16, eps=0.25, lam=0.9),
    "base":   dict(tanh=False, softplus=False, init_std=1.0, value_norm=True, vclip=0.2, adv_norm=True, ent=1e-2,
                   shared=False, obs_norm=True, last_gain=0.5, wp=8, wv=8, eps=0.2, lam=0.95),
    "shared": dict(tanh=True, softplus=True, init_std=0.5, value_norm=False, vclip=None, adv_norm=False, ent=0.0,
                   shared=True, obs_norm=False, last_gain=0.5, wp=8, wv=8, eps=0.25, lam=0.9),
}


def run_case(name, c, seed):
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    std_fn = (pdist.softplus_std_fn if c["softplus"] else pdist.safe_exp_std_fn)(c["init_std"], 1e-3)
    mk = pdist.normal_tanh_distribution if c["tanh"] else pdist.normal_clipped_distribution
    dist = mk(A, gaussian_std_fn=std_fn)
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
    logits_b = out0.policy_logits.numpy() + rng.normal(size=(T + 1, B, 2 * A)).astype(np.float32) * 0.2
    base_b = out0.baseline.numpy() + rng.normal(size=(T + 1, B)).astype(np.float32) * 0.3
    act = dist(tf.constant(logits_b)).sample().numpy()
    if c["tanh"]:
        act[0, 0, 0], act[1, 1, 1], act[2, 2, 2] = 0.9995, -1.0, 0.99999       # tail branches 
    else:
        act[0, 0, 0] = 1.0                                                     # saturated clip action 
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
    res = {"obs": obs, "act": act, "logits_b": logits_b, "base_b": base_b, "rew": rew, "done": done,
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


if __name__ == "__main__":
    out = {}
    for i, (n, c) in enumerate(CASES.items()):
        out.update(run_case(n, c, 100 + i))
        print("[INFO] case", n, "loss", float(out[f"{n}/loss"]), "vars", len([k for k in out if k.startswith(f"{n}/var")]))
    np.savez(sys.argv[1], **out)
    print("[INFO] wrote", sys.argv[1])
