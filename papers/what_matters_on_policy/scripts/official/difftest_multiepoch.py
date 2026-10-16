# MULTI-EPOCH differential-test fixtures. Same official modules and study-era distribution as
# difftest_studyera.py, but runs the OFFICIAL loop ppo_training_step_utils.ppo_training_step with
# batch_mode='split_with_advantage_recomputation', several epochs and several minibatches per epoch. The minibatch
# shuffle (tf.random.shuffle inside single_epoch) is replaced by fixed permutations that are saved, so the JAX side can
# replay them. Covers: per-epoch advantage recompute with the CURRENT params, PopArt stats updated per recompute,
# the obs-stat update before the epochs, minibatch gather order, per-minibatch adv-norm, clip and keras Adam under
# PolynomialDecay across epochs.
#   CUDA_VISIBLE_DEVICES=.venv/bin/python scripts/official/difftest_multiepoch.py <out.npz>
import collections
import math
import sys

import numpy as np
import tensorflow as tf

import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
sys.path.insert(0, OFFICIAL_ROOT)
from seed_rl.agents.policy_gradient.modules import (advantages, continuous_control_agent, generalized_onpolicy_loss,  # noqa: E402
                                                    input_normalization, policy_losses, popart, running_statistics)
from seed_rl.agents.policy_gradient.modules import ppo_training_step_utils as ptu  # noqa: E402
from seed_rl.common import utils  # noqa: E402
from studyera_dist import StudyEraDist  # noqa: E402

D_OBS, A = 7, 3
CASES = {
    "me_rec": dict(tanh=True, softplus=True, init_std=0.5, value_norm=True, vclip=None, adv_norm=False, obs_norm=True,
                   wp=8, wv=16, eps=0.25, lam=0.9, T=8, B=6, epochs=3, nmb=4, pert=0.2),
    "me_clip": dict(tanh=False, softplus=False, init_std=1.0, value_norm=True, vclip=0.2, adv_norm=True, obs_norm=True,
                    wp=8, wv=8, eps=0.2, lam=0.95, T=8, B=6, epochs=3, nmb=4, pert=0.2),
}


class _Logger:
    def log_session_from_dict(self, d):
        return None

    def log(self, *a, **k):
        return None


def run_case(name, c, seed):
    T, B = c["T"], c["B"]
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    dist = StudyEraDist(A, c["tanh"], c["softplus"], c["init_std"])
    obs_normalizer = input_normalization.InputNormalization(running_statistics.AverageMeanStd())
    agent = continuous_control_agent.ContinuousControlAgent(
        dist, observation_normalizer=obs_normalizer, num_layers_policy=2, num_layers_value=2,
        num_units_policy=c["wp"], num_units_value=c["wv"], shared=False, activation=tf.nn.tanh,
        kernel_init=tf.keras.initializers.Orthogonal(math.sqrt(2)),
        last_kernel_init_policy=tf.keras.initializers.Orthogonal(0.5),
        last_kernel_init_value=tf.keras.initializers.Orthogonal(1.0),
        std_independent_of_input=True, input_clipping=10.0)
    obs = (rng.normal(size=(T + 1, B, D_OBS)) * 3 + 1).astype(np.float32)
    zeros_b = np.zeros((T + 1, B), np.float32)
    env0 = utils.EnvOutput(tf.constant(zeros_b), tf.constant(zeros_b > 0), tf.constant(obs),
                           tf.constant(zeros_b > 0), tf.constant(zeros_b.astype(np.int32)))
    prev = tf.zeros((T + 1, B, A))
    out0, _ = agent((prev, env0), (), unroll=True)
    logits_b = out0.policy_logits.numpy() + rng.normal(size=(T + 1, B, 2 * A)).astype(np.float32) * c["pert"]
    base_b = out0.baseline.numpy() + rng.normal(size=(T + 1, B)).astype(np.float32) * 0.3
    act = dist(tf.constant(logits_b)).sample().numpy()
    rew = rng.normal(size=(T + 1, B)).astype(np.float32)
    done = rng.random((T + 1, B)) < 0.15
    done[0] = False
    env = utils.EnvOutput(tf.constant(rew), tf.constant(done), tf.constant(obs), tf.constant(zeros_b > 0),
                          tf.constant(zeros_b.astype(np.int32)))
    agent_out = continuous_control_agent.AgentOutput(tf.constant(act), tf.constant(logits_b), tf.constant(base_b))
    prior_obs = (rng.normal(size=(40, D_OBS)) * 2).astype(np.float32)
    agent.update_observation_normalization_statistics(tf.constant(prior_obs))
    reward_normalizer = popart.PopArt(running_statistics.AverageMeanStd(), compensate=False)
    reward_normalizer.init()
    prior_targets = (rng.normal(size=(50,)) * 4 + 2).astype(np.float32)
    reward_normalizer.update_normalization_statistics(tf.constant(prior_targets[:, None]))
    loss_fn = generalized_onpolicy_loss.GeneralizedOnPolicyLoss(
        agent, reward_normalizer, dist, advantages.GAE(lambda_=c["lam"]),
        policy_losses.ppo(c["eps"], normalize_advantages=c["adv_norm"]), 0.99, None,
        value_ppo_style_clip_eps=c["vclip"], baseline_cost=1.0)
    loss_fn.init()
    Unroll = collections.namedtuple("Unroll", "agent_state prev_actions env_outputs agent_outputs")
    args = Unroll((), prev, env, agent_out)
    tvars = loss_fn.trainable_variables
    w0 = [v.numpy().copy() for v in tvars]
    # learner.training_step: obs-stat update with observation[:-1] BEFORE the epochs
    agent.update_observation_normalization_statistics(env.observation[:-1])
    perms = np.stack([rng.permutation(T * B) for _ in range(c["epochs"])]).astype(np.int32)
    queue = list(perms)
    orig_shuffle = tf.random.shuffle
    tf.random.shuffle = lambda x, *a, **k: tf.constant(queue.pop(0))
    try:
        lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, c["epochs"] * c["nmb"] + 4, 0.0)
        opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        ptu.ppo_training_step(c["epochs"], loss_fn, args, "split_with_advantage_recomputation",
                              tf.distribute.get_strategy(), B, T, c["nmb"], 0.5, opt, _Logger())
    finally:
        tf.random.shuffle = orig_shuffle
    assert not queue, "every epoch must consume exactly one permutation"
    res = {"obs": obs, "act": act, "logits_b": logits_b, "base_b": base_b, "rew": rew, "done": done,
           "prior_obs": prior_obs, "prior_targets": prior_targets, "perms": perms,
           "lr_decay_steps": np.float32(c["epochs"] * c["nmb"] + 4), "opt_iterations": np.int64(opt.iterations.numpy())}
    m, s = reward_normalizer.mean_std_tracker.get_mean_std()
    res["popart_mean"], res["popart_std"] = np.asarray(m), np.asarray(s)
    m, s = agent.observation_normalizer.get_mean_std()
    res["obs_mean"], res["obs_std"] = m.numpy(), s.numpy()
    names = []
    for i, (v, a0) in enumerate(zip(tvars, w0)):
        names.append(v.name)
        res[f"var{i}"], res[f"wend_{i}"] = a0, v.numpy().copy()
    res["names"] = np.array(names)
    return {f"{name}/{k}": v for k, v in res.items()}


if __name__ == "__main__":
    out = {}
    for i, (n, c) in enumerate(CASES.items()):
        out.update(run_case(n, c, 300 + i))
        print("[INFO] case", n, "opt iterations", int(out[f"{n}/opt_iterations"]),
              "names", list(out[f"{n}/names"])[:3])
    out["meta/seed_rl_dist_commit"] = np.array("68c15aa")
    out["meta/seed_rl_modules_commit"] = np.array("0e1e0ac9178a670ad1e1463baed92020e88905ec")
    np.savez(sys.argv[1], **out)
    print("[INFO] wrote", sys.argv[1])
