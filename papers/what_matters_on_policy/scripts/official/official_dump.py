# Official-code reference curve : the OFFICIAL seed_rl modules (ContinuousControlAgent,
# GeneralizedOnPolicyLoss, PopArt/AverageMeanStd, ppo loss, compute_advantages_and_split) driven by a minimal
# synchronous loop that mirrors learner.training_step (obs-stat update with observation[:-1]; split /
# split_with_advantage_recomputation; shuffled minibatches; clip_by_global_norm; keras Adam with PolynomialDecay over
# ceil(2e6/2048) iterations) on gymnasium MuJoCo (v4) envs. The SEED gRPC actor/learner stack is NOT used (not
# runnable here); everything numerical comes from the official modules.
#   .venv/bin/python scripts/official/official_loop.py <Hopper-v4> <base|rec|base_pre|rec_pre|released> <seed> <out.json> [total_steps]
# *_pre (the official-code check): the action distribution is seed_rl@68c15aa's NormalSquashedDistribution (scripts/official/studyera_dist.py;
# the semantics in force when the study ran): the agent outputs and stores the PRE-squash x, the env gets post(x).
# released: the released mujoco defaults (gin/ppo.gin + gcp/train_mujoco.sh: tanh with HEAD semantics, safe_exp std 1,
# PopArt, lambda 0.95, eps 0.2, no value clipping, batch_mode split, unroll 16 x 128 envs = 2048, minibatch 64,
# 10 epochs, lr 3e-4 -> 0 over 2e6 frames, clip_norm 0.5, value width 64).
import json
import logging
import math
import sys
import time

import gymnasium as gym
import numpy as np
import tensorflow as tf

import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
sys.path.insert(0, OFFICIAL_ROOT)
from seed_rl.agents.policy_gradient.modules import (advantages, continuous_control_agent, generalized_onpolicy_loss,  # noqa: E402
                                                    input_normalization, policy_losses, popart, running_statistics)
from seed_rl.agents.policy_gradient.modules import ppo_training_step_utils as ptu  # noqa: E402
from seed_rl.common import parametric_distribution as pdist, utils  # noqa: E402
from studyera_dist import StudyEraDist  # noqa: E402
import collections  # noqa: E402
logging.getLogger().setLevel(logging.ERROR)   # TFP subclass-annotation warnings flood the log on every call

Unroll = collections.namedtuple("Unroll", "agent_state prev_actions env_outputs agent_outputs")
CFG = {
    # Table 2 base (separate nets, value clipping 0.2, clip transform, safe_exp, init std 1, lambda 0.95, eps 0.2,
    # split = 'shuffle transitions', value width 64)
    "base": dict(tanh=False, softplus=False, init_std=1.0, lam=0.95, eps=0.2, vclip=0.2, mode="split", wv=64),
    # REC (paper's recommendations as in our agent.yaml)
    "rec": dict(tanh=True, softplus=True, init_std=0.5, lam=0.9, eps=0.25, vclip=None,
                mode="split_with_advantage_recomputation", wv=256),
    "released": dict(tanh=True, softplus=False, init_std=1.0, lam=0.95, eps=0.2, vclip=None, mode="split", wv=64,
                     N=128, T=16),
}
CFG["base_pre"] = dict(CFG["base"], pre=True)
CFG["rec_pre"] = dict(CFG["rec"], pre=True)
# full run (cause test for the A2b cell): REC study-era with initial std 1.0 (one factor)
CFG["rec_pre_std1"] = dict(CFG["rec_pre"], init_std=1.0)
CFG["rec_pre_lr1em4"] = dict(CFG["rec_pre"], lr=1e-4)   # A5a cell
# recheck (copy of scripts/official/official_loop.py; only CFG entries, the lr horizon and the value-norm switch added)
CFG["rec_pre_vnone"] = dict(CFG["rec_pre"], vnorm=False)            # A7 cell: reward_normalizer=None
CFG["rec_pre_to0"] = dict(CFG["rec_pre"], lr_frames=1_001_472)      # /: lr -> 0 at the 1M budget
CFG["rec_pre_lr1em4_to0"] = dict(CFG["rec_pre"], lr=1e-4, lr_frames=1_001_472)
CFG["rec_pre_std1_to0"] = dict(CFG["rec_pre"], init_std=1.0, lr_frames=1_001_472)



def main():
    env_name, cfg_name, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    K = int(sys.argv[5])
    total = K * 2048
    DUMP = {}
    c = CFG[cfg_name]
    tf.random.set_seed(seed)
    N, T, EPOCHS, MB = c.get("N", 256), c.get("T", 8), 10, 64
    pre = c.get("pre", False)
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(N)])
    obs, _ = envs.reset(seed=seed)
    A = envs.single_action_space.shape[0]
    if pre:
        dist = StudyEraDist(A, c["tanh"], c["softplus"], c["init_std"])
        std_fn, post = dist.std_fn, (lambda x: dist.postprocess(tf.constant(x)).numpy())
    else:
        std_fn = (pdist.softplus_std_fn if c["softplus"] else pdist.safe_exp_std_fn)(c["init_std"], 1e-3)
        dist = (pdist.normal_tanh_distribution if c["tanh"] else pdist.normal_clipped_distribution)(A, gaussian_std_fn=std_fn)
        post = lambda x: x
    agent = continuous_control_agent.ContinuousControlAgent(
        dist, observation_normalizer=input_normalization.InputNormalization(running_statistics.AverageMeanStd()),
        num_layers_policy=2, num_layers_value=2, num_units_policy=64, num_units_value=c["wv"], shared=False,
        activation=tf.nn.tanh, kernel_init=tf.keras.initializers.Orthogonal(math.sqrt(2)),
        last_kernel_init_policy=tf.keras.initializers.Orthogonal(0.01),
        last_kernel_init_value=tf.keras.initializers.Orthogonal(1.0), std_independent_of_input=True,
        input_clipping=10.0)
    pa = popart.PopArt(running_statistics.AverageMeanStd(), compensate=False) if c.get("vnorm", True) else None
    if pa is not None:
        pa.init()
    loss_fn = generalized_onpolicy_loss.GeneralizedOnPolicyLoss(
        agent, pa, dist, advantages.GAE(lambda_=c["lam"]), policy_losses.ppo(c["eps"]), 0.99, None,
        value_ppo_style_clip_eps=c["vclip"], baseline_cost=1.0)

    def mk_env_out(r, d, o):
        z = np.zeros_like(r, np.float32)
        return utils.EnvOutput(tf.constant(r, tf.float32), tf.constant(d), tf.constant(o, tf.float32),
                               tf.constant(z > 0), tf.constant(z.astype(np.int32)))

    act_fn = tf.function(lambda e: agent((tf.zeros((N, A)), e), ()))
    # create variables
    agent((tf.zeros((1, N, A)), mk_env_out(np.zeros((1, N)), np.zeros((1, N), bool), obs[None])), (), unroll=True)
    loss_fn.init()
    DUMP['names'] = np.array([v.name for v in loss_fn.trainable_variables])
    for i, v in enumerate(loss_fn.trainable_variables):
        DUMP[f'w0_{i}'] = v.numpy().copy()
    final_iteration = math.ceil(c.get("lr_frames", 2_000_000) / 2048)
    lr = tf.keras.optimizers.schedules.PolynomialDecay(c.get("lr", 3e-4), final_iteration * EPOCHS * (N * T // MB), 0.0)
    opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

    @tf.function
    def train_mb(args):
        with tf.GradientTape() as tape:
            loss = loss_fn(*args)
        tv = loss_fn.trainable_variables
        g = tape.gradient(loss, tv, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gn = tf.linalg.global_norm(g)
        g, _ = tf.clip_by_global_norm(g, 0.5, use_norm=gn)
        opt.apply_gradients(zip(g, tv))
        return loss, gn

    def evaluate(n_ep=10):
        ev = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(n_ep)])
        o, _ = ev.reset(seed=10_000 + seed)
        ret, alive = np.zeros(n_ep), np.ones(n_ep, bool)
        for _ in range(1000):
            eo = utils.EnvOutput(tf.zeros(n_ep), tf.zeros(n_ep, bool), tf.constant(o, tf.float32), tf.zeros(n_ep, bool),
                                 tf.zeros(n_ep, tf.int32))
            ao, _ = agent((tf.zeros((n_ep, A)), eo), ())
            o, r, term, trunc, _ = ev.step(post(ao.action.numpy()))
            ret += r * alive
            alive &= ~(term | trunc)
            if not alive.any():
                break
        return float(ret.mean())

    r_prev = np.zeros(N, np.float32)
    d_prev = np.zeros(N, bool)
    curve, diag, t0, steps, it = [(0, 0.0)], [], time.time(), 0, 0
    next_eval = 100_000
    ep_ret, done_rets = np.zeros(N), []
    while steps < total:
        pass
        R, Dn, O, ACT, LOG, BASE = [r_prev], [d_prev], [obs], [], [], []
        for t in range(T):
            ao, _ = act_fn(mk_env_out(r_prev, d_prev, obs))
            a = ao.action.numpy()                       # stored action (pre-squash x for *_pre)
            obs, r, term, trunc, _ = envs.step(post(a))
            d = term | trunc
            ep_ret += r
            done_rets += list(ep_ret[d])
            ep_ret[d] = 0
            ACT.append(a); LOG.append(ao.policy_logits.numpy()); BASE.append(ao.baseline.numpy())
            R.append(r.astype(np.float32)); Dn.append(d); O.append(obs)
            r_prev, d_prev = r.astype(np.float32), d
        ao, _ = act_fn(mk_env_out(r_prev, d_prev, obs))     # bootstrap row (agent output at the last obs)
        ACT.append(ao.action.numpy()); LOG.append(ao.policy_logits.numpy()); BASE.append(ao.baseline.numpy())
        env_out = mk_env_out(np.stack(R), np.stack(Dn), np.stack(O))
        args = Unroll((), tf.zeros((T + 1, N, A)), env_out,
                      continuous_control_agent.AgentOutput(tf.constant(np.stack(ACT)), tf.constant(np.stack(LOG)),
                                                           tf.constant(np.stack(BASE))))
        DUMP[f'it{it}/obs'] = np.stack(O); DUMP[f'it{it}/act'] = np.stack(ACT); DUMP[f'it{it}/logits'] = np.stack(LOG)
        DUMP[f'it{it}/base'] = np.stack(BASE); DUMP[f'it{it}/rew'] = np.stack(R); DUMP[f'it{it}/done'] = np.stack(Dn)
        m0, s0 = agent.observation_normalizer.get_mean_std(); DUMP[f'it{it}/obs_mean_pre'] = np.asarray(m0); DUMP[f'it{it}/obs_std_pre'] = np.asarray(s0)
        for i, v in enumerate(loss_fn.trainable_variables):
            DUMP[f'it{it}/wpre_{i}'] = v.numpy().copy()
        agent.update_observation_normalization_statistics(env_out.observation[:-1])
        PERMS = []
        gns = []
        for e in range(EPOCHS):
            if e == 0 or c["mode"] == "split_with_advantage_recomputation":
                split = ptu.compute_advantages_and_split(loss_fn, args)
            perm = np.random.default_rng(seed * 100003 + it * 17 + e).permutation(N * T)
            PERMS.append(perm)
            for j in range(N * T // MB):
                idx = tf.constant(perm[j * MB:(j + 1) * MB])
                mb = tf.nest.map_structure(lambda x: tf.gather(x, idx, axis=1), split)
                _, gn = train_mb(mb)
                gns.append(float(gn))
        DUMP[f'it{it}/perms'] = np.stack(PERMS).astype(np.int32)
        for i, v in enumerate(loss_fn.trainable_variables):
            DUMP[f'it{it}/wpost_{i}'] = v.numpy().copy()
        m1, s1 = agent.observation_normalizer.get_mean_std(); DUMP[f'it{it}/obs_mean'] = np.asarray(m1); DUMP[f'it{it}/obs_std'] = np.asarray(s1)
        if pa is not None:
            m2, s2 = pa.mean_std_tracker.get_mean_std(); DUMP[f'it{it}/pa_mean'] = np.asarray(m2); DUMP[f'it{it}/pa_std'] = np.asarray(s2)
        DUMP[f'it{it}/opt_iter'] = np.int64(opt.iterations.numpy()); DUMP[f'it{it}/gn_mean'] = np.float32(np.mean(gns))
        steps += N * T
        it += 1
        print('[INFO] dumped iteration', it, flush=True)
        np.savez(out, **DUMP, cfg=np.array(str(c)))
        if False:
            curve.append((steps, evaluate()))
            sd = float(tf.reduce_mean(std_fn(agent.trainable_variables[0])))
            diag.append({"steps": steps, "std": sd, "grad_norm_mean": float(np.nanmean(gns)), "nonfinite": int(np.sum(~np.isfinite(gns))),
                         "train_ep_return": float(np.mean(done_rets)) if done_rets else None})
            done_rets = []
            print(f"[METRIC] official_loop env={env_name} cfg={cfg_name} seed={seed} steps={steps} eval={curve[-1][1]:.1f} "
                  f"train_ret={diag[-1]['train_ep_return']} std={sd:.3f} gn={diag[-1]['grad_norm_mean']:.3g} nonfinite={diag[-1]['nonfinite']} "
                  f"wall={time.time() - t0:.0f}", flush=True)
            next_eval += 100_000
            json.dump({"env": env_name, "cfg": cfg_name, "seed": seed, "curve": curve, "diag": diag,
                       "score": float(np.mean([r for s, r in curve[1:]])), "tf": tf.__version__,
                       "seed_rl_commit": "0e1e0ac9178a670ad1e1463baed92020e88905ec",
                       "dist_commit": "68c15aa" if pre else "0e1e0ac", "cfg_values": c, "N": N, "T": T, "wall_s": time.time() - t0},
                      open(out, "w"), indent=1)


if __name__ == "__main__":
    main()


