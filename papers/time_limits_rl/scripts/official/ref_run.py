"""Reference run (step 3): the authors' library (Tonic @0e20c89, torch PPO) on the paper's headline
continuous tasks, gym MuJoCo v4 (the paper used -v1 under mujoco-py 0.5; v1 is not installable here).

Driver loop mirrors tonic/utils/trainer.py:33-71 (start, agent.step, environment.step, agent.update, episode
bookkeeping on infos['resets']); evaluation follows the paper's Figure 6 metric as our E7 does (discounted gamma=0.99
return over a horizon H = 3000 with NO time limit, terminations end the episode, 5 episodes, stochastic
agent.test_step -- Tonic's own test semantics).

  .venv-ref/bin/python -I scripts/official/ref_run.py --exp ip|hopper --arm <arm> --config tonic|paper --seed N --steps N --out f
Configs:
  tonic : Tonic's PPO defaults (Segment 4096, 80 full-batch iterations, actor KL stop 0.015, lambda 0.97, actor Adam
          3e-4, critic Adam 1e-3, MeanStd obs normalizer) -- only the discount is set per arm.
  paper : Tonic's PPO set to the paper's stated (Baselines / Schulman et al. 2017 Table 3) numbers where they differ:
          horizon 2048, 10 epochs of minibatch 64, lambda 0.95, both Adams at 3e-4, grad-norm clip 0.5, no KL stop.
          (Still Tonic's separate actor/critic losses, per-segment advantage normalisation, no reward scaling.)
  paper_vn : 'paper' + Baselines VecNormalize reward scaling (baselines/common/vec_env/vec_normalize.py step_wait:
          ret = ret*0.99 + r; ret_rms.update(ret); r / sqrt(var + 1e-8) clipped to +-10; ret reset at episode ends),
          applied between the env and agent.update -- a HARNESS addition (Tonic has no such scaler; its Return
          normalizer asserts gamma < 1), because the paper's PPO ran in Baselines' run_mujoco with VecNormalize.
          Episode returns / metrics are always recorded on the RAW rewards.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.environ.get("OFFICIAL_ROOT", "external/official"), "tonic"))  # Tonic clone
import gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tonic  # noqa: E402
import tonic.torch  # noqa: E402

torch.set_num_threads(1)

IP_ARMS = {  # time-limited task: the timeout IS terminal (terminal_timeouts=True); time feature on/off
    "ppo_g099": dict(time_feature=False, gamma=0.99), "ta_ppo_g099": dict(time_feature=True, gamma=0.99),
    "ppo_g1": dict(time_feature=False, gamma=1.0), "ta_ppo_g1": dict(time_feature=True, gamma=1.0)}
HOP_ARMS = {  # time-unlimited task trained with a time limit T
    "peb_ppo_T300": dict(T=300, peb=True), "ppo_T300": dict(T=300, peb=False), "ppo_T1000": dict(T=1000, peb=False)}


def make_env(exp, arm):
    if exp == "ip":
        a = IP_ARMS[arm]
        return lambda: tonic.environments.Gym("InvertedPendulum-v4", terminal_timeouts=True,
                                              time_feature=a["time_feature"])
    a = HOP_ARMS[arm]
    if a["peb"]:  # TimeLimit stripped, Sequential resets at T without a termination -> bootstrapped
        return lambda: tonic.environments.Gym("Hopper-v4", terminal_timeouts=False, max_episode_steps=a["T"])
    # standard agent: gym's TimeLimit(T) sets done at T -> Tonic records it as a termination
    return lambda: tonic.environments.builders.build_environment(
        lambda name: gym.make(name, max_episode_steps=a["T"]), "Hopper-v4", terminal_timeouts=True)


class RewardScaler:
    """Baselines RunningMeanStd(shape=()) + VecNormalize reward path (epsilon 1e-8, cliprew 10, gamma 0.99)."""

    def __init__(self, n, gamma=0.99):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4
        self.ret = np.zeros(n)
        self.gamma = gamma

    def __call__(self, rewards, resets):
        self.ret = self.ret * self.gamma + rewards
        x = self.ret
        bm, bv, bc = x.mean(), x.var(), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot
        out = np.clip(rewards / np.sqrt(self.var + 1e-8), -10.0, 10.0).astype(np.float32)
        self.ret[resets] = 0.0
        return out


def make_agent(config, gamma):
    if config == "tonic":
        return tonic.torch.agents.PPO(replay=tonic.replays.Segment(discount_factor=gamma))
    return tonic.torch.agents.PPO(
        replay=tonic.replays.Segment(size=2048, batch_iterations=10, batch_size=64, discount_factor=gamma,
                                     trace_decay=0.95),
        actor_updater=tonic.torch.updaters.ClippedRatio(kl_threshold=float("inf"), gradient_clip=0.5),
        critic_updater=tonic.torch.updaters.VRegression(
            optimizer=lambda p: torch.optim.Adam(p, lr=3e-4), gradient_clip=0.5))


def evaluate(agent, seed, steps, H=3000, n_ep=5, gamma=0.99):
    env = tonic.environments.Gym("Hopper-v4", terminal_timeouts=False)
    env.seed(seed)
    out = []
    for _ in range(n_ep):
        o, G, Gu, k = env.reset(), 0.0, 0.0, 0
        while k < H:
            a = agent.test_step(o[None].astype(np.float32), steps)[0]
            o, r, d, _ = env.step(a)
            G += gamma ** k * r; Gu += r; k += 1
            if d:
                break
        out.append(dict(disc=G, undisc=Gu, length=k))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True); p.add_argument("--arm", required=True)
    p.add_argument("--config", default="tonic"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100000); p.add_argument("--eval_every", type=int, default=100000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    gamma = IP_ARMS[a.arm]["gamma"] if a.exp == "ip" else 0.99
    env = tonic.environments.distribute(make_env(a.exp, a.arm))
    env.initialize(seed=a.seed)
    agent = make_agent(a.config, gamma)
    agent.initialize(observation_space=env.observation_space, action_space=env.action_space, seed=a.seed)
    obs = env.start()
    scaler = RewardScaler(len(obs)) if a.config == "paper_vn" else None
    score, length, episodes, evals = 0.0, 0, [], []
    t0 = time.time()
    for steps in range(a.steps):
        actions = agent.step(obs, steps)
        obs, infos = env.step(actions)
        if scaler is None:
            agent.update(**infos, steps=steps)
        else:
            agent.update(**{**infos, "rewards": scaler(infos["rewards"], infos["resets"])}, steps=steps)
        score += infos["rewards"][0]; length += 1
        if infos["resets"][0]:
            episodes.append([steps + 1, float(score), length, bool(infos["terminations"][0])])
            score, length = 0.0, 0
        if a.exp == "hopper" and ((steps + 1) % a.eval_every == 0 or steps + 1 == a.steps):
            ev = evaluate(agent, a.seed + 10000, steps + 1)
            evals.append(dict(step=steps + 1, episodes=ev))
            print(f"[METRIC] step={steps + 1} disc={np.mean([e['disc'] for e in ev]):.2f} "
                  f"len={np.mean([e['length'] for e in ev]):.0f}", flush=True)
    last = [e[1] for e in episodes if e[0] > 0.9 * a.steps]
    res = dict(exp=a.exp, arm=a.arm, config=a.config, seed=a.seed, steps=a.steps, gamma=gamma,
               last10_return=float(np.mean(last)) if last else None, n_last10=len(last), episodes=episodes,
               evals=evals, wall_s=time.time() - t0,
               versions=dict(gym=gym.__version__, torch=torch.__version__, numpy=np.__version__,
                             tonic="0e20c894ee68278ab68322de61bb2c7204a11d5f"))
    json.dump(res, open(a.out, "w"))
    print(f"[INFO] done {a.exp} {a.arm} {a.config} seed={a.seed} last10={res['last10_return']} "
          f"wall={res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
