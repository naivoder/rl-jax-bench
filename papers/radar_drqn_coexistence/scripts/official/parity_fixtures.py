"""official-code check parity fixtures recorded from official / oracle code. Two parts, two venvs:

  part "cleanrl_env" (run with $OFFICIAL_ROOT/.venv-cleanrl: gymnasium 0.29.1, jax 0.4.8, CleanRL @ fe8d8a0 deps):
      CleanRL dqn_jax.py's OWN make_env (imported from cleanrl/cleanrl/dqn_jax.py, unmodified) inside
      gymnasium.vector.SyncVectorEnv exactly as dqn_jax.py builds it, stepped for 1200 steps with a fixed action
      sequence (a PD controller with 10 % random flips, so episodes both terminate and truncate at 500).
      Records the data path dqn_jax.py consumes: next_obs (auto-reset), real_next_obs (final_observation on
      truncation, as dqn_jax.py L193-196), rewards, terminations, truncations, and every
      final_info['episode']['r'] (the episodic_return dqn_jax.py logs), plus each reset observation.
      -> $OFFICIAL_ROOT/fixtures/cleanrl_env_traj.npz
  part "pomdp_lstm" (run with $OFFICIAL_ROOT/.venv-pomdp, torch, float64):
      the LSTM module pomdp-baselines' Critic_RNN builds (Critic_RNN(...).rnn, an nn.LSTM, official init) run on
      a random input sequence from a NONZERO initial (h, c), plus the same module stepped one input at a time.
      -> $OFFICIAL_ROOT/fixtures/pomdp_lstm.npz
Usage:
  $OFFICIAL_ROOT/.venv-cleanrl/bin/python scripts/official/parity_fixtures.py cleanrl_env
  $OFFICIAL_ROOT/.venv-pomdp/bin/python scripts/official/parity_fixtures.py pomdp_lstm
"""
import os
import sys

import numpy as np

HERE = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "official"))   # see setup.sh


def cleanrl_env():
    sys.path.insert(0, os.path.join(HERE, "cleanrl"))
    sys.argv = sys.argv[:1]
    import gymnasium as gym
    import importlib.util
    spec = importlib.util.spec_from_file_location("dqn_jax", os.path.join(HERE, "cleanrl", "cleanrl", "dqn_jax.py"))
    dqn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dqn)                       # top level only: the __main__ block does not run
    envs = gym.vector.SyncVectorEnv([dqn.make_env("CartPole-v1", 7, 0, False, "parity")])
    obs, _ = envs.reset(seed=7)
    rng = np.random.default_rng(7)
    rec = dict(obs=[], act=[], next_obs=[], real_next_obs=[], rew=[], term=[], trunc=[], ep_step=[], ep_ret=[], ep_len=[])
    reset_obs = [obs[0].copy()]
    for t in range(1200):
        o = obs[0]
        a = int(o[2] + 0.5 * o[3] + 0.05 * o[0] + 0.1 * o[1] > 0)          # PD controller
        if rng.random() < (0.1 if t < 700 else 0.45):                         # later: noisy enough to fall
            a = 1 - a
        actions = np.array([a])
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    rec["ep_step"].append(t)
                    rec["ep_ret"].append(float(info["episode"]["r"][0]) if np.ndim(info["episode"]["r"]) else float(info["episode"]["r"]))
                    rec["ep_len"].append(int(np.asarray(info["episode"]["l"]).ravel()[0]))
        real_next_obs = next_obs.copy()                                       # verbatim dqn_jax.py logic
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        # labelled addition: on termination dqn_jax.py stores next_obs (= the reset obs); record the true final obs too
        final_obs = infos["final_observation"][0] if (terminations[0] or truncations[0]) else next_obs[0]
        for k, v in (("obs", o.copy()), ("act", a), ("next_obs", np.asarray(final_obs, np.float32)),
                     ("real_next_obs", real_next_obs[0].copy()), ("rew", float(rewards[0])),
                     ("term", bool(terminations[0])), ("trunc", bool(truncations[0]))):
            rec[k].append(v)
        if terminations[0] or truncations[0]:
            reset_obs.append(next_obs[0].copy())
        obs = next_obs
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["reset_obs"] = np.asarray(reset_obs)
    path = os.path.join(HERE, "fixtures", "cleanrl_env_traj.npz")
    np.savez_compressed(path, **out)
    print(f"[INFO] wrote {path} episodes={len(rec['ep_ret'])} returns={rec['ep_ret']} n_trunc={int(out['trunc'].sum())} "
          f"n_term={int(out['term'].sum())} gymnasium={gym.__version__}")


def pomdp_lstm():
    import torch
    sys.path.insert(0, os.path.join(HERE, "pomdp-baselines"))
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    from policies.models.recurrent_critic import Critic_RNN
    from policies.rl.sacd import SACD
    critic = Critic_RNN(obs_dim=3, action_dim=4, encoder="lstm", algo=SACD(entropy_alpha=0.0, automatic_entropy_tuning=False),
                        action_embedding_size=8, observ_embedding_size=16, reward_embedding_size=4, rnn_hidden_size=12,
                        dqn_layers=[16, 16], rnn_num_layers=1)
    rnn = critic.rnn
    with torch.no_grad():
        for n, p in rnn.named_parameters():                   # official init zeroes biases; perturb so the bias path is tested
            if "bias" in n:
                p.add_(0.2 * torch.randn_like(p))
    T, B, D, H = 9, 3, rnn.input_size, rnn.hidden_size
    x = torch.randn(T, B, D)
    h0, c0 = 0.5 * torch.randn(1, B, H), 0.5 * torch.randn(1, B, H)
    with torch.no_grad():
        out_seq, (hT, cT) = rnn(x, (h0, c0))
        hs, (h, c) = [], (h0, c0)
        for t in range(T):                                    # stepwise: the act-with-carry path
            o, (h, c) = rnn(x[t:t + 1], (h, c))
            hs.append(o[0])
        out_step = torch.stack(hs)
    sd = {k: v.numpy() for k, v in rnn.state_dict().items()}
    path = os.path.join(HERE, "fixtures", "pomdp_lstm.npz")
    np.savez_compressed(path, x=x.numpy(), h0=h0.numpy()[0], c0=c0.numpy()[0], out_seq=out_seq.numpy(),
                        out_step=out_step.numpy(), hT=hT.numpy()[0], cT=cT.numpy()[0],
                        **{f"rnn/{k}": v for k, v in sd.items()})
    print(f"[INFO] wrote {path} D={D} H={H} T={T} B={B} torch={torch.__version__} "
          f"max|seq-step|={float((out_seq - out_step).abs().max()):.2e}")


if __name__ == "__main__":
    {"cleanrl_env": cleanrl_env, "pomdp_lstm": pomdp_lstm}[sys.argv[1]]()
