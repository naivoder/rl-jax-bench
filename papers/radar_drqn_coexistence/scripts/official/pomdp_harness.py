"""Oracle harness: pomdp-baselines @ e7c19c3 recurrent discrete critic -> sequence-target fixture for our DRQN core.

Runs UNMODIFIED official code from $OFFICIAL_ROOT/pomdp-baselines (float64, CPU):
  policies.models.recurrent_critic.Critic_RNN (LSTM, discrete actions) and policies.rl.sacd.SACD.critic_loss,
  plus the masked TD loss of ModelFreeOffPolicy_Separate_RNN.forward (policy_rnn.py L151-174), transcribed
  verbatim below because that method also runs the actor update.
Labelled harness choices that turn SAC-discrete's critic target into the double-DQN sequence target:
  * alpha_entropy = 0 (automatic_entropy_tuning=False, entropy_alpha=0.0);
  * qf2 := qf1 (weights copied) in both the online and the target critic, so min(q1, q2) = q1;
  * a stub actor returning one-hot(argmax_a Q1_online(h_t, a)) at every position (T+1), so
    E_{a~pi}[Q_tgt(h', a)] = Q_tgt(h', argmax Q_on(h', .)), i.e. Eq. 14 / van Hasselt et al.
  * the target critic's weights are the online ones plus N(0, 0.3^2) noise, so the online and target argmaxes differ.
Batch (time-major as the official code): actions (T+1,B,A) one-hot with a zero row 0, rewards (T+1,B,1),
observs (T+1,B,obs), dones (T+1,B,1) with a terminal inside, masks (T,B,1) with padded tails.
Writes $OFFICIAL_ROOT/fixtures/pomdp_seq_target.npz (inputs, both state dicts, q1_pred, q_target, qf1_loss, d qf1_loss / d params).
Run: $OFFICIAL_ROOT/.venv-pomdp/bin/python scripts/official/pomdp_harness.py
"""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "official"))   # see setup.sh
REPO = os.path.join(HERE, "pomdp-baselines")
sys.path.insert(0, REPO)
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

from policies.models.recurrent_critic import Critic_RNN  # noqa: E402
from policies.rl.sacd import SACD  # noqa: E402

OBS, A, T, B, H = 3, 4, 6, 5, 12
GAMMA = 0.9


def main():
    algo = SACD(entropy_alpha=0.0, automatic_entropy_tuning=False)
    critic = Critic_RNN(obs_dim=OBS, action_dim=A, encoder="lstm", algo=algo, action_embedding_size=8,
                        observ_embedding_size=16, reward_embedding_size=4, rnn_hidden_size=H, dqn_layers=[16, 16],
                        rnn_num_layers=1)
    critic.qf2.load_state_dict(critic.qf1.state_dict())
    # official LSTM init is orthogonal weights / zero bias; perturb the biases so the bias path is exercised too
    with torch.no_grad():
        for n, p in critic.rnn.named_parameters():
            if "bias" in n:
                p.add_(0.1 * torch.randn_like(p))
    target = copy.deepcopy(critic)
    with torch.no_grad():
        for p in target.parameters():
            p.add_(0.3 * torch.randn_like(p))
    target.qf2.load_state_dict(target.qf1.state_dict())

    rng = np.random.default_rng(0)
    act_idx = rng.integers(0, A, (T, B))
    actions = np.zeros((T + 1, B, A))
    for t in range(T):
        actions[t + 1, np.arange(B), act_idx[t]] = 1.0
    rewards = np.zeros((T + 1, B, 1))
    rewards[1:] = rng.normal(size=(T, B, 1))
    observs = rng.normal(size=(T + 1, B, OBS))
    dones = np.zeros((T + 1, B, 1))
    dones[3, 1] = 1.0          # terminal at transition t = 2 of sequence 1
    dones[T, 2] = 1.0
    masks = np.ones((T, B, 1))
    masks[4:, 1] = 0.0         # sequence 1 ends after its terminal (padded tail)
    masks[5:, 3] = 0.0
    tt = {k: torch.tensor(v) for k, v in dict(actions=actions, rewards=rewards, observs=observs, dones=dones,
                                               masks=masks).items()}

    class GreedyStubActor:
        def __call__(self, prev_actions, rewards, observs):
            with torch.no_grad():
                q1, _ = critic(prev_actions=prev_actions, rewards=rewards, observs=observs,
                               current_actions=torch.zeros(observs.shape[0], observs.shape[1], A))
                probs = torch.nn.functional.one_hot(q1.argmax(-1), A).double()
            return probs, torch.zeros_like(probs)

    (q1_pred, q2_pred), q_target = algo.critic_loss(
        markov_actor=False, markov_critic=False, actor=GreedyStubActor(), actor_target=None, critic=critic,
        critic_target=target, observs=tt["observs"], actions=tt["actions"], rewards=tt["rewards"], dones=tt["dones"],
        gamma=GAMMA)
    # --- verbatim from policy_rnn.py ModelFreeOffPolicy_Separate_RNN.forward (L151, L171-174) ---
    num_valid = torch.clamp(tt["masks"].sum(), min=1.0)
    q1_pred, q2_pred = q1_pred * tt["masks"], q2_pred * tt["masks"]
    q_target = q_target * tt["masks"]
    qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid
    # ---
    critic.zero_grad()
    qf1_loss.backward()
    grads = {f"grad/{n}": p.grad.detach().numpy().copy() for n, p in critic.named_parameters()
             if p.grad is not None}
    out = dict(actions=actions, rewards=rewards, observs=observs, dones=dones, masks=masks, act_idx=act_idx,
               q1_pred=q1_pred.detach().numpy(), q_target=q_target.detach().numpy(), qf1_loss=qf1_loss.item(),
               gamma=GAMMA, **grads)
    out.update({f"online/{k}": v.numpy() for k, v in critic.state_dict().items()})
    out.update({f"target/{k}": v.numpy() for k, v in target.state_dict().items()})
    os.makedirs(os.path.join(HERE, "fixtures"), exist_ok=True)
    path = os.path.join(HERE, "fixtures", "pomdp_seq_target.npz")
    np.savez_compressed(path, **out)
    print(f"[INFO] wrote {path} loss={qf1_loss.item():.10f} params={sorted(critic.state_dict())}")


if __name__ == "__main__":
    main()
