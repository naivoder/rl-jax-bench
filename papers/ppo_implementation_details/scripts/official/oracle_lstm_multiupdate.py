"""ORACLE side of the follow-up multi-update LSTM differential test, torch CPU.

Replays U whole PPO updates of CleanRL's LSTM logic (harness/cleanrl_ppo_lstm_maskedcartpole.py, or the L3-off file
harness/cleanrl_ppo_lstm_maskedcartpole_noreset.py) from OUR initial weights, teacher-forced on our env stream (obs,
done flags, training rewards) and our sampled actions, with our per-epoch env-column permutations substituted for
np.random.shuffle. The LSTM state is CleanRL's OWN: zero at the start, carried step to step through
agent.get_action_and_value, and carried across updates as initial_lstm_state = next_lstm_state (as in the file).
The GAE block and the whole update block (flatten -> epochs -> minibatches -> loss -> backward -> clip -> Adam) are
extracted VERBATIM from the file and exec'd. Only the env step is replaced (teacher forcing), and the action is passed
in instead of sampled.
Usage: .venv/bin/python -I scripts/official/oracle_lstm_multiupdate.py <ours_lstm_multi_*.npz> <out.npz>
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

HERE = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # oracle_difftest.py sits next to this file
from oracle_difftest import FakeEnvs, T, block, load_sd  # noqa: E402

torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)

src_npz, dst = sys.argv[1:3]
z = np.load(src_npz)
reset = bool(int(z["meta/reset"]))
path = os.path.join(HERE, "harness", "cleanrl_ppo_lstm_maskedcartpole.py" if reset
                    else "cleanrl_ppo_lstm_maskedcartpole_noreset.py")
spec = importlib.util.spec_from_file_location("cl_lstm", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
U, N, M = int(z["meta/U"]), int(z["meta/N"]), int(z["meta/M"])
envs = FakeEnvs(4, 2, False)
agent = mod.Agent(envs)
load_sd(agent, z)
args = types.SimpleNamespace(num_envs=N, num_steps=M, batch_size=N * M, num_minibatches=int(z["meta/n_mb"]),
                             update_epochs=int(z["meta/K"]), learning_rate=float(z["meta/lr0"]), anneal_lr=True,
                             num_iterations=U, gamma=float(z["meta/gamma"]), gae_lambda=float(z["meta/lam"]),
                             norm_adv=True, clip_coef=float(z["meta/clip_coef"]), clip_vloss=True,
                             ent_coef=float(z["meta/ent_coef"]), vf_coef=float(z["meta/vf_coef"]),
                             max_grad_norm=float(z["meta/max_grad_norm"]), target_kl=None)
optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
gae_code, gae_lines = block(path, "next_value = agent.get_value(", "returns = advantages + values",
                            back_to="with torch.no_grad():")
upd_code, upd_lines = block(path, "b_obs = obs.reshape(", "break")
out = {"meta/gae_lines": np.array(gae_lines), "meta/update_lines": np.array(upd_lines), "meta/file": path}

next_lstm_state = (torch.zeros(1, N, 128), torch.zeros(1, N, 128))
for u in range(U):
    p = f"u{u}/"
    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
    iteration = u + 1
    frac = 1.0 - (iteration - 1.0) / args.num_iterations
    optimizer.param_groups[0]["lr"] = frac * args.learning_rate
    obs, dones = T(z[p + "obs"]), T(z[p + "done"])
    actions, rewards = T(z[p + "action"]).float(), T(z[p + "reward"])
    logprobs, values = torch.zeros(M, N), torch.zeros(M, N)
    for step in range(M):  # rollout, teacher-forced: next_obs/next_done come from our stream
        next_obs, next_done = obs[step], dones[step]
        with torch.no_grad():
            _, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                next_obs, next_lstm_state, next_done, actions[step].long())
        values[step], logprobs[step] = value.flatten(), logprob
    next_obs, next_done = T(z[p + "next_obs"]), T(z[p + "next_done"])
    perms = iter(np.asarray(z[p + "perms"]))

    def shuffle(x, _perms=perms):
        x[:] = next(_perms)

    fake_np = types.SimpleNamespace(arange=np.arange, random=types.SimpleNamespace(shuffle=shuffle))
    ns = dict(torch=torch, nn=torch.nn, np=fake_np, agent=agent, args=args, envs=envs, optimizer=optimizer,
              device="cpu", obs=obs, dones=dones, actions=actions, rewards=rewards, logprobs=logprobs,
              values=values, next_obs=next_obs, next_done=next_done, initial_lstm_state=initial_lstm_state,
              next_lstm_state=next_lstm_state)
    exec(gae_code, ns)
    exec(upd_code, ns)
    out.update({p + "logp": logprobs, p + "value": values, p + "final_h": next_lstm_state[0][0],
                p + "final_c": next_lstm_state[1][0], p + "init_h": initial_lstm_state[0][0],
                p + "init_c": initial_lstm_state[1][0], p + "adv": ns["advantages"], p + "ret": ns["returns"],
                p + "lr": np.float32(optimizer.param_groups[0]["lr"])})
    out.update({p + "params_after/" + n: q.detach().clone() for n, q in agent.named_parameters()})
np.savez(dst, **{k: (v.detach().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in out.items()})
print("wrote", dst, "keys", len(out), "gae lines", gae_lines, "update lines", upd_lines)
