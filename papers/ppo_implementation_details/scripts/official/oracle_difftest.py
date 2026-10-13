"""ORACLE side of the small-scale differential tests (torch, CPU). Runs the authors' code on our exported inputs.

* Agent classes are imported from the official files (CleanRL @fe8d8a0 ppo.py / ppo_continuous_action.py /
  ppo_atari_lstm.py; the post's repo @fbef824 ppo_shared.py).
* The GAE block and the minibatch loss/backward/clip/Adam block are extracted VERBATIM from the same file's source
  (located by content anchors) and exec'd with the batch variables bound -- nothing is re-implemented here.
* LSTM: CleanRL's Agent with its Nature CNN replaced by Linear(obs,64)+Tanh and nn.LSTM(64,128); the
  official get_states divides by 255, so the obs are fed x255.
Usage: .venv/bin/python -I scripts/official/oracle_difftest.py <dir with ours_*.npz>
"""
import importlib.util
import os
import sys
import textwrap
import types

import numpy as np
import torch

HERE = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
CLEANRL = os.path.join(HERE, "cleanrl", "cleanrl")
POSTREPO = os.path.join(HERE, "ppo-implementation-details")
sys.path.insert(0, os.path.join(HERE, "cleanrl"))  # for cleanrl_utils (atari wrappers import; unused here)
torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)


def load_module(path, name):
    if "gym" not in sys.modules:  # ppo_shared.py imports gym (0.21); only spaces metadata are used -> gymnasium
        import gymnasium
        sys.modules["gym"] = gymnasium
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def block(path, start, end, back_to=None):
    lines = open(path).read().split("\n")
    i = next(k for k, l in enumerate(lines) if start in l)
    if back_to is not None:
        while back_to not in lines[i]:
            i -= 1
    j = next(k for k in range(i, len(lines)) if end in lines[k])
    return textwrap.dedent("\n".join(lines[i:j + 1])), (i + 1, j + 1)


class FakeSpace:
    def __init__(self, shape=None, n=None):
        self.shape, self.n = shape, n


class FakeEnvs:
    def __init__(self, obs_dim, n_out, cont):
        self.single_observation_space = FakeSpace((obs_dim,))
        self.single_action_space = FakeSpace((n_out,)) if cont else FakeSpace((), n_out)


def T(x):
    return torch.tensor(np.asarray(x))


def load_sd(agent, z):
    sd = {k[3:]: T(z[k]) for k in z.files if k.startswith("sd/")}
    missing = set(agent.state_dict()) ^ set(sd)
    assert not missing, missing
    agent.load_state_dict(sd)


def args_ns(z, **kw):
    a = types.SimpleNamespace(gamma=0.99, gae_lambda=0.95, num_steps=128, clip_coef=float(z["meta/clip_coef"]),
                              ent_coef=float(z["meta/ent_coef"]), vf_coef=float(z["meta/vf_coef"]),
                              clip_vloss=bool(z["meta/clip_vloss"]), norm_adv=True, max_grad_norm=0.5, gae=True)
    a.__dict__.update(kw)
    return a


def nn_proxy(rec):
    real = torch.nn.utils.clip_grad_norm_

    def clip(params, max_norm):
        params = list(params)
        rec["grad"] = {n: p.grad.detach().clone() for n, p in rec["named"]}
        out = real(params, max_norm)
        rec["gradclip"] = {n: p.grad.detach().clone() for n, p in rec["named"]}
        return out
    return types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=clip))


def ff(name, src_file, mod):
    z = np.load(os.path.join(DIR, f"ours_{name}.npz"))
    cont, n_out, d = bool(z["meta/cont"]), int(z["meta/n_out"]), int(z["meta/obs_dim"])
    agent = mod.Agent(FakeEnvs(d, n_out, cont))
    load_sd(agent, z)
    out = {}
    with torch.no_grad():
        act = T(z["fwd/act"]) if cont else T(z["fwd/act"]).long()
        _, lp, ent, v = agent.get_action_and_value(T(z["fwd/obs"]), act)
        out.update({"fwd/logp": lp, "fwd/entropy": ent, "fwd/value": v.view(-1)})
        out["fwd/out0"] = agent.actor_mean(T(z["fwd/obs"])) if cont else (
            agent.actor(T(z["fwd/obs"])) if not hasattr(agent, "network") else agent.actor(agent.network(T(z["fwd/obs"]))))
    # GAE block, verbatim
    code, rng = block(src_file, "next_value = agent.get_value(next_obs)", "returns = advantages + values",
                      back_to="with torch.no_grad():")
    ns = dict(torch=torch, agent=agent, next_obs=T(z["gae/next_obs"]), next_done=T(z["gae/next_done"]),
              rewards=T(z["gae/rewards"]), dones=T(z["gae/dones"]), values=T(z["gae/values"]), args=args_ns(z),
              device="cpu")
    exec(code, ns)
    out["gae/adv"], out["gae/ret"] = ns["advantages"], ns["returns"]
    out["meta/gae_lines"] = np.array(rng)
    # minibatch loss -> backward -> clip -> Adam, verbatim, two consecutive minibatches
    code, rng = block(src_file, "_, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds]",
                      "optimizer.step()")
    out["meta/loss_lines"] = np.array(rng)
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(z["b/lr"]), eps=1e-5)  # official: Adam(eps=1e-5)
    perm = np.asarray(z["b/perm"])
    rec = {"named": list(agent.named_parameters())}
    for step in range(2):
        mb_inds = perm[step * 128:(step + 1) * 128]
        ns = dict(torch=torch, nn=nn_proxy(rec), agent=agent, args=args_ns(z), optimizer=optimizer,
                  b_obs=T(z["b/obs"]), b_actions=T(z["b/actions"]) if cont else T(z["b/actions"]).float(),
                  b_logprobs=T(z["b/logprobs"]), b_advantages=T(z["b/advantages"]), b_returns=T(z["b/returns"]),
                  b_values=T(z["b/values"]), mb_inds=mb_inds, clipfracs=[])
        exec(code, ns)
        pre = f"step{step}/"
        out.update({pre + "pg_loss": ns["pg_loss"], pre + "v_loss": ns["v_loss"], pre + "entropy": ns["entropy_loss"],
                    pre + "loss": ns["loss"], pre + "k1": ns["old_approx_kl"], pre + "k3": ns["approx_kl"],
                    pre + "clipfrac": np.float32(ns["clipfracs"][0])})
        out.update({pre + "grad/" + n: g for n, g in rec["grad"].items()})
        out.update({pre + "gradclip/" + n: g for n, g in rec["gradclip"].items()})
        out.update({pre + "params_after/" + n: p.detach().clone() for n, p in agent.named_parameters()})
    # lr schedule (official formula lines, verbatim)
    code, _ = block(src_file, "frac = 1.0 - (", "lrnow = frac * args.learning_rate")
    lrs = []
    for iteration in range(1, 11):  # CleanRL names it iteration/num_iterations, the post's files update/num_updates
        ns = dict(iteration=iteration, update=iteration, num_updates=10,
                  args=types.SimpleNamespace(num_iterations=10, learning_rate=2.5e-4))
        exec(code, ns)
        lrs.append(ns["lrnow"])
    out["lr/oracle"] = np.array(lrs)
    save(name, out)


def lstm(mod):
    src = os.path.join(CLEANRL, "ppo_atari_lstm.py")
    z = np.load(os.path.join(DIR, "ours_lstm.npz"))
    agent = mod.Agent(FakeEnvs(4, 2, False))
    agent.network = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.Tanh())  # encoder
    agent.lstm = torch.nn.LSTM(64, 128)
    load_sd(agent, z)
    obs, dones = T(z["seq/obs"]), T(z["seq/dones"])
    M, N, d = obs.shape
    h0 = (T(z["seq/h0"])[None], T(z["seq/c0"])[None])
    out = {}
    with torch.no_grad():  # official get_states over the whole (M, N) sequence (flattened t-major, as b_obs is)
        hidden, (hT, cT) = agent.get_states(obs.reshape(M * N, d) * 255.0, h0, dones.reshape(-1))
        out["seq/logits"] = agent.actor(hidden).reshape(M, N, 2)
        out["seq/value"] = agent.critic(hidden).reshape(M, N)
        out["seq/hT"], out["seq/cT"] = hT[0], cT[0]
    code, rng = block(src, "_, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(", "optimizer.step()")
    out["meta/loss_lines"] = np.array(rng)
    flatinds = np.arange(M * N).reshape(M, N)
    mbenvinds = np.asarray(z["b/mbenvinds"])
    mb_inds = flatinds[:, mbenvinds].ravel()  # official index construction
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(z["b/lr"]), eps=1e-5)
    rec = {"named": list(agent.named_parameters())}
    a = types.SimpleNamespace(clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, clip_vloss=True, norm_adv=True,
                              max_grad_norm=0.5)
    ns = dict(torch=torch, nn=nn_proxy(rec), agent=agent, args=a, optimizer=optimizer,
              b_obs=obs.reshape(M * N, d) * 255.0, b_actions=T(z["b/actions"]).reshape(-1).float(),
              b_logprobs=T(z["b/logprobs"]).reshape(-1), b_advantages=T(z["b/advantages"]).reshape(-1),
              b_returns=T(z["b/returns"]).reshape(-1), b_values=T(z["b/values"]).reshape(-1),
              b_dones=dones.reshape(-1), mb_inds=mb_inds, mbenvinds=mbenvinds, clipfracs=[],
              initial_lstm_state=h0)
    exec(code, ns)
    out.update({"step0/pg_loss": ns["pg_loss"], "step0/v_loss": ns["v_loss"], "step0/entropy": ns["entropy_loss"],
                "step0/loss": ns["loss"], "step0/k1": ns["old_approx_kl"], "step0/k3": ns["approx_kl"],
                "step0/clipfrac": np.float32(ns["clipfracs"][0])})
    out.update({"step0/grad/" + n: g for n, g in rec["grad"].items()})
    out.update({"step0/params_after/" + n: p.detach().clone() for n, p in agent.named_parameters()})
    save("lstm", out)


def save(name, out):
    conv = {k: (v.detach().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in out.items()}
    np.savez(os.path.join(DIR, f"oracle_{name}.npz"), **conv)
    print(f"[INFO] oracle wrote oracle_{name}.npz keys={len(conv)}")


if __name__ == "__main__":
    DIR = sys.argv[1]
    m_ppo = load_module(os.path.join(CLEANRL, "ppo.py"), "cleanrl_ppo")
    m_cont = load_module(os.path.join(CLEANRL, "ppo_continuous_action.py"), "cleanrl_ppo_cont")
    m_shared = load_module(os.path.join(POSTREPO, "ppo_shared.py"), "post_ppo_shared")
    m_lstm = load_module(os.path.join(CLEANRL, "ppo_atari_lstm.py"), "cleanrl_ppo_lstm")
    ff("separate", os.path.join(CLEANRL, "ppo.py"), m_ppo)
    ff("separate_novclip", os.path.join(CLEANRL, "ppo.py"), m_ppo)
    ff("shared", os.path.join(POSTREPO, "ppo_shared.py"), m_shared)
    ff("gaussian", os.path.join(CLEANRL, "ppo_continuous_action.py"), m_cont)
    lstm(m_lstm)
