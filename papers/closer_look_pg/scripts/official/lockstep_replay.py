"""Lockstep replay : K recorded training batches of the JAX re-implementation replayed through
the OFFICIAL MadryLab/implementation-matters@5ee6ecb primitives with the official persistent state.

Per iteration k (order of agent.Trainer.take_steps):
  values = official value net on the batch states (agent.collect_saps); adv, ret = agent.advantage_and_return;
  steps.value_step(...) with the recorded value permutations; steps.ppo_step / steps.trpo_step with the recorded
  minibatch permutations / Fisher subset and the recorded old log-probs; POLICY/VALUE LambdaLR .step() if annealing.
The optimisers (Adam, eps 1e-5) and schedulers are created ONCE (agent.Trainer.__init__) and persist across iterations.
Only official functions are executed; np.random.shuffle / np.random.choice are replayed as in oracle_harness.py.

Run with this directory's isolated venv:  .venv/bin/python lockstep_replay.py <in.npz> <out.npz>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I does not prepend the script directory
import numpy as np
import torch as ch
from torch import optim

from oracle_harness import Replay, export_mlp, load_adv_fn, set_mlp  # noqa: E402  (same directory)
from policy_gradients import steps, torch_utils  # noqa: E402
from policy_gradients.models import CtsPolicy, ValueDenseNet  # noqa: E402

ch.set_default_dtype(ch.float32)


def main(inp, outp):
    d = dict(np.load(inp))
    c = json.loads(str(d["config_json"]))
    K = int(c["iters"])
    obs_dim, act_dim = d["obs"].shape[-1], d["act"].shape[-1]
    init = c["init"]
    net = CtsPolicy(obs_dim, act_dim, init)
    set_mlp(net.affine_layers, net.final_mean, {k.replace("pi0_", "pi_"): v for k, v in d.items()}, "pi_")
    with ch.no_grad():
        net.log_stdev.copy_(ch.tensor(d["pi0_log_std"]))
    vnet = ValueDenseNet(obs_dim, init)
    set_mlp(vnet.affine_layers, vnet.final, {k.replace("v0_", "v_"): v for k, v in d.items()}, "v_")
    val_opt = optim.Adam(vnet.parameters(), lr=float(c["value_lr"]), eps=1e-5)
    PV = torch_utils.Parameters({"share_weights": False, "num_minibatches": int(c["value_minibatches"]),
                                 "val_epochs": int(c["value_epochs"]), "value_calc": "gae",
                                 "value_clipping": bool(c["value_clipping"]), "clip_val_eps": float(c["clip_val_eps"])})
    if c["alg"] == "ppo":
        pol_opt = optim.Adam(net.parameters(), lr=float(c["policy_lr"]), eps=1e-5)
        PP = torch_utils.Parameters({"share_weights": False, "ppo_epochs": int(c["policy_epochs"]),
                                     "num_minibatches": int(c["minibatches"]), "clip_eps": float(c["clip_eps"]),
                                     "entropy_coeff": 0.0, "clip_grad_norm": -1})
        PP.POLICY_ADAM = pol_opt
    else:
        PP = torch_utils.Parameters({"fisher_frac_samples": float(c["fisher_fraction"]), "damping": float(c["damping"]),
                                     "cg_steps": int(c["cg_steps"]), "max_kl": float(c["max_kl"]),
                                     "max_backtrack": int(c["backtrack_steps"]), "should_log_kl": False})
    scheds = []
    if c["anneal"]:
        lam = lambda f: 1 - f / float(c["train_iterations"])       # agent.Trainer.__init__ (TRAIN_STEPS)
        scheds = [optim.lr_scheduler.LambdaLR(val_opt, lr_lambda=lam)]
        if c["alg"] == "ppo":
            scheds.append(optim.lr_scheduler.LambdaLR(pol_opt, lr_lambda=lam))
        for _ in range(int(c.get("start_it", 0))):                 # resume at iteration start_it (agent_from_data)
            for sch in scheds:
                sch.step()
    adv_fn = load_adv_fn()
    self_ = type("S", (), {"GAMMA": float(c["gamma"]), "LAMBDA": float(c["lam"])})()
    out = {k: [] for k in ("adv", "ret", "logp", "values")}
    after = {}
    orig_shuffle, orig_choice = np.random.shuffle, np.random.choice
    for k in range(K):
        obs, act = ch.tensor(d["obs"][k]), ch.tensor(d["act"][k])
        rew, nd = ch.tensor(d["rew"][k]), ch.tensor(d["not_done"][k])
        with ch.no_grad():
            values = vnet(obs).squeeze(-1)
            logp_own = net.get_loglikelihood(net(obs), act)
        adv, ret = adv_fn(self_, rew[None], values[None], nd[None])
        adv, ret = adv[0], ret[0]
        out["adv"].append(adv.numpy()); out["ret"].append(ret.numpy()); out["logp"].append(logp_own.numpy())
        out["values"].append(values.numpy())
        np.random.shuffle = Replay(perms=d["v_perms"][k]).shuffle
        try:
            steps.value_step(obs, ret, adv, nd, vnet, val_opt, PV, None)
        finally:
            np.random.shuffle = orig_shuffle
        lp_old = ch.tensor(d["logp"][k])                   # the batch's recorded log pi_old (== logp_own if in sync)
        if c["alg"] == "ppo":
            np.random.shuffle = Replay(perms=d["p_perms"][k]).shuffle
            try:
                steps.ppo_step(obs, act, lp_old, None, None, None, adv, net, PP, None, k)
            finally:
                np.random.shuffle = orig_shuffle
        else:
            np.random.choice = Replay(subset=d["subset"][k]).choice
            try:
                steps.trpo_step(obs, act, lp_old, None, None, None, adv, net, PP, None, k)
            finally:
                np.random.choice = orig_choice
        for s in scheds:
            s.step()
        snap = {**export_mlp(net.affine_layers, net.final_mean, "pi_"),
                **export_mlp(vnet.affine_layers, vnet.final, "v_")}
        snap["pi_log_std"] = net.log_stdev.detach().numpy().copy()
        for key, v in snap.items():
            after.setdefault(f"after_{key}", []).append(v)
    res = {k: np.stack(v) for k, v in out.items()}
    res.update({k: np.stack(v) for k, v in after.items()})
    np.savez(outp, **res)
    print("lockstep replay ok", K, "iterations; torch", ch.__version__)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
