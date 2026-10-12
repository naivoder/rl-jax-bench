"""Oracle harness: run the OFFICIAL MadryLab/implementation-matters@5ee6ecb primitives (PyTorch, CPU) on a batch
recorded by the JAX re-implementation, and save their outputs for a differential test.

Run with this directory's isolated venv:  .venv/bin/python oracle_harness.py <inputs.npz> <outputs.npz>
Only functions from the official repo are executed (imported unmodified; agent.Trainer.advantage_and_return is
extracted from agent.py source with ast because agent.py imports gym/mujoco_py, which are not installable here).
np.random.shuffle / np.random.choice are monkeypatched to replay the JAX side's minibatch permutations and Fisher
subset so the two implementations see identical minibatches.
"""
import ast
import os
import sys

import numpy as np
import torch as ch

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
SRC = os.path.join(OFFICIAL_ROOT, "implementation-matters", "src")
sys.path.insert(0, SRC)
from policy_gradients import steps, torch_utils  # noqa: E402
from policy_gradients.models import CtsPolicy, ValueDenseNet  # noqa: E402

ch.set_default_dtype(ch.float32)


def load_adv_fn():
    src = open(os.path.join(SRC, "policy_gradients", "agent.py")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "advantage_and_return":
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {"ch": ch, "shape_equal_cmp": torch_utils.shape_equal_cmp,
                  "get_path_indices": torch_utils.get_path_indices, "discount_path": torch_utils.discount_path}
            exec(compile(mod, "agent.py", "exec"), ns)
            return ns["advantage_and_return"]
    raise RuntimeError("advantage_and_return not found")


def set_mlp(layers, final, d, prefix):
    with ch.no_grad():
        for i, l in enumerate(layers):
            l.weight.copy_(ch.tensor(d[f"{prefix}W{i}"].T))
            l.bias.copy_(ch.tensor(d[f"{prefix}b{i}"]))
        final.weight.copy_(ch.tensor(d[f"{prefix}W2"].T))
        final.bias.copy_(ch.tensor(d[f"{prefix}b2"]))


def export_mlp(layers, final, prefix, grad=False):
    out = {}
    for i, l in enumerate(list(layers) + [final]):
        w = l.weight.grad if grad else l.weight
        b = l.bias.grad if grad else l.bias
        out[f"{prefix}W{i}"] = w.detach().numpy().T.copy()
        out[f"{prefix}b{i}"] = b.detach().numpy().copy()
    return out


def make_policy(d):
    obs_dim, act_dim = d["obs"].shape[1], d["act"].shape[1]
    net = CtsPolicy(obs_dim, act_dim, "xavier")
    set_mlp(net.affine_layers, net.final_mean, d, "pi_")
    with ch.no_grad():
        net.log_stdev.copy_(ch.tensor(d["pi_log_std"]))
    return net


def make_value(d):
    net = ValueDenseNet(d["obs"].shape[1], "xavier")
    set_mlp(net.affine_layers, net.final, d, "v_")
    return net


class Replay:
    """Replays recorded permutations / subsets through np.random.shuffle / np.random.choice."""

    def __init__(self, perms=None, subset=None):
        self.perms, self.i, self.subset = perms, 0, subset

    def shuffle(self, x):
        x[:] = self.perms[self.i]
        self.i += 1

    def choice(self, *a, **k):
        return np.asarray(self.subset)


def main(inp, outp):
    d = dict(np.load(inp))
    out = {}
    T = d["rew"].shape[0]
    # ---- GAE + returns (agent.advantage_and_return)
    adv_fn = load_adv_fn()
    self_ = type("S", (), {"GAMMA": float(d["gamma"]), "LAMBDA": float(d["lam"])})()
    adv, ret = adv_fn(self_, ch.tensor(d["rew"])[None], ch.tensor(d["values"])[None], ch.tensor(d["not_done"])[None])
    out["adv"], out["ret"] = adv[0].numpy(), ret[0].numpy()
    # ---- value MRE / MSE (steps.value_loss_gae / value_loss_returns with re=True)
    for clip in (0, 1):
        P = torch_utils.Parameters({"value_clipping": bool(clip), "clip_val_eps": 0.2})
        args = [ch.tensor(d["v_new"]), ch.tensor(d["ret_ours"]), ch.tensor(d["adv_ours"]), ch.tensor(d["not_done"]), P,
                ch.tensor(d["values"])]
        mse, mre, _ = steps.value_loss_gae(*args, re=True)
        out[f"gae_mse_clip{clip}"], out[f"gae_mre_code_clip{clip}"] = mse.item(), mre.item()
    mse, mre, _ = steps.value_loss_returns(*args, re=True)
    out["returns_mse"], out["returns_mre"] = mse.item(), mre.item()
    # ---- policy: log-lik, KL, normalised surrogate and its gradient
    net = make_policy(d)
    obs, act = ch.tensor(d["obs"]), ch.tensor(d["act"])
    pds = net(obs)
    lp = net.get_loglikelihood(pds, act)
    out["logp"] = lp.detach().numpy()
    net2 = make_policy({**d, **{k.replace("pi2_", "pi_"): v for k, v in d.items() if k.startswith("pi2_")}})
    out["kl"] = net.calc_kl(pds, net2(obs)).detach().numpy()
    adv_t = ch.tensor(d["adv_ours"])
    out["adv_normalized"] = steps.adv_normalize(adv_t).numpy()
    net.zero_grad()
    lp_old = ch.tensor(d["logp_old"])
    s = steps.surrogate_reward(adv_t, new=net.get_loglikelihood(net(obs), act), old=lp_old).mean()
    s.backward()
    out.update(export_mlp(net.affine_layers, net.final_mean, "gpi_", grad=True))
    out["gpi_log_std"] = net.log_stdev.grad.numpy().copy()
    out["surrogate"] = s.item()
    # ---- CG on an SPD system
    H = ch.tensor(d["H"])
    out["cg_x"] = torch_utils.cg_solve(lambda v: H @ v, ch.tensor(d["cg_b"]), 10).numpy()
    # ---- one PPO step (steps.ppo_step) with replayed minibatch permutations
    net = make_policy(d)
    P = torch_utils.Parameters({"share_weights": False, "ppo_epochs": int(d["ppo_epochs"]),
                                "num_minibatches": int(d["n_mb"]), "clip_eps": 0.2, "entropy_coeff": 0.0,
                                "clip_grad_norm": -1})
    P.POLICY_ADAM = ch.optim.Adam(net.parameters(), lr=float(d["ppo_lr"]), eps=1e-5)
    rep = Replay(perms=d["ppo_perms"])
    orig_shuffle, orig_choice = np.random.shuffle, np.random.choice
    np.random.shuffle = rep.shuffle
    try:
        steps.ppo_step(obs, act, lp_old, None, None, None, adv_t, net, P, None, 0)
    finally:
        np.random.shuffle = orig_shuffle
    out.update(export_mlp(net.affine_layers, net.final_mean, "ppo_pi_"))
    out["ppo_pi_log_std"] = net.log_stdev.detach().numpy().copy()
    # ---- one value step (steps.value_step) with replayed permutations
    vnet = make_value(d)
    P = torch_utils.Parameters({"share_weights": False, "num_minibatches": int(d["n_mb"]), "val_epochs": int(d["v_epochs"]),
                                "value_calc": "gae", "value_clipping": bool(d["v_clip"]), "clip_val_eps": 0.2})
    vopt = ch.optim.Adam(vnet.parameters(), lr=float(d["v_lr"]), eps=1e-5)
    rep = Replay(perms=d["v_perms"])
    np.random.shuffle = rep.shuffle
    try:
        steps.value_step(obs, ch.tensor(d["ret_ours"]), adv_t, ch.tensor(d["not_done"]), vnet, vopt, P, None)
    finally:
        np.random.shuffle = orig_shuffle
    out.update(export_mlp(vnet.affine_layers, vnet.final, "vstep_v_"))
    # ---- one TRPO step (steps.trpo_step) with the replayed Fisher subset
    net = make_policy(d)
    P = torch_utils.Parameters({"fisher_frac_samples": 0.1, "damping": float(d["damping"]), "cg_steps": 10,
                                "max_kl": float(d["max_kl"]), "max_backtrack": 10, "should_log_kl": False})
    np.random.choice = Replay(subset=d["fisher_subset"]).choice
    try:
        steps.trpo_step(obs, act, lp_old, None, None, None, adv_t, net, P, None, 0)
    finally:
        np.random.choice = orig_choice
    out.update(export_mlp(net.affine_layers, net.final_mean, "trpo_pi_"))
    out["trpo_pi_log_std"] = net.log_stdev.detach().numpy().copy()
    # ---- normalisers: ZFilter (clip 10) and RewardFilter (gamma, clip 10) on a recorded stream with resets
    zf = torch_utils.ZFilter(torch_utils.Identity(), shape=[d["stream_obs"].shape[1]], clip=10.0)
    rf = torch_utils.RewardFilter(torch_utils.Identity(), shape=(), gamma=float(d["gamma"]), clip=10.0)
    zo, ro = [], []
    for x, r, rs in zip(d["stream_obs"], d["stream_rew"], d["stream_reset"]):
        if rs:
            rf.reset()
        zo.append(zf(x))
        ro.append(rf(r))
    out["stream_obs_norm"] = np.asarray(zo, np.float32)
    out["stream_rew_norm"] = np.asarray(ro, np.float32)
    np.savez(outp, **{k: np.asarray(v) for k, v in out.items()})
    print("oracle ok", len(out), "outputs; torch", ch.__version__)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
