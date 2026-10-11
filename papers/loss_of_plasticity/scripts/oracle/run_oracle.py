"""Runs the OFFICIAL code (github.com/shibhansh/loss-of-plasticity @ a6b7958, unmodified, imported from
$OFFICIAL_ROOT/repo, default external/official/repo) on cases for the differential tests.

    $OFFICIAL_ROOT/.venv/bin/python scripts/oracle/run_oracle.py <case> <in.npz|-> <out.npz>

Cases
  gnt_traj   : record a GnT / ContinualBackprop trajectory (pre/post state of every step) for teacher-forced diffs
  ltu        : FixLTUNet outputs on given weights + inputs
  diag       : compute_matrix_rank_summaries + dead-unit count on given activation matrices
  bp_traj    : Backprop (SGD / Adam, optional weight decay) from given init weights on a given stream (per-step loss)
  snp_step   : one Backprop step with weight decay + S&P noise on given weights (noise distribution check)
  cbp_run    : ContinualBackprop from torch's own init on a given SCR stream (reference curve; per-step error)
Runs on CPU, float32, torch.set_num_threads(1).

LOP_ORACLE_REPO (environment) selects another unmodified tree of the official code, e.g.
$OFFICIAL_ROOT/repo_pre_f00ca40 (`git archive 63c35f3`: identical to a6b7958 except
lop/utils/AdamGnT.py, which is the paper-era dcc3619 version).
"""
import os
import sys

sys.path.insert(0, os.environ.get("LOP_ORACLE_REPO", os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official")), "repo")))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lop.algos.bp import Backprop  # noqa: E402
from lop.algos.cbp import ContinualBackprop  # noqa: E402
from lop.nets.deep_ffnn import DeepFFNN  # noqa: E402
from lop.nets.ffnn import FFNN  # noqa: E402
from lop.nets.fix_ltu_net import FixLTUNet  # noqa: E402
from lop.nets.linear import MyLinear  # noqa: E402
from lop.utils.miscellaneous import compute_matrix_rank_summaries  # noqa: E402

torch.set_num_threads(1)


def snapshot(net_layers, gnt, opt):
    d = {}
    lin = [m for m in net_layers if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        d[f"W{i}"] = m.weight.detach().numpy().copy()
        d[f"b{i}"] = m.bias.detach().numpy().copy()
        if opt is not None and m.weight in opt.state and len(opt.state[m.weight]) > 0:
            for k in ("exp_avg", "exp_avg_sq", "step"):
                d[f"{k}_W{i}"] = opt.state[m.weight][k].detach().numpy().copy().astype(np.float32)
                d[f"{k}_b{i}"] = opt.state[m.bias][k].detach().numpy().copy().astype(np.float32)
    if gnt is not None:
        for l in range(gnt.num_hidden_layers):
            d[f"util{l}"] = gnt.util[l].numpy().copy()
            d[f"bcu{l}"] = gnt.bias_corrected_util[l].numpy().copy()
            d[f"f{l}"] = gnt.mean_feature_act[l].numpy().copy()
            d[f"age{l}"] = gnt.ages[l].numpy().copy()
            d[f"acc{l}"] = np.float64(gnt.accumulated_num_features_to_replace[l])
    return d


def gnt_traj(cfg):
    torch.manual_seed(int(cfg["seed"]))
    kind = str(cfg["kind"])
    if kind == "ffnn":
        net = FFNN(input_size=int(cfg["n_in"]), num_features=int(cfg["width"]), hidden_activation=str(cfg["act"]))
        loss = "mse"
    else:
        net = DeepFFNN(input_size=int(cfg["n_in"]), num_features=int(cfg["width"]), num_outputs=int(cfg["n_out"]),
                       num_hidden_layers=int(cfg["n_hidden"]), act_type=str(cfg["act"]))
        loss = "nll"
    learner = ContinualBackprop(net=net, step_size=float(cfg["lr"]), loss=loss, opt=str(cfg["opt"]),
                                replacement_rate=float(cfg["rho"]), decay_rate=float(cfg["eta"]),
                                maturity_threshold=int(cfg["mt"]), util_type=str(cfg["util_type"]),
                                accumulate=True)
    gnt = learner.gnt
    recorded = []
    orig = gnt.test_features

    def wrapped(features):
        r, n = orig(features)
        recorded.append([x.numpy().copy() for x in r])
        return r, n
    gnt.test_features = wrapped
    n_steps = int(cfg["n_steps"])
    g = torch.Generator().manual_seed(int(cfg["seed"]) + 1)
    out = {}
    for t in range(n_steps):
        x = torch.randint(0, 2, (1, int(cfg["n_in"])), generator=g).float()
        if loss == "mse":
            y = torch.randn((1, 1), generator=g)
        else:
            y = torch.randint(0, int(cfg["n_out"]), (1,), generator=g)
        pre = snapshot(net.layers, gnt, learner.opt)
        learner.learn(x=x, target=y)
        post = snapshot(net.layers, gnt, learner.opt)
        out[f"t{t}_x"] = x.numpy()
        out[f"t{t}_y"] = y.numpy()
        for k, v in pre.items():
            out[f"t{t}_pre_{k}"] = v
        for k, v in post.items():
            out[f"t{t}_post_{k}"] = v
        for l, r in enumerate(recorded[-1]):
            out[f"t{t}_rep{l}"] = r.astype(np.int64)
    out["n_steps"] = np.int64(n_steps)
    return out


def ltu(inp):
    out = {}
    for s in range(int(inp["n_seeds"])):
        net = FixLTUNet(num_inputs=20, num_features=100, beta=0.7)
        with torch.no_grad():
            net.layers[0].weight.data = torch.tensor(inp[f"W_{s}"])
            net.layers[0].bias.data = torch.tensor(inp[f"b_{s}"])
            net.layers[1].weight.data = torch.tensor(inp[f"w2_{s}"])[None, :]
            net.layers[1].bias.data = torch.tensor(inp[f"b2_{s}"]).reshape(1)
            S = (net.num_inputs - torch.sum(net.layers[0].weight.data, dim=1) + net.layers[0].bias.data) / 2
            net.tau = net.beta * (net.num_inputs + 1) - S
            net.hidden_activation.tau = net.tau
            X = torch.tensor(inp[f"X_{s}"].astype(np.float32))
            Y, _ = net.predict(X)
        out[f"Y_{s}"] = Y.numpy()[:, 0]
        out[f"tau_{s}"] = net.tau.numpy()
    return out


def diag(inp):
    out = {}
    for i in range(int(inp["n"])):
        m = torch.tensor(inp[f"m_{i}"])
        rank, er, ar, aar = compute_matrix_rank_summaries(m=m, use_scipy=True)
        out[f"erank_{i}"] = np.float64(er)
        out[f"dead_{i}"] = np.int64((m.abs().sum(dim=0) == 0).sum())
    return out


def _load_lin(layers, inp):
    lin = [m for m in layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for i, m in enumerate(lin):
            m.weight.data = torch.tensor(inp[f"W{i}"]).clone()
            m.bias.data = torch.tensor(inp[f"b{i}"]).clone()


def bp_traj(inp):
    kind = str(inp["kind"])
    if kind == "ffnn":
        net = FFNN(input_size=int(inp["n_in"]), num_features=int(inp["width"]), hidden_activation=str(inp["act"]))
        loss = "mse"
    elif kind == "linear":
        net = MyLinear(input_size=int(inp["n_in"]))
        loss = "mse"
    else:
        net = DeepFFNN(input_size=784, num_features=int(inp["width"]), num_outputs=10, num_hidden_layers=3,
                       act_type="relu")
        loss = "nll"
    _load_lin(net.layers if kind != "linear" else [net.fc1], inp)
    if str(inp["opt"]) == "adamgnt":   # the official CBP learner with rho = 0 (AdamGnT, no replacement)
        learner = ContinualBackprop(net=net, step_size=float(inp["lr"]), loss=loss, opt="adam", replacement_rate=0.0,
                                    decay_rate=0.99, maturity_threshold=100, util_type="adaptable_contribution")
    else:
        learner = Backprop(net=net, step_size=float(inp["lr"]), loss=loss, opt=str(inp["opt"]),
                           weight_decay=float(inp["wd"]))
    X = inp["X"]
    Y = inp["Y"]
    n = X.shape[0]
    errs = np.zeros(n, np.float32)
    for t in range(n):
        x = torch.tensor(X[t:t + 1].astype(np.float32))
        if loss == "mse":
            errs[t] = float(learner.learn(x=x, target=torch.tensor(Y[t:t + 1].astype(np.float32))[:, None]))
        else:
            if kind == "mnist":
                x = x / 255.0
            l, _ = learner.learn(x=x, target=torch.tensor(Y[t:t + 1].astype(np.int64)))
            errs[t] = float(l)
    final = {}
    lin = [m for m in (net.layers if kind != "linear" else [net.fc1]) if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        final[f"W{i}"] = m.weight.detach().numpy()
    return {"errs": errs, **final}


def snp_step(inp):
    """One official Backprop step (SGD, weight decay, optional Shrink-and-Perturb noise) on a DeepFFNN from given
    weights and one example; returns all post-step weights and biases (the S&P noise = post(sigma) - post(0))."""
    torch.manual_seed(int(inp["torch_seed"]))
    net = DeepFFNN(input_size=784, num_features=int(inp["width"]), num_outputs=10, num_hidden_layers=3,
                   act_type="relu")
    _load_lin(net.layers, inp)
    sigma = float(inp["sigma"])
    learner = Backprop(net=net, step_size=float(inp["lr"]), loss="nll", opt="sgd", weight_decay=float(inp["wd"]),
                       to_perturb=sigma > 0, perturb_scale=sigma)
    learner.learn(x=torch.tensor(inp["x"].astype(np.float32)), target=torch.tensor(inp["y"].astype(np.int64)))
    lin = [m for m in net.layers if isinstance(m, torch.nn.Linear)]
    out = {}
    for i, m in enumerate(lin):
        out[f"W{i}"] = m.weight.detach().numpy()
        out[f"b{i}"] = m.bias.detach().numpy()
    return out


def cbp_run(inp):
    torch.manual_seed(int(inp["torch_seed"]))
    net = FFNN(input_size=20, num_features=5, hidden_activation=str(inp["act"]))
    if float(inp["rho"]) > 0:
        learner = ContinualBackprop(net=net, step_size=float(inp["lr"]), opt=str(inp["opt"]),
                                    replacement_rate=float(inp["rho"]), decay_rate=0.99, device="cpu",
                                    maturity_threshold=int(inp["mt"]), util_type=str(inp["util_type"]),
                                    accumulate=False)
    else:
        learner = Backprop(net=net, step_size=float(inp["lr"]), opt=str(inp["opt"]), weight_decay=float(inp["wd"]),
                           to_perturb=float(inp["sigma"]) > 0, perturb_scale=float(inp["sigma"]))
    X = torch.tensor(inp["X"].astype(np.float32))
    Y = torch.tensor(inp["Y"].astype(np.float32))[:, None]
    n = X.shape[0]
    errs = torch.zeros(n)
    for i in range(n):
        errs[i] = learner.learn(x=X[i:i + 1], target=Y[i:i + 1])
    T = 10_000
    return {"mse_per_10k": errs.numpy().reshape(-1, T).mean(1)}


def mnist_run(inp):
    """Official DeepFFNN + Backprop / ContinualBackprop (accumulate=True, as online_expr.py) on OUR task sequence
    (pixel permutations and example orders); torch's own init.  Diagnostics as online_expr.py (before each task,
    first 2000 examples of the task)."""
    torch.manual_seed(int(inp["torch_seed"]))
    net = DeepFFNN(input_size=784, num_features=int(inp["width"]), num_outputs=10, num_hidden_layers=3)
    rho = float(inp["rho"])
    if rho > 0:
        learner = ContinualBackprop(net=net, step_size=float(inp["lr"]), opt="sgd", loss="nll", replacement_rate=rho,
                                    maturity_threshold=100, decay_rate=0.99, util_type="adaptable_contribution",
                                    accumulate=True, device="cpu")
    else:
        learner = Backprop(net=net, step_size=float(inp["lr"]), opt="sgd", loss="nll", weight_decay=float(inp["wd"]),
                           to_perturb=float(inp["sigma"]) > 0, perturb_scale=float(inp["sigma"]))
    X = torch.tensor(inp["X"].astype(np.float32) / 255.0)
    Y = torch.tensor(inp["Y"].astype(np.int64))
    perms, orders = inp["perms"], inp["orders"]
    n_tasks, n_ex = perms.shape[0], int(inp["n_ex"])
    acc = np.zeros(n_tasks)
    dead = np.zeros((n_tasks, 3))
    erank = np.zeros((n_tasks, 3))
    wmag = np.zeros(n_tasks)
    for t in range(n_tasks):
        xt = X[orders[t]][:, perms[t]]
        yt = Y[orders[t]]
        with torch.no_grad():
            m = net.predict(xt[:2000])[1]
            for l in range(3):
                _, er, _, _ = compute_matrix_rank_summaries(m=m[l], use_scipy=True)
                erank[t, l] = float(er)
                dead[t, l] = float((m[l].abs().sum(dim=0) == 0).sum())
        correct = 0.0
        for i in range(n_ex):
            loss, out = learner.learn(x=xt[i:i + 1], target=yt[i:i + 1])
            correct += float((out.argmax(dim=1) == yt[i:i + 1]).float().mean())
        acc[t] = correct / n_ex
        lin = [mm for mm in net.layers if isinstance(mm, torch.nn.Linear)]
        wmag[t] = float(sum(mm.weight.abs().sum() for mm in lin) / sum(mm.weight.numel() for mm in lin))
        print("[INFO] oracle mnist task", t, acc[t], flush=True)
    return {"acc": acc, "dead": dead, "erank": erank, "wmag_end": wmag}


if __name__ == "__main__":
    case, inp_path, out_path = sys.argv[1:4]
    inp = dict(np.load(inp_path, allow_pickle=False)) if inp_path != "-" else {}
    if case == "gnt_traj":
        res = gnt_traj(inp)
    else:
        res = {"ltu": ltu, "diag": diag, "bp_traj": bp_traj, "cbp_run": cbp_run, "mnist_run": mnist_run,
               "snp_step": snp_step}[case](inp)
    np.savez(out_path, **res)
    print("[INFO] oracle", case, "->", out_path, flush=True)
