"""Run DI-engine's dqfd_nstep_td_error (labelled second-best oracle, commit d0b21d06) on random batches and dump
inputs + outputs to JSON for tests/test_official_parity.py.  Runs under a torch venv (NOT the JAX venv):

    $OFFICIAL_ROOT/.venv/bin/python -I scripts/official/oracle_di_engine.py tests/fixtures

writes tests/fixtures/{di_engine_batches.json (loss/grad on random batches, 6 hyper-parameter cases),
di_engine_net.json (torch DuelingQ forward on numpy-seeded weights), di_engine_train.json (25 Adam steps, same batches)};
tests/test_official_parity.py replays all three through the JAX implementation.

ding's package __init__ pulls in many dependencies we do not need, so td.py is exec'd with three tiny stubs for its
imports (hpc_wrapper -> identity decorator, to_tensor unused, value_rescale loaded from the snapshot unchanged).
"""
import importlib.util, json, os, sys, types
import numpy as np
import torch

PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DI-engine clone, made by scripts/official/setup.sh
SNAP = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official")), "DI-engine")


def load_td():
    ding = types.ModuleType("ding"); ding.__path__ = []
    hpc = types.ModuleType("ding.hpc_rl"); hpc.hpc_wrapper = lambda *a, **k: (lambda f: f)
    tu = types.ModuleType("ding.torch_utils"); tu.to_tensor = lambda x, *a, **k: x
    ru = types.ModuleType("ding.rl_utils"); ru.__path__ = []
    sys.modules.update({"ding": ding, "ding.hpc_rl": hpc, "ding.torch_utils": tu, "ding.rl_utils": ru})
    spec = importlib.util.spec_from_file_location("ding.rl_utils.value_rescale", f"{SNAP}/ding/rl_utils/value_rescale.py")
    vr = importlib.util.module_from_spec(spec); sys.modules["ding.rl_utils.value_rescale"] = vr; spec.loader.exec_module(vr)
    spec = importlib.util.spec_from_file_location("ding.rl_utils.td", f"{SNAP}/ding/rl_utils/td.py")
    td = importlib.util.module_from_spec(spec); sys.modules["ding.rl_utils.td"] = td; spec.loader.exec_module(td)
    return td


HP_CASES = [  # (gamma, T, lam_1, lam_n, lam_e, margin) -- case 0-2 are the paper's values, 3-5 stress every weight
    (0.99, 10, 1.0, 1.0, 1.0, 0.8), (0.99, 10, 1.0, 1.0, 1.0, 0.8), (0.99, 10, 1.0, 1.0, 1.0, 0.8),
    (0.97, 5, 0.3, 0.5, 2.0, 0.5), (0.9, 3, 2.0, 0.0, 1.0, 1.5), (0.99, 10, 1.0, 1.0, 0.0, 0.8)]


def make_batch(rng, B, A, T):
    f = lambda *s: rng.normal(size=s).astype(np.float32)
    q, q1_on, qn_on, q1_tg, qn_tg = f(B, A), f(B, A), f(B, A), f(B, A), f(B, A)
    act = rng.integers(0, A, B)
    rew = f(T, B)
    term_pos = rng.integers(0, 2 * T, B)  # >= T means no termination inside the window
    done = (term_pos < T).astype(np.float32)
    for j in range(B):
        if term_pos[j] < T:
            rew[term_pos[j] + 1:, j] = 0.0
    done1 = (term_pos == 0).astype(np.float32)  # 1-step: terminated at the very first step
    is_demo = rng.integers(0, 2, B).astype(np.float32)
    w = rng.uniform(0.2, 1.0, B).astype(np.float32)
    return dict(q=q, q1_on=q1_on, qn_on=qn_on, q1_tg=q1_tg, qn_tg=qn_tg, act=act, rew=rew, done=done, done1=done1, is_demo=is_demo, w=w)


def di_loss(td, b, gamma, T, lam1, lamn, lame, margin, q=None):
    t = torch.from_numpy
    q = q if q is not None else t(b["q"]).requires_grad_(True)
    data = td.dqfd_nstep_td_data(q, t(b["qn_tg"]), t(b["act"]), t(b["qn_on"]).argmax(1), t(b["rew"]), t(b["done"]), t(b["done1"]), t(b["w"]),
                                 t(b["q1_tg"]), t(b["q1_on"]).argmax(1), t(b["is_demo"]))
    return data, td.dqfd_nstep_td_error(data, gamma, lambda_n_step_td=lamn, lambda_supervised_loss=lame, margin_function=margin,
                                        lambda_one_step_td=lam1, nstep=T)


def canonical_weights(seed, A=6, C=4, ch=16, hid=64):
    """Weights in the FLAX layout (conv kernel (3,3,C,ch); dense kernel (in,out); flatten order HWC), from numpy only."""
    rng = np.random.default_rng(seed)
    n = lambda *s: (rng.normal(size=s) * (1.0 / np.sqrt(np.prod(s[:-1])))).astype(np.float32)
    b = lambda k: (0.1 * rng.normal(size=(k,))).astype(np.float32)
    return dict(conv_k=n(3, 3, C, ch), conv_b=b(ch), vh_k=n(ch * 100, hid), vh_b=b(hid), vo_k=n(hid, 1), vo_b=b(1),
                ah_k=n(ch * 100, hid), ah_b=b(hid), ao_k=n(hid, A), ao_b=b(A))


def load_torch_net(Net, w, A=6, C=4, ch=16, hid=64):
    net = Net(A, C, ch, hid)
    # HWC-flatten kernel (h*10*ch + w*ch + c, out)  ->  CHW-flatten torch weight (out, c*100 + h*10 + w)
    perm = lambda k: torch.from_numpy(k.reshape(10, 10, ch, -1).transpose(2, 0, 1, 3).reshape(ch * 100, -1).T.copy())
    with torch.no_grad():
        net.conv.weight.copy_(torch.from_numpy(w["conv_k"].transpose(3, 2, 0, 1).copy())); net.conv.bias.copy_(torch.from_numpy(w["conv_b"]))
        net.v_hidden.weight.copy_(perm(w["vh_k"])); net.v_hidden.bias.copy_(torch.from_numpy(w["vh_b"]))
        net.a_hidden.weight.copy_(perm(w["ah_k"])); net.a_hidden.bias.copy_(torch.from_numpy(w["ah_b"]))
        net.v_out.weight.copy_(torch.from_numpy(w["vo_k"].T.copy())); net.v_out.bias.copy_(torch.from_numpy(w["vo_b"]))
        net.a_out.weight.copy_(torch.from_numpy(w["ao_k"].T.copy())); net.a_out.bias.copy_(torch.from_numpy(w["ao_b"]))
    return net


def main(outdir):
    td = load_td()
    cases = []
    rng = np.random.default_rng(0)
    B, A = 64, 6
    for c, (gamma, T, l1, ln, le, mg) in enumerate(HP_CASES):
        b = make_batch(rng, B, A, T)
        data, (loss, per_sample, stats) = di_loss(td, b, gamma, T, l1, ln, le, mg)
        loss.backward()
        cases.append(dict(gamma=gamma, T=T, lam1=l1, lamn=ln, lame=le, margin=mg, **{k: v.tolist() for k, v in b.items()},
                          loss=float(loss), per_sample=per_sample.tolist(), stats=[float(x) for x in stats], grad_q=data.q.grad.tolist()))
    meta = dict(torch=torch.__version__, commit="d0b21d065b2d80317cc16b28a95301be8871d950")
    json.dump(dict(**meta, cases=cases), open(f"{outdir}/di_engine_batches.json", "w"))

    # ---- network forward parity (torch DuelingQ of scripts/ref_torch_dqfd.py on numpy-seeded weights)
    spec = importlib.util.spec_from_file_location("ref_torch_dqfd", __file__.replace("oracle_di_engine.py", "ref_torch_dqfd.py"))
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    w = canonical_weights(11)
    net = load_torch_net(ref.Net, w)
    x = (np.random.default_rng(12).random((16, 10, 10, 4)) < 0.2).astype(np.float32)          # NHWC
    with torch.no_grad():
        qout = net(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).numpy()
    json.dump(dict(**meta, q=qout.tolist()), open(f"{outdir}/di_engine_net.json", "w"))

    # ---- 25-step training trajectory: torch Adam + clip + DI-engine loss + explicit L2  vs  optax chain + src/algorithm.py
    G, T, Bt, steps, sync_at = 0.99, 10, 32, 25, 10
    wo = canonical_weights(21); net = load_torch_net(ref.Net, wo); tnet = load_torch_net(ref.Net, canonical_weights(22))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, eps=1.5e-4)
    r2 = np.random.default_rng(23)
    losses = []
    for it in range(steps):
        s = (r2.random((Bt, 10, 10, 4)) < 0.2).astype(np.float32); s1 = (r2.random((Bt, 10, 10, 4)) < 0.2).astype(np.float32)
        sn = (r2.random((Bt, 10, 10, 4)) < 0.2).astype(np.float32)
        act = r2.integers(0, 6, Bt); T_ = T
        rew = r2.normal(size=(T_, Bt)).astype(np.float32)
        term_pos = r2.integers(0, 2 * T_, Bt); done = (term_pos < T_).astype(np.float32)
        for j in range(Bt):
            if term_pos[j] < T_:
                rew[term_pos[j] + 1:, j] = 0.0
        done1 = (term_pos == 0).astype(np.float32); is_demo = r2.integers(0, 2, Bt).astype(np.float32); wt = r2.uniform(0.2, 1.0, Bt).astype(np.float32)
        tch = lambda a_: torch.from_numpy(a_.transpose(0, 3, 1, 2).copy())
        with torch.no_grad():
            qt = tnet(tch(np.concatenate([s1, sn]))); qo = net(tch(np.concatenate([s1, sn])))
        q = net(tch(s))
        t = torch.from_numpy
        data = td.dqfd_nstep_td_data(q, qt[Bt:], t(act), qo[Bt:].argmax(1), t(rew), t(done), t(done1), t(wt), qt[:Bt], qo[:Bt].argmax(1), t(is_demo))
        loss, _, _ = td.dqfd_nstep_td_error(data, G, lambda_n_step_td=1.0, lambda_supervised_loss=1.0, margin_function=0.8, lambda_one_step_td=1.0, nstep=T_)
        l2 = 0.5 * sum((p ** 2).sum() for p in net.parameters())
        total = loss + 1e-5 * l2
        opt.zero_grad(); total.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        losses.append(float(total))
        if it + 1 == sync_at:
            tnet.load_state_dict(net.state_dict())
    with torch.no_grad():
        qfin = net(torch.from_numpy(x.transpose(0, 3, 1, 2).copy())).numpy()
    json.dump(dict(**meta, losses=losses, q_final=qfin.tolist(), sync_at=sync_at), open(f"{outdir}/di_engine_train.json", "w"))
    print("wrote fixtures to", outdir, "torch", torch.__version__)


if __name__ == "__main__":
    main(sys.argv[1])
