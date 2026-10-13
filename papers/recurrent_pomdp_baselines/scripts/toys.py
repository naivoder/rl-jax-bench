"""Toy stage: exactly-computable toy (the README, cue-memory POMDP, H = 5).

Runs (all vmapped over 4 seeds, one process):
  A  recurrent SAC-D (separate), alpha 0.1 fixed, gamma 0.99, L = H        -> learned Q(tau_t, a) vs Q^soft closed form,
                                                                              pi(c | tau_4) vs sigma(1/alpha), V(tau_0)
  B  Markovian SAC-D (observation only)                                     -> Q(o_4, a) = 1/2 both, V_M = 1/2 + a log 2
  C  recurrent SAC-D with truncated context L = 2 (< H)                     -> Q at the decision step = 1/2 (§3.5)
  D  recurrent TD3, continuous reward (2c-1) a                              -> return J* = 1
  E  Markovian TD3, continuous                                              -> return ~ 0 (memoryless optimum)
  F  recurrent TD3 with the SHARED (official) encoder, continuous           -> reported only (no closed-form claim)
Writes runs/toys/toys.json; the README is written from it.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import info, load_config, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.train import train, tree_index  # noqa: E402

OUT = os.path.join(common.REPO, "runs", "toys")
ALPHA, GAMMA, H = 0.1, 0.99, 5


def closed_form():
    V = [0.0] * H
    V[H - 1] = ALPHA * math.log(1 + math.exp(1 / ALPHA))
    for t in range(H - 2, -1, -1):
        V[t] = GAMMA * V[t + 1] + ALPHA * math.log(2)
    Q = [GAMMA * V[t + 1] for t in range(H - 1)]
    return {"V": V, "Q_pre": Q, "pi_c": 1 / (1 + math.exp(-1 / ALPHA)), "V_markov_dec": 0.5 + ALPHA * math.log(2)}


def q_table(R, ts, i, L_window=None):
    """learned Q(tau_t, .) on deterministic eval histories of both cues, for replication i."""
    ag, n = R.agent, R.nets
    learner = tree_index(ts["learner"], i)
    out = {}
    for c in (0, 1):
        # build the history: cue c at t=0, f_t at t=4; the policy's own (deterministic) actions as prev actions
        obs = np.zeros((H + 1, 1, 3), np.float32)
        obs[0, 0, c] = 1.0
        obs[H - 1, 0, 2] = 1.0
        carry = ag.init_carry(1) if ag.arch != "markov" else None
        acts = np.zeros((H + 1, 1, R.env.act_dim), np.float32)
        rews = np.zeros((H + 1, 1, 1), np.float32)
        prev_a, r = jnp.zeros((1, R.env.act_dim)), jnp.zeros((1, 1))
        for t in range(H):
            a, carry = ag.act(learner, carry, prev_a, r, jnp.asarray(obs[t]), jax.random.PRNGKey(0), True)
            acts[t + 1] = np.asarray(a)
            rr = float(np.argmax(np.asarray(a)[0]) == c) if t == H - 1 else 0.0
            rews[t + 1, 0, 0] = rr
            prev_a, r = a, jnp.full((1, 1), rr)
        P = learner["params"]
        if ag.arch == "markov":
            q1 = n.q_out(P["critic"]["q1"], jnp.asarray(obs[:H, 0]))
            q2 = n.q_out(P["critic"]["q2"], jnp.asarray(obs[:H, 0]))
            logits = n.pi_out(P["actor"]["pi"], jnp.asarray(obs[:H, 0]))
        else:
            o, a_, r_ = jnp.asarray(obs), jnp.asarray(acts), jnp.asarray(rews)
            if L_window:   # critic as trained: zero-start window ending at the decision step
                s = H - L_window
                o, a_, r_ = o[s:], a_.at[s].set(0.0)[s:], r_.at[s].set(0.0)[s:]
            hq, _ = n.encode(P["critic"]["enc"], a_, r_, o)
            q1, q2 = ag._critic_qs(P["critic"], hq, o, a_)
            ha, _ = n.encode(P["actor"]["enc"], a_, r_, o)
            logits = n.pi_out(P["actor"]["pi"], jnp.concatenate([ha, n.short_o_out(P["actor"]["short"], o)], -1))
            q1, q2, logits = q1[:, 0], q2[:, 0], logits[:, 0]
            if L_window:
                q1, q2, logits = q1[L_window - 1:L_window], q2[L_window - 1:L_window], logits[L_window - 1:L_window]
            else:
                q1, q2, logits = q1[:H], q2[:H], logits[:H]
        q = np.minimum(np.asarray(q1), np.asarray(q2))
        p = np.asarray(jax.nn.softmax(logits, -1))
        out[c] = {"Q": q.tolist(), "pi": p.tolist()}
    return out


def run(name, cfg_over, seeds=(0, 1, 2, 3)):
    cfg = load_config("cue.yaml", cfg_over)
    common.set_precision("highest")
    ts, host, R = train(cfg, list(seeds), [cfg["arch"] if cfg["arch"] != "shared" else "official"],
                        os.path.join(OUT, name), resume=False)
    rets = np.asarray(host["evals"][-1]["return"])
    return R, ts, rets


def main():
    cf = closed_form()
    res = {"closed_form": cf}
    ok = True
    # A: recurrent SAC-D, full context
    R, ts, rets = run("A_rec_sacd", ["arch=separate", "algo=sacd"])
    errs, pis, v0 = [], [], []
    for i in range(4):
        qt = q_table(R, ts, i)
        for c in (0, 1):
            Q = np.array(qt[c]["Q"])
            want = np.array([[cf["Q_pre"][t]] * 2 for t in range(H - 1)] + [[1.0 - c, float(c)]])
            errs.append(np.abs(Q - want).max())
            pis.append(qt[c]["pi"][H - 1][c])
            p0 = np.array(qt[c]["pi"][0])
            v0.append(float(np.sum(p0 * (Q[0] - ALPHA * np.log(np.clip(p0, 1e-8, None))))))
    a_pass = bool(max(errs) < 0.05 and min(pis) > 0.99 and abs(np.mean(v0) - cf["V"][0]) < 0.05 and rets.min() > 0.99)
    res["A"] = {"max_abs_Q_err": float(max(errs)), "Q_err_per_seed_cue": [float(e) for e in errs],
                "pi_c_dec_min": float(min(pis)), "pi_c_closed": cf["pi_c"], "V0_learned_mean": float(np.mean(v0)),
                "V0_closed": cf["V"][0], "eval_return": rets.tolist(), "pass": bool(a_pass)}
    ok &= a_pass
    # B: Markovian SAC-D
    R, ts, rets = run("B_markov_sacd", ["arch=markov", "algo=sacd"])
    qd, vd = [], []
    for i in range(4):
        qt = q_table(R, ts, i)
        Q4 = np.array(qt[0]["Q"][H - 1])
        p4 = np.array(qt[0]["pi"][H - 1])
        qd += Q4.tolist()
        vd.append(float(np.sum(p4 * (Q4 - ALPHA * np.log(np.clip(p4, 1e-8, None))))))
    b_pass = bool(np.max(np.abs(np.array(qd) - 0.5)) < 0.05 and abs(np.mean(vd) - cf["V_markov_dec"]) < 0.05)
    res["B"] = {"Q_dec": qd, "V_dec_mean": float(np.mean(vd)), "V_dec_closed": cf["V_markov_dec"],
                "eval_return": rets.tolist(), "pass": bool(b_pass)}
    ok &= b_pass
    # C: truncated context L = 2
    R, ts, rets = run("C_rec_sacd_L2", ["arch=separate", "algo=sacd", "sampled_seq_len=2"])
    qd = []
    for i in range(4):
        for c in (0, 1):
            qd += np.array(q_table(R, ts, i, L_window=2)[c]["Q"][0]).tolist()
    c_pass = bool(np.max(np.abs(np.array(qd) - 0.5)) < 0.05 and abs(rets.mean() - 0.5) < 0.2)
    res["C"] = {"Q_dec_window2": qd, "eval_return": rets.tolist(), "pass": bool(c_pass)}
    ok &= c_pass
    # D/E/F: continuous TD3
    for key, over in (("D", ["arch=separate", "algo=td3"]), ("E", ["arch=markov", "algo=td3"]),
                      ("F", ["arch=shared", "algo=td3"])):
        R, ts, rets = run(f"{key}_{over[0][5:]}_td3", over + ["gamma=0.99"])
        if key == "D":
            p = bool(rets.min() > 0.95)
        elif key == "E":
            p = bool(abs(rets.mean()) < 0.25)
        else:
            p = None
        res[key] = {"eval_return": rets.tolist(), "pass": p}
        if p is not None:
            ok &= bool(p)
    res["toys_pass"] = bool(ok)
    write_json(os.path.join(OUT, "toys.json"), res)
    info("toys", passed=bool(ok), **{k: res[k]["pass"] for k in "ABCDEF"})
    print(json.dumps({k: (v if k == "closed_form" else {kk: vv for kk, vv in v.items() if "per_seed" not in kk})
                      for k, v in res.items() if k != "toys_pass"}, indent=1, default=float)[:4000])


def main_c_prime():
    """C' (added after C failed its 0.05 tolerance, see the README toy stage): the same truncation test with inputs 'o'.
    With 'oar' the previous action inside the 2-step window is produced by an actor that ran on the FULL history
    during collection, so it can carry cue information into the window and the truncated critic's fixed point is not
    exactly 1/2.  With inputs 'o' the window (o_3, o_4) carries no cue information by construction, so the fixed
    point is exactly 1/2 (the README premise).  Same tolerance (0.05) as C."""
    path = os.path.join(OUT, "toys.json")
    res = json.load(open(path))
    R, ts, rets = run("Cp_rec_sacd_L2_o", ["arch=separate", "algo=sacd", "sampled_seq_len=2", "inputs=o"])
    qd, same = [], []
    for i in range(4):
        qt = q_table(R, ts, i, L_window=2)
        qd += np.array(qt[0]["Q"][0]).tolist() + np.array(qt[1]["Q"][0]).tolist()
        same.append(bool(np.allclose(qt[0]["Q"][0], qt[1]["Q"][0])))
    # C diagnostics: are the Q values identical across cues (no cue information in the window)?
    p = bool(np.max(np.abs(np.array(qd) - 0.5)) < 0.05 and abs(rets.mean() - 0.5) < 0.2)
    res["C_prime"] = {"Q_dec_window2": qd, "max_abs_dev": float(np.max(np.abs(np.array(qd) - 0.5))),
                      "Q_identical_across_cues": same, "eval_return": rets.tolist(), "pass": p}
    qc = np.array(res["C"]["Q_dec_window2"])
    res["C"]["max_abs_dev"] = float(np.max(np.abs(qc - 0.5)))
    res["C"]["mean_Q_dec"] = float(qc.mean())
    write_json(path, res)
    info("toys C_prime", passed=p, max_abs_dev=res["C_prime"]["max_abs_dev"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "Cprime":
        main_c_prime()
    else:
        main()
