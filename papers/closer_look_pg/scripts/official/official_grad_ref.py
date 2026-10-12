"""Gradient-quality reference (paper Figs 9/10) built ONLY from the official MadryLab/implementation-matters@5ee6ecb
primitives on gymnasium MuJoCo v4 proxies (PyTorch, CPU).

The released repo has no gradient instrument (its logging.py imports sklearn's cosine_similarity but never uses it).
This harness composes the official pieces the instrument needs, unmodified:
  * checkpoints  = the official Trainer (paper Table-1 settings, official_ref_run.make_agent), trained in-process;
  * a sample of n pairs = Trainer.collect_saps(n) -> run_trajectories (env reset, ONE actor, n sequential steps) +
                   Trainer.advantage_and_return with the agent's own value net (official semantics);
  * the estimate = d/dtheta mean(steps.surrogate_reward(A, new=logp_theta, old=logp_theta.detach())) at theta_k
                   (normalised advantages inside surrogate_reward, ratio = 1), and the raw-A variant;
  * statistics   = sklearn.metrics.pairwise.cosine_similarity (the function the official logging.py imports).
Normalisers match our checkpoint semantics : at checkpoint k the filters are warmed by one
2048-step rollout of theta_k ("normaliser after rollout k"), then FROZEN (RunningStat.push -> no-op) for the
instrument, then RESTORED, so the training trajectory is not perturbed by the instrument.

Usage: .venv/bin/python -I official_grad_ref.py <hopper|walker2d> <ppo|trpo> <seed> <out.npz> [its=0,15,45]
        [sizes=100,1000,2000,10000] [m=10] [ref=100000]
"""
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import official_ref_run as R  # noqa: E402  (installs the gym shim and the official src path)

import numpy as np  # noqa: E402
import torch as ch  # noqa: E402
from policy_gradients import steps, torch_utils  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402


def estimate(agent, n, normalize=True):
    saps = agent.collect_saps(n, should_log=False)[0]
    net = agent.policy_model
    lp = net.get_loglikelihood(net(saps.states), saps.actions)
    if normalize:
        s = steps.surrogate_reward(saps.advantages, new=lp, old=lp.detach()).mean()
    else:
        s = (ch.exp(lp - lp.detach()) * saps.advantages).mean()
    g = ch.autograd.grad(s, list(net.parameters()))
    return ch.cat([x.reshape(-1) for x in g]).detach().numpy().astype(np.float64)


def both(agent, n):
    """Normalised and raw estimates from the SAME sample (same as our ablation)."""
    saps = agent.collect_saps(n, should_log=False)[0]
    net = agent.policy_model
    lp = net.get_loglikelihood(net(saps.states), saps.actions)
    out = []
    for s in (steps.surrogate_reward(saps.advantages, new=lp, old=lp.detach()).mean(),
              (ch.exp(lp - lp.detach()) * saps.advantages).mean()):
        g = ch.autograd.grad(s, list(net.parameters()), retain_graph=True)
        out.append(ch.cat([x.reshape(-1) for x in g]).detach().numpy().astype(np.float64))
    return out


def checkpoint_arrays(agent):
    """Policy/value weights (W stored [in, out]) and the frozen normaliser state, for replay in the JAX side."""
    out = {}
    pol, val = agent.policy_model, agent.val_model
    for pre, layers in (("pi_", list(pol.affine_layers) + [pol.final_mean]),
                        ("v_", list(val.affine_layers) + [val.final])):
        for i, l in enumerate(layers):
            out[f"{pre}W{i}"] = l.weight.detach().numpy().T.copy()
            out[f"{pre}b{i}"] = l.bias.detach().numpy().copy()
    out["pi_log_std"] = pol.log_stdev.detach().numpy().copy()
    e = agent.envs[0]
    zs = e.state_filter.rs
    out["obs_n"], out["obs_M"], out["obs_S"] = np.float64(zs._n), zs._M.copy(), zs._S.copy()
    rf = getattr(e.reward_filter, "rs", None)
    if rf is not None:
        out["rew_n"], out["rew_M"], out["rew_S"] = np.float64(rf._n), np.asarray(rf._M).copy(), np.asarray(rf._S).copy()
    return out


def stats(G, r):
    C = cosine_similarity(G)
    iu = np.triu_indices(G.shape[0], 1)
    return dict(pairwise=float(C[iu].mean()), to_true=float(cosine_similarity(G, r[None])[:, 0].mean()))


def main(env, alg, seed, out, its=(0, 15, 45), sizes=(100, 1000, 2000, 10000), m=10, ref=100000):
    agent, params, game = R.make_agent(env, alg, seed, "paper")
    push = torch_utils.RunningStat.push
    res, arrays = {"env": env, "alg": alg, "seed": seed, "sizes": list(sizes), "m": m, "ref_pairs": ref,
                   "gym_proxy": R._MAP[game], "cells": {}}, {}
    t0 = time.time()
    for it in range(max(its) + 1):
        if it in its:
            saved = [(copy.deepcopy(e.state_filter), copy.deepcopy(e.reward_filter)) for e in agent.envs]
            agent.collect_saps(params["t"], should_log=False)               # warm: normaliser after rollout k
            torch_utils.RunningStat.push = lambda self, x: None          # freeze normalisers
            ck = checkpoint_arrays(agent)                                  # exact params + normaliser state
            arrays.update({f"it{it}_{k}": v for k, v in ck.items()})
            try:
                ch.manual_seed(10_000 + 97 * seed + it); np.random.seed(10_000 + 97 * seed + it)
                g_ref, g_ref_raw = both(agent, ref)
                cell = {}
                for n in sizes:
                    G, Gr = zip(*[both(agent, n) for _ in range(m)])
                    G, Gr = np.stack(G), np.stack(Gr)
                    cell[n] = {**stats(G, g_ref), "raw": stats(Gr, g_ref_raw)}
                    arrays[f"it{it}_n{n}"] = G.astype(np.float32)
                arrays[f"it{it}_ref"] = g_ref.astype(np.float32)
                res["cells"][it] = cell
                print(f"[METRIC] {env}_{alg} s{seed} it{it} " + " ".join(
                    f"n={n}:pw={c['pairwise']:.3f}/tt={c['to_true']:.3f}" for n, c in cell.items()), flush=True)
            finally:
                torch_utils.RunningStat.push = push
                for e, (sf, rf) in zip(agent.envs, saved):
                    e.state_filter, e.reward_filter = sf, rf
        if it < max(its):
            R.RECORD.append({"it": it})
            agent.train_step()
    res["wall_s"] = time.time() - t0
    np.savez_compressed(out, **arrays)
    json.dump(res, open(out.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    try:                                   # one worker per output (lets several xargs chains share the job list)
        os.close(os.open(a[4] + ".lock", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        print("[INFO] already claimed", a[4]); sys.exit(0)
    kw = {}
    for x in a[5:]:
        k, v = x.split("=")
        kw[k] = tuple(int(y) for y in v.split(",")) if k in ("its", "sizes") else int(v)
    main(a[1], a[2], int(a[3]), a[4], **kw)
