"""Reference driver around the OFFICIAL HCA snippet (github.com/hca-neurips2019/hca @ 3a564ad, hca_classes.py).

Run with this directory's own venv (numpy only), never with the JAX venv:
    venv/bin/python -I ref_driver.py lockstep IN.npz OUT.npz
    venv/bin/python -I ref_driver.py independent --exp E1 --runs 100 --seed 0 --out OUT.json
    venv/bin/python -I ref_driver.py fig3r --runs 100 --seed 0 --out OUT.json

What is official and what is not (labelled):
  * OFFICIAL, exec'd unmodified from repo/hca_classes.py: StateHCA.update / ReturnHCA.update -- the per-episode
    policy-logit, value and hindsight-logit gradients.
  * NOT RELEASED by the authors, written here as a plain-numpy PORT from the paper text (paper §5, Fig. 2,
    App. E) and the reproduction's assumptions (A-ENV-*, A-DE-*, A-AB-*, A-BINS, A-LR,
    A-UPDATE): the three environments, the episode sampler (inverse-CDF on supplied uniforms), the caller that
    applies the gradients with learning rates once per episode, the DP value metric, and the Fig. 3-right
    estimator formulas (Thm 1 / Thm 5). This port does NOT import the JAX re-implementation; it shares only the
    state numbering convention (needed for inverse-CDF lockstep) with it.
The snippet keeps h as probabilities h[a, x, y] and returns gradients w.r.t. logits of the same shape (softmax
over axis 0 implied by its CE gradient onehot - h). We therefore keep logits psi[a, x, y] and pass softmax(psi).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# authors' clone lives in $OFFICIAL_ROOT/repo (default: <paper folder>/external/official/repo)
OFFICIAL = os.environ.get("OFFICIAL_ROOT", os.path.join(HERE, "..", "..", "external", "official"))
_ns = {"np": np}
with open(os.path.join(OFFICIAL, "repo", "hca_classes.py")) as _f:
    exec(compile(_f.read(), "hca_classes.py", "exec"), _ns)
StateHCA, ReturnHCA = _ns["StateHCA"], _ns["ReturnHCA"]
A = 2


# ------------------------------------------------------------------ env port (paper §5, Fig. 2)
def env_shortcut(n=5, absorb=0.1):
    """States 0..n-1 chain, G=n (+1), D=n+1 (0); -1 per decision state; a0=short->G, a1=long->next
    (last long -> G); any transition -> D w.p. absorb."""
    S, G, D = n + 2, n, n + 1
    P = np.zeros((S, A, S))
    for k in range(n):
        P[k, 0, G] += 1 - absorb
        P[k, 1, k + 1 if k < n - 1 else G] += 1 - absorb
        P[k, :, D] += absorb
    r = np.r_[-np.ones(n), 1.0, 0.0]
    term = np.r_[np.zeros(n, bool), True, True]
    return dict(P=P, r=r, sd=np.zeros(S), obs=np.arange(S), term=term, s0=0, L=n + 1, n_obs=S)


def env_delayed_effect(n=3, sigma=0.0, aliasing="per_position"):
    """x0=0; a0 -> + branch states 1..n, a1 -> - branch n+1..2n; chain moves forward regardless of action;
    chain rewards N(0, sigma^2); F+ = 2n+1 (+1), F- = 2n+2 (-1). Obs: chain position j (per_position alias)."""
    assert aliasing == "per_position", aliasing  # the only aliasing the paper experiments use (A-DE-ALIAS)
    S = 2 * n + 3
    Fp, Fm = 2 * n + 1, 2 * n + 2
    P = np.zeros((S, A, S))
    obs = np.zeros(S, int)
    sd = np.zeros(S)
    for b in (0, 1):
        P[0, b, 1 + b * n] = 1.0
        for j in range(1, n + 1):
            s = 1 + b * n + j - 1
            P[s, :, s + 1 if j < n else (Fp if b == 0 else Fm)] = 1.0
            obs[s], sd[s] = j, sigma
    obs[Fp], obs[Fm] = n + 1, n + 2
    r = np.zeros(S)
    r[Fp], r[Fm] = 1.0, -1.0
    term = np.zeros(S, bool)
    term[[Fp, Fm]] = True
    return dict(P=P, r=r, sd=sd, obs=obs, term=term, s0=0, L=n + 2, n_obs=n + 3)


def env_bandit(eps=0.1, sigma=1.5, hidden=False):
    """x0=0; arm i -> state i+1 w.p. 1-eps, the other arm state w.p. eps; reward N(i+1, sigma^2)."""
    P = np.zeros((3, A, 3))
    P[0, 0, 1], P[0, 0, 2] = 1 - eps, eps
    P[0, 1, 1], P[0, 1, 2] = eps, 1 - eps
    obs = np.array([0, 1, 1]) if hidden else np.array([0, 1, 2])
    return dict(P=P, r=np.array([0.0, 1.0, 2.0]), sd=np.array([0.0, sigma, sigma]), obs=obs,
                term=np.array([False, True, True]), s0=0, L=2, n_obs=int(obs.max()) + 1)


ENVS = {"shortcut": env_shortcut, "delayed_effect": env_delayed_effect, "ambiguous_bandit": env_bandit}


def bin_centres(env, n_bins, widen=2.0):
    """A-BINS (assumption, not in the paper or the snippet): linspace over the noise-free return range
    widened by `widen` x the largest summed reward-noise std along a path."""
    lo, hi, sd = np.inf, -np.inf, 0.0
    stack = [(env["s0"], 0.0, 0.0, 0)]
    while stack:
        s, z, v, d = stack.pop()
        z, v = z + env["r"][s], v + env["sd"][s] ** 2
        if env["term"][s] or d == env["L"] - 1:
            lo, hi, sd = min(lo, z), max(hi, z), max(sd, np.sqrt(v))
            continue
        for s2 in np.nonzero(env["P"][s].sum(0) > 0)[0]:
            stack.append((int(s2), z, v, d + 1))
    return np.linspace(lo - widen * sd, hi + widen * sd, n_bins)


def softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def start_value(env, pi):
    """Exact V^pi(s0) by backward recursion (acyclic env), gamma = 1."""
    V = np.zeros(len(env["r"]))
    for _ in range(env["L"]):
        V = env["r"] + (~env["term"]) * np.einsum("sa,sat,t->s", pi[env["obs"]], env["P"], V)
    return V[env["s0"]]


def q_start(env, pi):
    V = np.zeros(len(env["r"]))
    for _ in range(env["L"]):
        V = env["r"] + (~env["term"]) * np.einsum("sa,sat,t->s", pi[env["obs"]], env["P"], V)
    return env["r"][env["s0"]] + env["P"][env["s0"]] @ V


def rollout(env, pi, u_env, z, u_act):
    """Inverse-CDF sampling on supplied per-step uniforms/normals. Returns observation ids incl. the terminal
    reward-only step, actions (0 at the terminal step: the snippet needs one per state), rewards."""
    s, obs, acts, rews = env["s0"], [], [], []
    for t in range(env["L"]):
        o = env["obs"][s]
        obs.append(o)
        rews.append(env["r"][s] + env["sd"][s] * z[t])
        if env["term"][s]:
            acts.append(0)
            break
        a = min(int(np.sum(u_act[t] > np.cumsum(pi[o])[:-1])), A - 1)
        acts.append(a)
        s = min(int(np.sum(u_env[t] > np.cumsum(env["P"][s, a])[:-1])), len(env["r"]) - 1)
    return np.array(obs), np.array(acts), np.array(rews)


def train_run(env, method, lrs, gamma, centres, U_env, Z, U_act):
    """One learner. lrs = (alpha_pi, alpha_h, alpha_V). One snippet call per episode with the pre-update tables,
    then theta += a*dlogits, psi += a_h*dlogits_h, V += a_V*dV (A-UPDATE; the caller is not released)."""
    n_obs = env["n_obs"]
    theta, V = np.zeros((n_obs, A)), np.zeros(n_obs)
    if method == "state_hca":
        alg, psi = StateHCA(n_obs, A), np.zeros((A, n_obs, n_obs))
    else:
        alg, psi = ReturnHCA(n_obs, A, centres), np.zeros((A, n_obs, len(centres)))
    curve, rets = [start_value(env, softmax(theta, 1))], []
    for e in range(U_env.shape[0]):
        pi = softmax(theta, 1)
        st, ac, rw = rollout(env, pi, U_env[e], Z[e], U_act[e])
        dl, dv, dh = alg.update(pi, V, softmax(psi, 0), st, ac, rw, gamma)
        theta = theta + lrs[0] * dl
        psi = psi + lrs[1] * dh
        V = V + lrs[2] * dv
        curve.append(start_value(env, softmax(theta, 1)))
        rets.append(rw.sum())
    return np.array(curve), np.array(rets), theta, psi, V


# ------------------------------------------------------------------ experiments (configs: paper §5 / App. E)
PAPER_LRS = (0.3, 0.4, 0.3)  # App. E alpha = 0.3, alpha_b = 0.4 (read as alpha_h); V lr ASSUMED = alpha (A-LR)
EXPS = {
    "E1": dict(env="shortcut", kw=dict(n=5, absorb=0.1), episodes=100, n_bins=10, sweep=None),
    "E3": dict(env="delayed_effect", kw=dict(n=5, sigma=0.0), episodes=500, n_bins=3, sweep=None),  # MC only: snippet has no bootstrap
    "E4": dict(env="delayed_effect", kw=dict(n=3, sigma=2.0), episodes=500, n_bins=3, sweep=None),
    "E5": dict(env="delayed_effect", kw=dict(n=3), episodes=500, n_bins=3, sweep=("sigma", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
    "E6": dict(env="ambiguous_bandit", kw=dict(eps=0.1, sigma=1.5, hidden=False), episodes=100, n_bins=10, sweep=None),
    "E7": dict(env="ambiguous_bandit", kw=dict(eps=0.1, sigma=1.5, hidden=True), episodes=100, n_bins=10, sweep=None),
    "E8": dict(env="ambiguous_bandit", kw=dict(sigma=0.5, hidden=True), episodes=100, n_bins=10, sweep=("eps", [0.0, 0.1, 0.2, 0.3, 0.4])),
}


def settings(cfg):
    if cfg["sweep"] is None:
        return [dict(cfg["kw"])]
    k, vals = cfg["sweep"]
    return [dict(cfg["kw"], **{k: v}) for v in vals]


def cmd_lockstep(fin, fout):
    d = np.load(fin, allow_pickle=False)
    env = ENVS[str(d["env"])](**json.loads(str(d["env_kwargs"])))
    method, lrs, gamma, centres = str(d["method"]), d["lrs"], float(d["gamma"]), d["centres"]
    out = {k: [] for k in ("curve", "returns", "theta", "psi", "V")}
    for r in range(d["u_env"].shape[0]):
        res = train_run(env, method, lrs, gamma, centres, d["u_env"][r], d["z"][r], d["u_act"][r])
        for k, v in zip(out, res):
            out[k].append(v)
    np.savez(fout, **{k: np.stack(v) for k, v in out.items()})


def cmd_independent(exp, runs, seed, out_path):
    cfg = EXPS[exp]
    rng = np.random.default_rng([seed, int(exp[1:])])
    res = {"exp": exp, "runs": runs, "seed": seed, "lrs": PAPER_LRS, "settings": []}
    t0 = time.perf_counter()
    for kw in settings(cfg):
        env = ENVS[cfg["env"]](**kw)
        centres = bin_centres(env, cfg["n_bins"])
        E, L = cfg["episodes"], env["L"]
        rec = {"env_kwargs": kw, "methods": {}}
        draws = [(rng.random((E, L)), rng.standard_normal((E, L)), rng.random((E, L))) for _ in range(runs)]
        for m in ("state_hca", "return_hca"):  # common random numbers across the two arms
            curves = np.stack([train_run(env, m, PAPER_LRS, 1.0, centres, *dr)[0] for dr in draws])
            rec["methods"][m] = {"final": curves[:, -1].tolist(), "avg": curves.mean(1).tolist(),
                                 "mean_curve": curves.mean(0).tolist()}
        res["settings"].append(rec)
    res["wall_s"] = time.perf_counter() - t0
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"[INFO] official-ref {exp} runs={runs} wall_s={res['wall_s']:.1f}")


def cmd_fig3r(runs, seed, out_path, rollouts=1000, n_bins=10):
    """Fig. 3 right. Fixed policy pi(long)=p at every chain state; per repetition, 1000 rollouts; the hindsight
    tables are trained online with the snippet's dlogits_h (alpha_h = 0.4); each rollout's estimate uses the
    pre-update tables. Estimators (PORT of Thm 1 / Thm 5; not in the snippet):
      state : Qhat(x0,a) = sum_{k>=0} h(a|x0,X_k)/pi(a|x0) R_k  (k = 0 term = the snippet's j = i term)
      return: Qhat(x0,a) = Z h_z(a|x0,Z)/pi(a|x0)."""
    env = env_shortcut()
    centres = bin_centres(env, n_bins)
    ps = list(np.linspace(0.1, 0.9, 7))
    rng = np.random.default_rng([seed, 2])
    out = {"runs": runs, "seed": seed, "points": []}
    t0 = time.perf_counter()
    for p in ps:
        theta = np.tile([0.0, np.log(p / (1 - p))], (env["n_obs"], 1))
        pi = softmax(theta, 1)
        qs = q_start(env, pi)
        rec = {"p_long": p, "oracle": float(qs[0] - qs[1]), "state_hca": [], "return_hca": []}
        for _ in range(runs):
            s_alg, r_alg = StateHCA(env["n_obs"], A), ReturnHCA(env["n_obs"], A, centres)
            psi_s, psi_r = np.zeros((A, env["n_obs"], env["n_obs"])), np.zeros((A, env["n_obs"], n_bins))
            est_s, est_r = [], []
            for _k in range(rollouts):
                st, ac, rw = rollout(env, pi, rng.random(env["L"]), np.zeros(env["L"]), rng.random(env["L"]))
                hs, hr = softmax(psi_s, 0), softmax(psi_r, 0)
                q = (hs[:, st[0], st] * rw[None, :]).sum(1) / pi[st[0]]
                est_s.append(q[0] - q[1])
                Z = rw.sum()
                b = int(np.abs(centres - Z).argmin())
                q = Z * hr[:, st[0], b] / pi[st[0]]
                est_r.append(q[0] - q[1])
                psi_s = psi_s + 0.4 * s_alg.update(pi, np.zeros(env["n_obs"]), hs, st, ac, rw, 1.0)[2]
                psi_r = psi_r + 0.4 * r_alg.update(pi, np.zeros(env["n_obs"]), hr, st, ac, rw, 1.0)[2]
            rec["state_hca"].append(float(np.mean(est_s)))
            rec["return_hca"].append(float(np.mean(est_r)))
        out["points"].append(rec)
        print(f"[METRIC] official-ref E2 p_long={p:.3f} oracle={rec['oracle']:.4f} "
              f"state={np.mean(rec['state_hca']):.4f} return={np.mean(rec['return_hca']):.4f}")
    out["wall_s"] = time.perf_counter() - t0
    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "lockstep":
        cmd_lockstep(sys.argv[2], sys.argv[3])
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--exp")
        ap.add_argument("--runs", type=int, default=100)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--out", required=True)
        a = ap.parse_args()
        if cmd == "independent":
            cmd_independent(a.exp, a.runs, a.seed, a.out)
        elif cmd == "fig3r":
            cmd_fig3r(a.runs, a.seed, a.out)
        else:
            raise SystemExit(cmd)
