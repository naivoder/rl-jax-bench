"""Recheck probe (equations lens): which p_max normaliser / sigma0 / gamma reading reproduces the paper's ABSOLUTE
search rewards (Static -25.2 +- 6.48, Random ~ -0.13, Coverage -0.0502 +- 0.0344)? Pure numpy, CPU, no env.
Mirrors data.ssv_at (flat-angle distance, sigma_k = sigma0 (1+gamma)^age, FIFO of N_SV, before-push ages >= 1).
Statistics: per-episode mean of r_SV over T=1000 steps (incl. FIFO warm-up), and the pooled per-step SD
(the paper's +- could be either; both reported)."""
import math, sys, json
import numpy as np

NA, HALF, T = 19, 60.0, 1000
cen = np.linspace(-HALF, HALF, NA)

def sig0(mode):
    return 4.5 / math.sqrt(2 * math.log(20)) if mode == "radial" else 4.5 / 1.959963984540054

def run(policy, gamma, nsv, mode, norm, age0, rng, n_ep=4):
    s0 = sig0(mode)
    phi0 = lambda k: 1.0 / (2 * math.pi * (s0 * (1 + gamma) ** k) ** 2)
    pmax_full = sum(phi0(k) for k in range(nsv))
    ep_means, steps = [], []
    for e in range(n_ep):
        st = (rng.integers(NA), rng.integers(NA))
        fa, fe, ft = [], [], []
        vals = []
        for t in range(1, T + 1):
            if policy == "static":
                a = st
            elif policy == "random":
                a = (rng.integers(NA), rng.integers(NA))
            else:
                r = t - 1
                a = (r % NA, (r // NA) % NA)
            la, le = cen[a[0]], cen[a[1]]
            if age0:
                fa.insert(0, la); fe.insert(0, le); ft.insert(0, t)
                fa, fe, ft = fa[:nsv], fe[:nsv], ft[:nsv]
            A = np.array(fa); E = np.array(fe); age = t - np.array(ft, float)
            if len(A):
                sg = s0 * (1 + gamma) ** age
                v = np.sum(np.exp(-((la - A) ** 2 + (le - E) ** 2) / (2 * sg ** 2)) / (2 * np.pi * sg ** 2))
            else:
                v = 0.0
            if norm == "full":
                p = pmax_full
            elif norm == "single":
                p = phi0(0)
            elif norm == "growing":       # p_max,T with T = current step count (uncapped by N_SV)
                p = sum(phi0(k) for k in range(min(t, 2000)))
            vals.append(-v / p)
            if not age0:
                fa.insert(0, la); fe.insert(0, le); ft.insert(0, t)
                fa, fe, ft = fa[:nsv], fe[:nsv], ft[:nsv]
        vals = np.array(vals)
        ep_means.append(vals.mean()); steps.append(vals)
        if policy != "random" and policy != "static":
            break
    allv = np.concatenate(steps)
    return float(np.mean(ep_means)), float(allv.std()), float(np.std(ep_means)) if len(ep_means) > 1 else 0.0

if __name__ == "__main__":
    out = []
    rng = np.random.default_rng(0)
    for mode in ("radial", "axis"):
        for nsv in (50, 100):
            for gamma in (0.01, 0.015, 0.02, 0.024, 0.03):
                for norm in ("full", "single"):
                    for age0 in (False,):
                        row = dict(mode=mode, nsv=nsv, gamma=gamma, norm=norm, age0=age0)
                        for pol in ("static", "random", "coverage"):
                            m, sd_step, sd_ep = run(pol, gamma, nsv, mode, norm, age0, rng)
                            row[pol] = (round(m, 5), round(sd_step, 5), round(sd_ep, 5))
                        row["cov/rnd"] = round(row["coverage"][0] / row["random"][0], 3)
                        row["sta/rnd"] = round(row["static"][0] / row["random"][0], 1)
                        out.append(row)
                        print(json.dumps(row), flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "probe_ssv_norm.json", "w"), indent=1)
