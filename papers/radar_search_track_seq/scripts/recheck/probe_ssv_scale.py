"""Recheck probe: which SR reading reproduces the paper's printed triple WITH its +- read as a per-timestep pooled SD?
Paper §3: Static -25.2 +- 6.48; Coverage -0.0502 +- 0.0344; Random/PPO cluster -0.137..-0.127.
Scale-free targets: CoV(Static)=0.257, CoV(Cov)=0.685, SR(Cov)/SR(Rnd)=0.380, SR(Static)/SR(Rnd)=191.
Pure numpy re-implementation of data.ssv_at (flat-angle isotropic Gaussian, FIFO of N_SV, sigma_k = s0 (1+g)^k),
cross-checked against the env value at the primary constants. Descriptive; nothing here changes a primary."""
import numpy as np, json, itertools, math
T = 1000; NA = 19; H = 60.0
cen = np.linspace(-H, H, NA)
def actions(policy, seed=0):
    if policy == 'static':
        return np.tile(np.array([[7, 11]]), (T, 1))
    if policy == 'coverage':
        r = np.arange(T); return np.stack([r % NA, (r // NA) % NA], 1)
    rng = np.random.default_rng(seed); return rng.integers(0, NA, (T, 2))
def ssv_series(a, s0, g, nsv, after_push):
    az, el = cen[a[:, 0]], cen[a[:, 1]]
    out = np.zeros(T)
    for t in range(T):
        if after_push:
            idx = np.arange(max(0, t - nsv + 1), t + 1)
        else:
            idx = np.arange(max(0, t - nsv), t)
        if idx.size == 0: continue
        age = (t - idx).astype(float) + (0.0 if after_push else 0.0)
        if not after_push: age = (t - idx).astype(float)  # ages 1..N (look at t, scans at t-1..t-N)
        sig = s0 * (1 + g) ** age
        d2 = (az[t] - az[idx]) ** 2 + (el[t] - el[idx]) ** 2
        out[t] = np.sum(np.exp(-d2 / (2 * sig ** 2)) / (2 * np.pi * sig ** 2))
    return out
def pmax(s0, g, nsv):
    return sum(1 / (2 * math.pi * (s0 * (1 + g) ** k) ** 2) for k in range(nsv))
S0 = {'radial': 4.5 / math.sqrt(2 * math.log(20)), 'axis': 4.5 / 1.959964}
target = dict(cv_static=6.48 / 25.2, cv_cov=0.0344 / 0.0502, cov_rnd=0.0502 / 0.132, sta_rnd=25.2 / 0.132)
rows = []
acts = {p: actions(p) for p in ('static', 'coverage')}
rnd = [actions('random', s) for s in range(3)]
for mode, g, nsv, ap in itertools.product(('radial', 'axis'), (0.002, 0.005, 0.01, 0.015, 0.02, 0.024, 0.03, 0.04, 0.05),
                                          (25, 50, 100, 200, 361, 500, 1000), (False, True)):
    s0 = S0[mode]
    st = ssv_series(acts['static'], s0, g, nsv, ap); cv = ssv_series(acts['coverage'], s0, g, nsv, ap)
    rn = np.concatenate([ssv_series(r, s0, g, nsv, ap) for r in rnd])
    pm = pmax(s0, g, nsv); phi0 = 1 / (2 * math.pi * s0 ** 2)
    d = dict(mode=mode, g=g, nsv=nsv, after_push=ap,
             cv_static=st.std(ddof=1) / st.mean(), cv_cov=cv.std(ddof=1) / cv.mean(),
             cov_rnd=cv.mean() / rn.mean(), sta_rnd=st.mean() / rn.mean(),
             static_mean_pmax=st.mean() / pm, static_mean_phi0=st.mean() / phi0, static_mean_raw_deg=st.mean(),
             static_sd_phi0=st.std(ddof=1) / phi0, cov_mean_phi0=cv.mean() / phi0, cov_sd_phi0=cv.std(ddof=1) / phi0,
             rnd_mean_phi0=rn.mean() / phi0)
    d['logerr'] = float(sum(np.log(d[k] / target[k]) ** 2 for k in target))
    rows.append(d)
rows.sort(key=lambda r: r['logerr'])
import os; os.makedirs('runs/recheck/measurement', exist_ok=True)
json.dump(rows, open('runs/recheck/measurement/probe_ssv_scale.json', 'w'), indent=1)
prim = [r for r in rows if r['mode'] == 'radial' and r['g'] == 0.01 and r['nsv'] == 100 and not r['after_push']][0]
print('PRIMARY', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in prim.items()})
print('targets', {k: round(v, 4) for k, v in target.items()})
for r in rows[:12]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
