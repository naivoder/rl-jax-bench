"""Isolated driver for the official HCA snippet (github.com/hca-neurips2019/hca @ 3a564ad).

Run with this directory's own venv (numpy only):  venv/bin/python -I run_official.py in.npz out.npz
The snippet does not import numpy, so it is exec'd into a namespace that provides ``np``. The repo file is
never modified. Input: per-episode arrays ep{k}_states/actions/rewards (observation ids incl. the terminal
step, dummy action 0 there), pi [n_obs, A], V [n_obs], h_state [A, n_obs, n_obs], h_return [A, n_obs, n_bins],
return_bins [n_bins], gamma. Output: per-episode dlogits/dV/dlogits_h for StateHCA and ReturnHCA.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# authors' clone lives in $OFFICIAL_ROOT/repo (default: <paper folder>/external/official/repo)
OFFICIAL = os.environ.get("OFFICIAL_ROOT", os.path.join(HERE, "..", "..", "external", "official"))
ns = {"np": np}
with open(os.path.join(OFFICIAL, "repo", "hca_classes.py")) as f:
    exec(compile(f.read(), "hca_classes.py", "exec"), ns)

d = np.load(sys.argv[1])
pi, V, gamma = d["pi"], d["V"], float(d["gamma"])
n_obs, n_a = pi.shape
s_hca = ns["StateHCA"](n_obs, n_a)
r_hca = ns["ReturnHCA"](n_obs, n_a, d["return_bins"])
out = {}
for k in range(int(d["n_episodes"])):
    st, ac, rw = d[f"ep{k}_states"], d[f"ep{k}_actions"], d[f"ep{k}_rewards"]
    for tag, alg, h in (("state", s_hca, d["h_state"]), ("return", r_hca, d["h_return"])):
        dl, dv, dh = alg.update(pi, V, h, st, ac, rw, gamma)
        out[f"ep{k}_{tag}_dlogits"], out[f"ep{k}_{tag}_dV"], out[f"ep{k}_{tag}_dh"] = dl, dv, dh
np.savez(sys.argv[2], **out)
print(f"official snippet ran on {int(d['n_episodes'])} episodes")
