"""Review-pass report: (1) Breakout CPU arms (DQN1 eps 1e-8, paper Watkins cut, classic Q keepnorm)
on the Figure-4 125k grid vs the full-scale GPU record and the paper; (2) E8 with the -aligned metric, the authors' code
under the same metric, and the  classic_td_keepnorm arm; (3)  version-matched real-Hopper comparison (ours vs the
authors' code, both on gymnasium 0.29.1 + mujoco 2.3.7, 5 seeds each) with the minimum detectable difference.
Usage: python scripts/review_report.py -> runs/review/review_report.json"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats as sst

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rl_jax_bench import stats as st  # noqa: E402

from src import eval as ev  # noqa: E402

R = os.path.join(REPO, "runs")
PAPER = json.load(open(os.path.join(REPO, "src", "configs", "paper_digitized.json")))["figures"]
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
OFF = os.path.join(OFFICIAL_ROOT, "runs")
out = {}


def ms(v):
    v = np.asarray(v, float)
    m, se = st.mean_se(v)
    return dict(mean=m, se=se, n=int(len(v)), per_seed=np.round(v, 3).tolist())


def welch(a, b):
    t = sst.ttest_ind(a, b, equal_var=False)
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    df = float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))
    mde = float((sst.t.ppf(0.975, df) + sst.t.ppf(0.80, df)) * se)     # two-sided 5 %, 80 % power
    return dict(diff=float(np.mean(a) - np.mean(b)), se_diff=se, df=df, p=float(t.pvalue), mde80=mde)


def judge(o, p):
    d = (o - p) / abs(p)
    return float(d), ("MATCH" if abs(d) <= 0.05 else "CLOSE" if abs(d) <= 0.20 else "DIVERGENT")


def binmean(e, lo, hi):
    m = (e[:, 0] > lo) & (e[:, 0] <= hi)
    return float(e[m, 1].mean()) if m.any() else np.nan


def load_lanes(d):
    o = {}
    for p in glob.glob(os.path.join(d, "*", "seed_*", "metrics.json")):
        r = json.load(open(p))
        o.setdefault(r["arm"], {})[r["seed"]] = np.asarray(r["episodes"], float).reshape(-1, 3)
    return o


# ------------------------------------------------------------------ (1) Breakout CPU arms
d = os.path.join(R, "review", "review_e3_Breakout")
cpu, done_step, src = load_lanes(d), None, "metrics.json (finished)"
if not cpu and glob.glob(os.path.join(d, "ckpt", "*_host.npz")):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(REPO, "src", "configs", "review_e3_cpu.yaml")))
    lanes = [(a, s) for a in cfg["arms"] for s in cfg["seeds"]]
    hp = sorted(glob.glob(os.path.join(d, "ckpt", "*_host.npz")))[-1]
    done_step = int(re.search(r"s(\d+)_host", hp).group(1))
    z = np.load(hp)["ep"]
    for i, (a, s) in enumerate(lanes):
        cpu.setdefault(a, {})[s] = z[z[:, 0] == i][:, 1:]
    src = f"checkpoint host record at {done_step} steps (run capped)"
if cpu:
    g3 = load_lanes(os.path.join(R, "full", "full_e3_Breakout"))
    last = done_step or max(e[:, 0].max() for rr in cpu.values() for e in rr.values())
    grid = [x for x in (0.125, 0.25, 0.5, 0.75, 0.875, 1.0) if x * 1e6 <= last + 1]
    ref = {"dqn1": ("dqn1", "dqn1"), "stream_q_watkins_paper": ("stream_q", "stream_q"),
           "classic_q_keepnorm": ("classic_q", "classic_q")}
    e3 = {"source": src, "grid": grid, "arms": {}}
    for a, rr in cpu.items():
        g3arm, ser = ref[a]
        rows = {}
        for x in grid:
            v = [binmean(rr[s], (x - 0.125) * 1e6, x * 1e6) for s in sorted(rr)]
            w = [binmean(g3[g3arm][s], (x - 0.125) * 1e6, x * 1e6) for s in sorted(g3[g3arm])]
            pv = ev.grid_value(PAPER["fig4"]["Breakout"][ser]["points"], x)
            row = dict(ours=ms(v), full_gpu_ref_arm=g3arm, full_gpu=ms(w), vs_full_welch=welch(v, w), paper=pv)
            if pv is not None:
                row["delta_rel"], row["judgment"] = judge(np.mean(v), pv)
            rows[str(x)] = row
        e3["arms"][a] = rows
    out["breakout_cpu"] = e3

# ------------------------------------------------------------------ (2) E8
def e8_runs(path):
    o = {}
    for p in glob.glob(os.path.join(path, "*", "seed_*", "metrics.json")):
        r = json.load(open(p))
        o.setdefault(r["arm"], {})[r["seed"]] = r["summary"]
    return o


e8 = e8_runs(os.path.join(R, "review", "e8_ETTm2"))
sens = e8_runs(os.path.join(R, "review_sens", "e8_ETTm2"))
old = e8_runs(os.path.join(R, "full", "e8_ETTm2"))
if e8:
    seeds = sorted(e8["stream_td"])
    E = {"identity_vs_full_max_abs": float(max(abs(e8[a][s]["mse_last_official"] - old[a][s]["mse_last"])
                                                for a in old for s in old[a]))}
    allr = dict(e8, **sens)
    for k in ("mse_first", "mse_last", "mse_last_excl_tail", "mse_first_official", "mse_last_official",
              "corr_last_excl_tail"):
        E[k] = {a: ms([allr[a][s][k] for s in seeds]) for a in allr}
    refs = []
    for f in sorted(glob.glob(os.path.join(OFF, "td_ett", "ref_seed*.npz"))):
        z = np.load(f)
        refs.append(ev.ett_summary(z["pred"], ev.discounted_returns(z["cum"], 0.99), z["cum"]))
    E["authors"] = {k: ms([r[k] for r in refs]) for k in ("mse_first", "mse_last", "mse_last_excl_tail",
                                                         "mse_first_official", "mse_last_official")}
    E["stream_vs_authors_last_welch"] = welch([e8["stream_td"][s]["mse_last"] for s in seeds],
                                              [r["mse_last"] for r in refs])
    pc = st.paired_compare([e8["classic_td"][s]["mse_last"] for s in seeds], [e8["stream_td"][s]["mse_last"] for s in seeds])
    E["H9_aligned"] = pc
    if sens:
        ss = sorted(sens["classic_td_keepnorm"])
        E["keepnorm_minus_stream_last"] = st.paired_compare([sens["classic_td_keepnorm"][s]["mse_last"] for s in ss],
                                                            [e8["stream_td"][s]["mse_last"] for s in ss])
        E["keepnorm_minus_stream_first"] = st.paired_compare([sens["classic_td_keepnorm"][s]["mse_first"] for s in ss],
                                                             [e8["stream_td"][s]["mse_first"] for s in ss])
        E["keepnorm_minus_classic_last"] = st.paired_compare([sens["classic_td_keepnorm"][s]["mse_last"] for s in ss],
                                                             [e8["classic_td"][s]["mse_last"] for s in ss])
    out["e8"] = E

# ------------------------------------------------------------------ (3)  version-matched Hopper
PAT = re.compile(r"Episodic Return: ([-\d.e+]+), Time Step (\d+)")


def official_eps(logf):
    m = PAT.findall(open(logf).read())
    return np.array([[float(t), float(r), 0.0] for r, t in m]).reshape(-1, 3)


vm = load_lanes(os.path.join(R, "review", "ours_mujoco_vm"))
if vm:
    ours = vm["stream_ac"]
    auth = {}
    for s in (1, 2, 3):
        auth[s] = official_eps(os.path.join(OFF, "ac_hopper_2m", f"log_seed{s}.txt"))
    for s in (4, 5):
        f = os.path.join(OFF, "ac_hopper_350k_s45", f"log_seed{s}.txt")
        if os.path.exists(f):
            auth[s] = official_eps(f)
    old_ours = load_lanes(os.path.join(R, "full", "ours_mujoco")).get("stream_ac", {})
    # runs stop at a fixed step count; the last episode ENDS a little before it, so the reach is rounded up to 1k
    reach = np.ceil(min(min(e[:, 0].max() for e in ours.values()), min(e[:, 0].max() for e in auth.values())) / 1e3) * 1e3
    reach = 350_000.0 if reach > 345_000 else reach      # both sides ran exactly 3.5e5 steps
    H = {"steps_common": float(reach), "n_ours": len(ours), "n_authors": len(auth), "bins": {}}
    for x in (1e5, 2e5, 3e5, 3.5e5):
        if x > reach + 1:
            continue
        a = [binmean(ours[s], x - 5e4, x) for s in sorted(ours)]
        b = [binmean(auth[s], x - 5e4, x) for s in sorted(auth)]
        pv = ev.grid_value(PAPER["fig7"]["Hopper"]["stream_ac"]["points"], x / 1e6)
        row = dict(ours_vm=ms(a), authors=ms(b), welch=welch(a, b), paper=pv)
        if old_ours:
            c = [binmean(old_ours[s], x - 5e4, x) for s in sorted(old_ours)]
            row["ours_gym1.1.1_mujoco3.15_seeds1-3"] = ms(c)
        H["bins"][str(x)] = row
    # whole-curve summary: mean of the per-run 50k bins up to the common step
    nb = int(reach // 5e4)
    ca = [np.nanmean(ev.run_bins(ours[s], nb * 50_000, 50_000)) for s in sorted(ours)]
    cb = [np.nanmean(ev.run_bins(auth[s], nb * 50_000, 50_000)) for s in sorted(auth)]
    H["curve_mean"] = dict(ours=ms(ca), authors=ms(cb), welch=welch(ca, cb), n_bins=nb)
    out["hopper_vm"] = H


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


os.makedirs(os.path.join(R, "review"), exist_ok=True)
json.dump(_clean(out), open(os.path.join(R, "review", "review_report.json"), "w"), indent=1)
print(json.dumps(_clean(out), indent=1)[:6000])
