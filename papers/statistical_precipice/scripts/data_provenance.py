"""Does the public Dopamine mirror reproduce the paper's DQN (Nature) / DQN (Adam) rows under ANY
agent label or 'final' iteration choice? Point statistics only (no RNG): Fig 9 median / IQM / mean / OG and A.28
DP-25 / DP-50 / P(X>1) per (Dopamine label, iteration protocol) vs the paper's vector read, plus the data-identity
check 'is our point inside the paper's own 95 % CI' for every Fig 9 agent. numpy + scipy only. Writes
runs/full/data_provenance.json.

    python scripts/data_provenance.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src import data as D  # noqa: E402  (numpy-only module)

STATS = ("median", "iqm", "mean", "og", "dp25", "dp50", "superhuman")


def stats7(x):
    tm = x.mean(0)
    s = np.sort(x.ravel())
    K = s.size
    return dict(median=float(np.median(tm)), iqm=float(st.trim_mean(x, 0.25, axis=None)), mean=float(tm.mean()),
                og=float(1 - np.minimum(x, 1).mean()), dp25=float(s[:int(.25 * K)].mean()),
                dp50=float(s[:int(.5 * K)].mean()), superhuman=float((x > 1).mean()))


def main():
    R = json.load(open(os.path.join(REPO, "runs", "full", "ale.json")))
    paper = {}
    for sec in ("fig9", "a28"):
        for k, r in R[sec].items():
            a, s = k.split("|")
            paper.setdefault(a, {})[s] = dict(point=r["paper"], ci=r["paper_ci"])
    games = D.ale_games()
    src = D.sources()["normalisation"]
    per_game = {g: D.parse_dopamine_game(os.path.join(D.DATA, "dopamine", f"{g.lower()}.json")) for g in games}
    labels = sorted(next(iter(per_game.values())).keys())
    cubes = {}
    for lab in labels:
        c = np.stack([per_game[g][lab] for g in games], axis=1)  # (runs, games, iterations)
        cubes[lab] = np.moveaxis(D.normalise(np.moveaxis(c, 1, -1), games, src["ale_random"]["value"], src["ale_human"]["value"]), -1, 1)
    n_iter = {lab: int(c.shape[2]) for lab, c in cubes.items()}
    protos = {f"iter{i}": (lambda c, i=i: c[:, :, i]) for i in range(150, 199)}
    protos.update({f"last{k}_mean": (lambda c, k=k: c[:, :, -k:].mean(2)) for k in (2, 3, 5, 10)})
    protos["max_over_iters"] = lambda c: c.max(2)

    def rms(v, tgt):
        return float(np.sqrt(np.mean([((v[s] - tgt[s]["point"]) / tgt[s]["point"]) ** 2 for s in STATS])))

    grid = {}
    for target in ("DQN (Nature)", "DQN (Adam)"):
        rows = []
        for lab in labels:
            for pn, pf in protos.items():
                v = stats7(pf(cubes[lab]))
                rows.append(dict(label=lab, protocol=pn, rms_rel=rms(v, paper[target]),
                                 n_inside_paper_ci=int(sum(paper[target][s]["ci"][0] <= v[s] <= paper[target][s]["ci"][1] for s in STATS)),
                                 values=v))
        rows.sort(key=lambda r: r["rms_rel"])
        best_per_label = {lab: next(r for r in rows if r["label"] == lab) for lab in labels}
        final_per_label = {lab: next(r for r in rows if r["label"] == lab and r["protocol"] == "iter198") for lab in labels}
        grid[target] = dict(best=rows[0], best_per_label=best_per_label, final_per_label=final_per_label,
                            top5=[{k: r[k] for k in ("label", "protocol", "rms_rel", "n_inside_paper_ci")} for r in rows[:5]])
        print(f"[METRIC] target={target} best={rows[0]['label']}/{rows[0]['protocol']} rms={rows[0]['rms_rel']:.4f} inside={rows[0]['n_inside_paper_ci']}/7")
        for lab in labels:
            b, f = best_per_label[lab], final_per_label[lab]
            print(f"[INFO]   {target:13s} <- {lab:26s} final rms {f['rms_rel']:.3f} inside {f['n_inside_paper_ci']}/7 | best {b['protocol']:15s} rms {b['rms_rel']:.3f} inside {b['n_inside_paper_ci']}/7")
    # data-identity check under the shipped mapping and final iteration
    ident = {}
    for d_lab, p_lab in D.DOPAMINE_AGENTS.items():
        v = stats7(cubes[d_lab][:, :, -1])
        ident[p_lab] = {s: dict(ours=v[s], paper=paper[p_lab][s]["point"], paper_ci=paper[p_lab][s]["ci"],
                                inside=bool(paper[p_lab][s]["ci"][0] <= v[s] <= paper[p_lab][s]["ci"][1]),
                                rel=abs(v[s] - paper[p_lab][s]["point"]) / abs(paper[p_lab][s]["point"])) for s in STATS}
        outside = [s for s in STATS if not ident[p_lab][s]["inside"]]
        print(f"[METRIC] identity {p_lab:13s} points outside the paper's own 95% CI: {len(outside)}/7 {outside}")
    out = dict(n_iterations=n_iter, final_iteration_index=198, final_frames_M=199,
               note="Dopamine iteration i = 1M frames (250k agent steps x 4); index 198 is the 199th iteration -> 199M frames",
               labels=labels, protocols=list(protos), grid=grid, identity_shipped_mapping=ident)
    p = os.path.join(REPO, "runs", "full", "data_provenance.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[INFO] wrote {p}")


if __name__ == "__main__":
    main()
