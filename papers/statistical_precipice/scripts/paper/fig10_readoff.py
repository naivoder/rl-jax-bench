"""recheck: read paper Fig 10 right (IQM vs frames, 95 % bands) from the PDF's vector paths
(page 8). x and y are mapped by least-squares fits of the panel's grey gridlines to their tick labels (as for every other
vector read; no calibration against any of our numbers). Each agent's point curve is the 10-point polyline in its
Colab 'colorblind' colour; its band is the 20-vertex fill_between polygon of the same colour (lower edge, then upper).

    python scripts/paper/fig10_readoff.py   -> runs/full/paper_fig10.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pymupdf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.environ.get("PAPER_PDF", os.path.join(REPO, "external", "data", "2021-deep-rl-at-the-edge-of-the-statistical-precipice.pdf"))
FRAMES = [1, 10, 25, 50, 75, 100, 125, 150, 175, 200]
COL = {"DQN (Nature)": (0.0, 0.45, 0.7), "DQN (Adam)": (0.87, 0.56, 0.02), "C51": (0.01, 0.62, 0.45),
       "REM": (0.84, 0.37, 0.0), "IQN": (0.8, 0.47, 0.74), "Rainbow": (0.79, 0.57, 0.38),
       "M-IQN": (0.98, 0.69, 0.89), "DreamerV2": (0.58, 0.58, 0.58)}   # Colab cell 59 sns 'colorblind' order


def key(c):
    return tuple(round(x, 2) for x in c)


def main():
    pg = pymupdf.open(PDF)[7]
    dr = [d for d in pg.get_drawings() if d["rect"].x0 > 320 and 255 < d["rect"].y0 < 405]
    words = pg.get_text("words")
    grey = [d for d in dr if d["type"] == "s" and d.get("color") and abs(d["color"][0] - 0.8) < 0.01]
    hg = sorted({round(d["rect"].y0, 4) for d in grey if d["rect"].height < 0.01})
    vg = sorted({round(d["rect"].x0, 4) for d in grey if d["rect"].width < 0.01})
    ylab = [((w[1] + w[3]) / 2, float(w[4])) for w in words if 310 < w[0] < 330 and 280 < w[1] < 390]
    xlab = [((w[0] + w[2]) / 2, float(w[4])) for w in words if 382 < w[1] < 392 and 330 < w[0] < 495 and w[4].isdigit()]
    ypairs = [(g, min(ylab, key=lambda t: abs(t[0] - g))[1]) for g in hg]
    xpairs = [(g, min(xlab, key=lambda t: abs(t[0] - g))[1]) for g in vg]
    ay, by = np.polyfit(*np.array(ypairs).T, 1)
    ax, bx = np.polyfit(*np.array(xpairs).T, 1)
    res_y = float(np.max(np.abs(np.polyval([ay, by], [p[0] for p in ypairs]) - [p[1] for p in ypairs])))
    res_x = float(np.max(np.abs(np.polyval([ax, bx], [p[0] for p in xpairs]) - [p[1] for p in xpairs])))
    out = dict(page=8, frames_M=FRAMES, y_grid=ypairs, x_grid=xpairs, y_fit_residual=res_y, x_fit_residual=res_x, agents={})
    for a, c in COL.items():
        lines = [d for d in dr if d["type"] == "s" and d.get("color") and key(d["color"]) == key(c) and len(d["items"]) == 9]
        polys = [d for d in dr if d["type"] == "fs" and d.get("fill") and key(d["fill"]) == key(c) and len(d["items"]) == 20]
        if len(lines) != 1:
            continue
        p = lines[0]
        pts = np.array([[p["items"][0][1].x, p["items"][0][1].y]] + [[it[2].x, it[2].y] for it in p["items"]])
        fx = np.polyval([ax, bx], pts[:, 0])
        assert np.max(np.abs(fx - FRAMES)) < 0.05, (a, fx)
        rec = dict(point=np.polyval([ay, by], pts[:, 1]).tolist(), x_frames_read=fx.tolist())
        if len(polys) == 1:
            q = polys[0]
            qp = np.array([[q["items"][0][1].x, q["items"][0][1].y]] + [[it[2].x, it[2].y] for it in q["items"]])
            qx = np.rint(np.polyval([ax, bx], qp[:, 0])).astype(int)
            qv = np.polyval([ay, by], qp[:, 1])
            rec["lo"] = [float(qv[qx == f].min()) for f in FRAMES]
            rec["hi"] = [float(qv[qx == f].max()) for f in FRAMES]
        out["agents"][a] = rec
        print(f"[INFO] Fig 10 right {a:13s} IQM {np.round(rec['point'], 4).tolist()}")
    path = os.path.join(REPO, "runs", "full", "paper_fig10.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[INFO] grid fit residual x {res_x:.1e} frames, y {res_y:.1e} IQM units; wrote {path}")


if __name__ == "__main__":
    main()
