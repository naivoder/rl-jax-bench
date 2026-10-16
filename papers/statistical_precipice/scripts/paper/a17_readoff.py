"""recheck: read paper Fig A.17 (expected IQM of N subsampled runs, N = 3, 5, 10, 25, 50, 100,
for DER, OTR, CURL, DrQ(eps), SPR on Atari 100k) from the PDF. The bars are an embedded raster, but the axes and the
tick marks are vector: y is calibrated by a least-squares fit of the vector tick y positions to their labels, x by the
vector tick x positions (one per N), and each bar top is the first fully saturated pixel row of a 1200 dpi render in a
7-pixel column centred on that tick. Read precision: 1 px = 0.06 pt = ~2e-5 normalised units (reported per panel).

    python scripts/paper/a17_readoff.py   -> runs/full/paper_a17.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pymupdf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.environ.get("PAPER_PDF", os.path.join(REPO, "external", "data", "2021-deep-rl-at-the-edge-of-the-statistical-precipice.pdf"))
PAGE = 20
NAMES = ["DER", "OTR", "CURL", "DrQ(eps)", "SPR"]   # panel order left -> right (paper A.17)
N = [3, 5, 10, 25, 50, 100]
DPI = 1200


def main():
    pg = pymupdf.open(PDF)[PAGE - 1]
    dr = pg.get_drawings()
    words = pg.get_text("words")
    s = DPI / 72
    axes = [d["rect"] for d in dr if d["type"] == "f" and d.get("fill") == (1.0, 1.0, 1.0) and d["rect"].y1 < 140
            and d["rect"].width < 60]
    axes.sort(key=lambda r: r.x0)
    assert len(axes) == len(NAMES), axes
    out = dict(page=PAGE, dpi=DPI, N=N, panels={})
    for nm, ax in zip(NAMES, axes):
        yt = [d["rect"].y0 for d in dr if d["type"] == "fs" and abs(d["rect"].x1 - ax.x0) < 0.1 and d["rect"].height < 0.01]
        xt = sorted(d["rect"].x0 for d in dr if d["type"] == "fs" and abs(d["rect"].y0 - ax.y1) < 0.1
                    and d["rect"].width < 0.01 and ax.x0 < d["rect"].x0 < ax.x1)
        assert len(xt) == len(N), (nm, xt)
        lab = [w for w in words if ax.x0 - 20 < w[2] < ax.x0 and ax.y0 < (w[1] + w[3]) / 2 < ax.y1]
        pairs = [(y, float(min(lab, key=lambda w: abs((w[1] + w[3]) / 2 - y))[4])) for y in sorted(yt)]
        a, b = np.polyfit([p[0] for p in pairs], [p[1] for p in pairs], 1)
        resid = float(np.max(np.abs(np.polyval([a, b], [p[0] for p in pairs]) - [p[1] for p in pairs])))
        pix = pg.get_pixmap(dpi=DPI, clip=pymupdf.Rect(ax.x0, ax.y0, ax.x1, ax.y1))
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].astype(int)
        vals = []
        for x in xt:
            px = int(round((x - ax.x0) * s))
            sat = (img[:, px - 3:px + 4, :].max(2) - img[:, px - 3:px + 4, :].min(2)) > 60
            top = int(np.where(sat.all(1))[0].min())
            vals.append(float(a * (ax.y0 + top / s) + b))
        out["panels"][nm] = dict(expected_iqm=vals, units_per_px=float(abs(a) / s), tick_fit_residual=resid, ticks=pairs)
        print(f"[INFO] A.17 {nm:9s} E[IQM_N] {np.round(vals, 5).tolist()}  read precision {abs(a) / s:.1e}/px")
    path = os.path.join(REPO, "runs", "full", "paper_a17.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("[INFO] wrote", path)


if __name__ == "__main__":
    main()
