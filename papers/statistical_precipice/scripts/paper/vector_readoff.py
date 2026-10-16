"""Exact read-off of the paper's interval plots from the PDF's VECTOR drawings (no pixels).

The figures in arXiv 2108.13264v1 are embedded as vector paths: every axes is a white filled rectangle, every tick
has a grey vertical gridline, every interval is a filled rectangle and every point estimate a short black vertical
line. This script maps x positions to data values with a least-squares fit of gridline x -> tick label value (the
residual is reported; it is ~1e-4 pt), assigns bars to rows by the row labels' y centres, and writes JSON.

    python scripts/paper/vector_readoff.py   -> data/paper_readoff/readoff.json

A bar whose end touches the axes edge is flagged ``at_edge_lo`` / ``at_edge_hi`` (INFORMATIONAL ONLY). It is NOT
clipped: rliable's plot_interval_estimates draws each interval with ``ax.barh(left=lower, ...)`` and matplotlib makes a
bar's ``left`` a sticky edge, so autoscaling stops exactly at the smallest lower end (verified in the oracle venv,
matplotlib 3.11.2: barh(left=[0.5, 0.62]) -> xlim (0.5, ...)). The edge value IS the paper's lower end and is
compared like any other (M2; the earlier '<=' clipped convention was wrong). Each row also carries the
panel's gridline-fit residual ``tol`` (data units), the read precision used as the comparison tolerance.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pymupdf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.environ.get("PAPER_PDF", os.path.join(REPO, "external", "data", "2021-deep-rl-at-the-edge-of-the-statistical-precipice.pdf"))
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _lines(words):
    """Group words into text lines by (block, line) -> (text, x0, y0, x1, y1)."""
    g = {}
    for w in words:
        g.setdefault((w[5], w[6]), []).append(w)
    out = []
    for ws in g.values():
        ws.sort(key=lambda w: w[0])
        out.append((" ".join(w[4] for w in ws), min(w[0] for w in ws), min(w[1] for w in ws),
                    max(w[2] for w in ws), max(w[3] for w in ws)))
    return out


def panels(page_no: int, y_lo: float, y_hi: float, min_w: float = 30.0):
    pg = pymupdf.open(PDF)[page_no - 1]
    dr = pg.get_drawings()
    words = pg.get_text("words")
    axes = [d["rect"] for d in dr if d["type"] == "f" and d.get("fill") == (1.0, 1.0, 1.0)
            and y_lo <= d["rect"].y0 <= y_hi and d["rect"].width > min_w and d["rect"].height > 20]
    axes.sort(key=lambda r: (round(r.y0), r.x0))
    lines = _lines([w for w in words if y_lo - 20 <= w[1] <= y_hi + 40])
    res = []
    for ax in axes:
        grid = sorted({round(d["rect"].x0, 3) for d in dr if d["type"] == "s" and d["rect"].width < 0.01
                       and abs(d["rect"].y0 - ax.y0) < 0.5 and abs(d["rect"].y1 - ax.y1) < 0.5
                       and ax.x0 - 0.5 <= d["rect"].x0 <= ax.x1 + 0.5
                       and d.get("color") and min(d["color"]) > 0.5})
        ticks = [w for w in words if ax.y1 < w[1] < ax.y1 + 10 and NUM.match(w[4]) and ax.x0 - 8 < (w[0] + w[2]) / 2 < ax.x1 + 8]
        pairs = []
        for gx in grid:
            near = [w for w in ticks if abs((w[0] + w[2]) / 2 - gx) < 3.0]
            if near:
                pairs.append((gx, float(near[0][4])))
        if len(pairs) < 2:
            continue
        gx, gv = np.array(pairs).T
        slope, icpt = np.polyfit(gx, gv, 1)
        resid = float(np.max(np.abs(slope * gx + icpt - gv)))
        f = lambda x: float(slope * x + icpt)  # noqa: E731
        title = [t for t in lines if ax.y0 - 12 < t[2] < ax.y0 and ax.x0 - 5 < (t[1] + t[3]) / 2 < ax.x1 + 5]
        bars = [d for d in dr if d["type"] in ("fs", "f") and d.get("fill") != (1.0, 1.0, 1.0)
                and ax.x0 - 0.5 <= d["rect"].x0 and d["rect"].x1 <= ax.x1 + 0.5 and ax.y0 <= d["rect"].y0 and d["rect"].y1 <= ax.y1]
        ticksb = [d for d in dr if d["type"] == "s" and d.get("color") == (0.0, 0.0, 0.0) and d["rect"].width < 0.01
                  and ax.x0 <= d["rect"].x0 <= ax.x1 and ax.y0 <= d["rect"].y0 and d["rect"].y1 <= ax.y1]
        # row labels: text lines left of the axes row (any column), matched by y centre
        rowlab = [t for t in lines if t[3] < ax.x0 + 1 and ax.y0 <= (t[2] + t[4]) / 2 <= ax.y1 and not NUM.match(t[0])]
        rows = []
        for b in bars:
            r = b["rect"]
            yc = (r.y0 + r.y1) / 2
            lab = min(rowlab, key=lambda t: abs((t[2] + t[4]) / 2 - yc)) if rowlab else None
            pt = [d["rect"].x0 for d in ticksb if r.y0 - 0.6 <= d["rect"].y0 and d["rect"].y1 <= r.y1 + 0.6]
            # recheck : an interval bar carries at most ONE black point line, and it lies inside the bar
            assert len(pt) <= 1 and all(r.x0 - 0.05 <= x <= r.x1 + 0.05 for x in pt), (page_no, r, pt)
            rows.append(dict(label=lab[0] if lab else None, label_dy=abs((lab[2] + lab[4]) / 2 - yc) if lab else None,
                             lo=f(r.x0), hi=f(r.x1), point=f(pt[0]) if pt else None,
                             at_edge_lo=bool(r.x0 <= ax.x0 + 0.05), at_edge_hi=bool(r.x1 >= ax.x1 - 0.05), tol=resid,
                             y=yc, fill=b.get("fill")))
        rows.sort(key=lambda d: d["y"])
        res.append(dict(page=page_no, title=title[0][0] if title else None, axes=[ax.x0, ax.y0, ax.x1, ax.y1],
                        grid=pairs, fit_residual=resid, units_per_pt=float(slope), rows=rows))
    return res


def fig6(page_no: int = 5):
    """recheck: Fig 6 (page 5) coverage (top row) and CI-width (bottom row) polylines, one 6-point
    line per method, read exactly. y is mapped by a least-squares fit of the horizontal gridlines to their tick labels;
    the colour -> method map is taken from the figure's own legend (marker fill colour next to each legend word). The
    x positions are the six N = 3, 5, 10, 20, 50, 100 of the axis labels (left to right)."""
    pg = pymupdf.open(PDF)[page_no - 1]
    dr = pg.get_drawings()
    words = pg.get_text("words")
    key = lambda c: tuple(round(v, 3) for v in c)  # noqa: E731
    legend = {}
    for w in words:
        if w[4] in ("basic", "bc", "bca", "percentile") and 100 < w[1] < 150:
            yc = (w[1] + w[3]) / 2
            mk = [d for d in dr if d["type"] == "fs" and d["rect"].x1 <= w[0] and w[0] - d["rect"].x1 < 12
                  and abs((d["rect"].y0 + d["rect"].y1) / 2 - yc) < 2.5]
            assert len(mk) == 1, (w, mk)
            legend[key(mk[0]["fill"])] = w[4]
    assert sorted(legend.values()) == ["basic", "bc", "bca", "percentile"], legend
    axes = [d["rect"] for d in dr if d["type"] == "f" and d.get("fill") == (1.0, 1.0, 1.0)
            and 340 <= d["rect"].x0 <= 490 and 55 <= d["rect"].y0 <= 160]
    axes.sort(key=lambda r: (round(r.y0), r.x0))
    assert len(axes) == 4, axes
    out = {}
    for ax, (stat, quantity) in zip(axes, [("median", "coverage"), ("iqm", "coverage"), ("median", "width"), ("iqm", "width")]):
        title = " ".join(w[4] for w in sorted([w for w in words if ax.x0 - 2 < w[0] < ax.x1 and ax.y0 - 12 < w[1] < ax.y0 and w[4] != "95%"], key=lambda w: w[0]))
        if quantity == "coverage":
            assert stat.lower() in title.lower(), (stat, title)
        ticks = []
        for w in words:
            if ax.x0 - 15 < w[2] <= ax.x0 + 1 and ax.y0 - 5 <= (w[1] + w[3]) / 2 <= ax.y1 + 5 and NUM.match(w[4]):
                ticks.append(((w[1] + w[3]) / 2, float(w[4])))
        grid = sorted({round(d["rect"].y0, 3) for d in dr if d["type"] == "s" and d["rect"].height < 0.01 and d.get("color")
                       and abs(d["color"][0] - 0.8) < 0.01 and abs(d["rect"].x0 - ax.x0) < 0.6 and abs(d["rect"].x1 - ax.x1) < 0.6})
        pairs = [(gy, min(ticks, key=lambda t: abs(t[0] - gy))[1]) for gy in grid if any(abs(t[0] - gy) < 3 for t in ticks)]
        gy, gv = np.array(pairs).T
        slope, icpt = np.polyfit(gy, gv, 1)
        resid = float(np.max(np.abs(slope * gy + icpt - gv)))
        rec = dict(axes=[ax.x0, ax.y0, ax.x1, ax.y1], grid=pairs, fit_residual=resid, N=[3, 5, 10, 20, 50, 100])
        lines = [d for d in dr if d["type"] == "s" and len(d["items"]) == 5 and d.get("color") and key(d["color"]) in legend
                 and d["rect"].x0 >= ax.x0 - 1 and d["rect"].x1 <= ax.x1 + 1 and d["rect"].y0 >= ax.y0 - 1 and d["rect"].y1 <= ax.y1 + 1]
        for L in lines:
            pts = [L["items"][0][1]] + [it[2] for it in L["items"]]
            assert all(pts[i].x < pts[i + 1].x for i in range(5)), "polyline not ordered in N"
            vals = [float(slope * p.y + icpt) for p in pts]
            rec[legend[key(L["color"])]] = [v / 100.0 for v in vals] if quantity == "coverage" else vals
        assert all(m in rec for m in legend.values()), (stat, quantity, list(rec))
        out.setdefault(stat, {})[quantity] = rec
    return out


FIGS = {  # page, y range (pt) of the axes tops
    "fig9": (8, 60, 140), "fig11a": (9, 0, 800), "a20": (22, 0, 800), "a27": (25, 60, 230), "a28": (25, 290, 310),
    "a29": (25, 600, 680)}


def main():
    out = {}
    for name, (p, lo, hi) in FIGS.items():
        out[name] = panels(p, lo, hi)
    out["fig6"] = fig6()   # recheck : coverage as a fraction, width in normalised-score units
    os.makedirs(os.path.join(REPO, "data", "paper_readoff"), exist_ok=True)
    path = os.path.join(REPO, "data", "paper_readoff", "readoff.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    for name, ps in out.items():
        if name == "fig6":
            print(f"[INFO] fig6: coverage and width polylines for {sorted(ps)}; max fit residual "
                  f"{max(q['fit_residual'] for st_ in ps.values() for q in st_.values()):.2e}")
            continue
        print(f"[INFO] {name}: {len(ps)} panels; titles {[q['title'] for q in ps]}; max fit residual "
              f"{max([q['fit_residual'] for q in ps], default=float('nan')):.2e}")


if __name__ == "__main__":
    main()
