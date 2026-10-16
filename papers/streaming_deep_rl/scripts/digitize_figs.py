"""Digitize the paper's vector figures (Fig. 4 MinAtar, Fig. 7/8 ablations) from the PDF drawing paths.

Axis calibration: a least-squares fit of the tick MARK positions (the
short grey/black stroke at each tick, read from the drawing paths) to the numeric value of the tick label next to
each mark. Matplotlib draws the mark AT the data value; the label is anchored centre_baseline, so its bbox center is
offset from the mark by font metrics (0.690 pt in Fig. 4, 0.648 pt in Figs. 7/8). The pre-fix calibration on
label-bbox centers (CALIB = "labels", `--calib labels`) biased every y value upward by that constant. Curves are the
stroke paths of each series color (line width > the band outlines). Output: JSON {figure: {panel: {series: [[x, y]]}}}.
Run: python scripts/digitize_figs.py <out.json> [--calib marks|labels]  (needs pymupdf: `uv sync --extra pdf`)
"""
import json
import os
import re
import sys

import numpy as np
import pymupdf as fitz

# arXiv:2410.14606v2 (sha256 9215cd3e...aa24); scripts/official/setup.sh downloads it to the default location
PDF = os.environ.get("PAPER_PDF", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            "external", "data", "2410.14606v2.pdf"))
CALIB = "marks"   # "labels" = the pre-fix behavior (label-bbox centers), kept for before/after comparison


def tick_marks(dr):
    """Short grey/black axis-tick strokes: horizontal (y ticks) and vertical (x ticks), as (x0, x1, y) / (x, y0, y1)."""
    h, v = [], []
    for x in dr:
        c = x.get("color")
        if c is None or not (abs(c[0] - c[1]) < 0.02 and abs(c[1] - c[2]) < 0.02):
            continue
        for it in x["items"]:
            if it[0] != "l":
                continue
            a, b = it[1], it[2]
            if abs(a.y - b.y) < 1e-3 and 0.5 < abs(a.x - b.x) < 6:
                h.append((min(a.x, b.x), max(a.x, b.x), a.y))
            if abs(a.x - b.x) < 1e-3 and 0.5 < abs(a.y - b.y) < 6:
                v.append((a.x, min(a.y, b.y), max(a.y, b.y)))
    return h, v


def mark_for_label(w, axis, h, v):
    """Page coordinate of the tick mark belonging to tick label w (y label: the horizontal stroke just right of the
    label at its height; x label: the vertical stroke just above it). Raises if there is none (never falls back)."""
    if axis == "y":
        yc = 0.5 * (w[1] + w[3])
        cand = [s for s in h if abs(s[2] - yc) < 2.0 and w[2] - 1.0 < s[0] < w[2] + 8.0]
        if not cand:
            raise RuntimeError(f"no y tick mark next to label {w[4]!r} at y={yc:.2f}")
        return min(cand, key=lambda s: abs(s[2] - yc))[2]
    xc = 0.5 * (w[0] + w[2])
    cand = [s for s in v if abs(s[0] - xc) < 2.0 and w[1] - 8.0 < s[2] <= w[1] + 1.0]
    if not cand:
        raise RuntimeError(f"no x tick mark above label {w[4]!r} at x={xc:.2f}")
    return min(cand, key=lambda s: abs(s[0] - xc))[0]


def col(c):
    return None if c is None else tuple(round(v, 2) for v in c)


def panels(page, ymin, ymax):
    """Group axes: black stroked rectangles/lines are spines; find axes boxes via tick labels instead."""
    words = [w for w in page.get_text("words") if ymin <= w[1] <= ymax]
    return words


def digitize(pageno, ymin, ymax, names):
    d = fitz.open(PDF)
    p = d[pageno - 1]
    dr = [x for x in p.get_drawings() if ymin <= x["rect"].y0 and x["rect"].y1 <= ymax]
    words = [w for w in p.get_text("words") if ymin <= w[1] and w[3] <= ymax]
    num = re.compile(r"^[−-]?\d+(\.\d+)?$")
    hm, vm = tick_marks(dr)
    # spines: grey/black axes; use the stroke paths of series colors ('s' type, no fill)
    series = {}
    for x in dr:
        if x["type"] != "s" or x.get("fill") is not None:
            continue
        c = col(x["color"])
        if c is None:
            continue
        if c in ((0.0, 0.0, 0.0),) or (c and abs(c[0] - c[1]) < 0.02 and abs(c[1] - c[2]) < 0.02):
            continue  # black/grey = spines / dashed reference
        pts = []
        for it in x["items"]:
            if it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        if len(pts) >= 2:
            series.setdefault((c, x.get("dashes"), round(x.get("width") or 0, 2)), []).append(np.array(pts))
    # axes boxes: cluster series paths by x-range
    boxes = []
    for c, lst in series.items():
        for a in lst:
            boxes.append((a[:, 0].min(), a[:, 0].max(), c, a))
    # tick labels
    nums = [(w, float(w[4].replace("−", "-"))) for w in words if num.match(w[4].replace("−", "-"))]
    out = {}
    xs = sorted({round(b[0]) for b in boxes})
    groups = []
    for b in sorted(boxes, key=lambda b: (b[0], b[1])):
        for g in groups:
            if abs(g[0] - b[0]) < 15 and abs(np.mean([a[3][:, 1].mean() for a in g[1]]) - b[3][:, 1].mean()) < 120:
                g[1].append(b)
                break
        else:
            groups.append([b[0], [b]])
    for gi, (x0, bl) in enumerate(groups):
        xl = min(b[0] for b in bl)
        xr = max(b[1] for b in bl)
        yt = min(b[3][:, 1].min() for b in bl)
        yb = max(b[3][:, 1].max() for b in bl)
        # y tick labels: left of xl within 30pt, vertically within [yt-15, yb+15]
        ylw = [(w, v) for w, v in nums if xl - 35 < w[2] < xl + 2 and yt - 25 < w[1] < yb + 15]
        # x tick labels: below yb within 20pt, horizontally within [xl-10, xr+10]
        cand = [(w, v) for w, v in nums if yb - 2 < w[1] < yb + 22 and xl - 12 < w[0] < xr + 10]
        ymax_c = max(0.5 * (w[1] + w[3]) for w, _ in cand)
        xlw = [(w, v) for w, v in cand if abs(0.5 * (w[1] + w[3]) - ymax_c) < 1.0]
        ylw = [(w, v) for w, v in ylw if abs(0.5 * (w[1] + w[3]) - ymax_c) > 1.0]
        xlab = [(0.5 * (w[0] + w[2]), v) for w, v in xlw]
        yl = [(0.5 * (w[1] + w[3]), v) for w, v in ylw]
        off = None
        if CALIB == "marks":   # Replace each label center by its tick MARK position
            ylm = [(mark_for_label(w, "y", hm, vm), v) for w, v in ylw]
            xlm = [(mark_for_label(w, "x", hm, vm), v) for w, v in xlw]
            off = dict(y_label_minus_mark_pt=[round(c - m, 4) for (c, _), (m, _) in zip(yl, ylm)],
                       x_label_minus_mark_pt=[round(c - m, 4) for (c, _), (m, _) in zip(xlab, xlm)])
            yl, xlab = ylm, xlm
        elif CALIB != "labels":
            raise ValueError(CALIB)
        ax_, bx_ = np.polyfit([v for _, v in xlab], [c for c, _ in xlab], 1)
        ay_, by_ = np.polyfit([v for _, v in yl], [c for c, _ in yl], 1)
        res_x = max(abs(ax_ * v + bx_ - c) for c, v in xlab)
        res_y = max(abs(ay_ * v + by_ - c) for c, v in yl)
        ser = {}
        for b in bl:
            a = b[3]
            ser.setdefault(str(b[2]), []).append(np.stack([(a[:, 0] - bx_) / ax_, (a[:, 1] - by_) / ay_], 1).tolist())
        out[gi] = dict(bbox=[xl, yt, xr, yb], yticks=yl, xticks=xlab, fit_resid_pt=[res_x, res_y],
                       pt_per_unit=[ax_, ay_], intercept_pt=[bx_, by_], calib=CALIB, label_offsets=off, series=ser)
    return out


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1].isdigit():
    pg, y0, y1 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    r = digitize(pg, y0, y1, None)
    for k, v in r.items():
        print(k, [round(x) for x in v["bbox"]], "yt", v["yticks"], "xt", v["xticks"], list(v["series"].keys()))


# ---------------------------------------------------------------------------------------------------- export
def bands(pageno, ymin, ymax):
    """Filled CI polygons ('fs' with a series fill color), as point arrays in page coordinates."""
    p = fitz.open(PDF)[pageno - 1]
    out = []
    for x in p.get_drawings():
        if x["type"] != "fs" or not (ymin <= x["rect"].y0 and x["rect"].y1 <= ymax):
            continue
        c = col(x.get("fill"))
        if c is None or (abs(c[0] - c[1]) < 0.02 and abs(c[1] - c[2]) < 0.02):
            continue
        segs = [(it[1].x, it[1].y, it[2].x, it[2].y) for it in x["items"] if it[0] == "l"]
        if len(segs) > 20:
            out.append((c, np.array(segs)))
    return out


def band_at(segs, xq):
    ys = []
    for x0, y0, x1, y1 in segs:
        if min(x0, x1) <= xq <= max(x0, x1) and x1 != x0:
            ys.append(y0 + (y1 - y0) * (xq - x0) / (x1 - x0))
    return (min(ys), max(ys)) if len(ys) >= 2 else None


FIGS = {  # page, y-range (pt), panel names, series key -> name (Fig. 7 has two solid-red paths: max final = no_sparse)
    "fig4": (14, 60, 166, ["Asterix", "Breakout", "Seaquest", "SpaceInvaders"],
             {"(0.12, 0.47, 0.71)|[] 0": "stream_q", "(0.84, 0.15, 0.16)|[] 0": "classic_q",
              "(0.89, 0.47, 0.76)|[] 0": "dqn1"}),
    "fig7": (15, 495, 576, ["Ant", "HalfCheetah", "Hopper", "Walker2d"],
             {"(0.12, 0.47, 0.71)|[] 0": "stream_ac", "(0.84, 0.15, 0.16)|[ 2.117325 .91560009 ] 0": "no_obgd",
              "(0.84, 0.15, 0.16)|[ 3.6624003 .91560009 .57225 .91560009 ] 0": "no_norm",
              "(0.84, 0.15, 0.16)|[ .57225 .9442125 ] 0": "no_ln",
              "(0.84, 0.15, 0.16)|[] 0": ["classic_ac", "no_sparse"]}),
    "fig8": (16, 150, 222, ["Ant", "HalfCheetah", "Hopper", "Walker2d"],
             {"(0.12, 0.47, 0.71)|[] 0": "stream_ac", "(0.12, 0.47, 0.71)|[ 2.117325 .91560009 ] 0": "stream_ac0",
              "(0.84, 0.15, 0.16)|[] 0": "classic_ac", "(0.84, 0.15, 0.16)|[ 2.117325 .91560009 ] 0": "classic_ac0"}),
}


def export(path):
    import ast
    res = {}
    for fig, (pg, y0, y1, pnames, smap) in FIGS.items():
        r = digitize(pg, y0, y1, None)
        bl = bands(pg, y0, y1)
        res[fig] = {}
        for gi, v in sorted(r.items()):
            ax_, ay_ = v["pt_per_unit"]
            if CALIB == "labels":   # pre-fix: intercept from the first label only
                bx_ = v["xticks"][0][0] - ax_ * v["xticks"][0][1]
                by_ = v["yticks"][0][0] - ay_ * v["yticks"][0][1]
            else:                   # The least-squares intercept of the tick-mark fit
                bx_, by_ = v["intercept_pt"]
            panel = {}
            for key, lst in v["series"].items():
                c, dash, _w = ast.literal_eval(key)
                name = smap[f"{c}|{dash}"]
                lst = sorted(lst, key=lambda a: a[-1][1]) if isinstance(name, list) else lst
                names = name if isinstance(name, list) else [name]
                for nm, pts in zip(names, lst):
                    a = np.array(pts)
                    ent = dict(x_end=float(a[:, 0].max()), points=a.round(4).tolist(), band={})
                    for xq in (0.2, 0.5, 1.0, 1.5, 2.0, 5.0):
                        if xq > ent["x_end"] + 1e-6:
                            continue
                        yq = float(np.interp(xq, a[:, 0], a[:, 1]))
                        ent.setdefault("at", {})[str(xq)] = round(yq, 3)
                        xp, yp = ax_ * xq + bx_, ay_ * yq + by_
                        best = None
                        for bc, segs in bl:   # same-color polygon whose center is closest to the line point
                            if bc != tuple(c):
                                continue
                            b = band_at(segs, xp)
                            if b and b[0] - 0.3 <= yp <= b[1] + 0.3:
                                d_ = abs(0.5 * (b[0] + b[1]) - yp)
                                if best is None or d_ < best[0]:
                                    best = (d_, b)
                        if best is not None:
                            b = best[1]
                            lo, hi = sorted(((b[1] - by_) / ay_, (b[0] - by_) / ay_))
                            ent["band"][str(xq)] = [round(lo, 3), round(hi, 3)]
                    panel[nm] = ent
            res[fig][pnames[gi]] = panel
    with open(path, "w") as f:
        json.dump(dict(source=PDF, calib=CALIB, units="x in 1e6 steps; y = average episodic return; band = shaded "
                       "polygon at x (caption: 90% CI; official plot.py: avg +- 1 s.e.)",
                       figures=res), f, indent=1)
    return res


if __name__ == "__main__" and len(sys.argv) in (2, 4) and not sys.argv[1].isdigit():
    if len(sys.argv) == 4:
        assert sys.argv[2] == "--calib", sys.argv
        CALIB = sys.argv[3]
    rr = export(sys.argv[1])
