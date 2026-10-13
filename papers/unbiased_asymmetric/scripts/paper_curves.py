"""Extract the paper's Figure 3 (and Figure 4) learning curves EXACTLY from the PDF's vector graphics (no pixel reading).

Each panel's mean curve is a 100-segment polyline and its s.e.m. band a filled polygon (matplotlib). Colours (legend
above Fig. 3): C0 blue asym-hs, C1 orange asym-s, C2 green A2C, C3 red react-2, C4 purple react-4.

Axis calibration (recheck): the axis maps are fitted through the panel's own white GRIDLINES,
which matplotlib draws exactly at the tick positions; the tick labels only assign a value to each gridline (nearest
gridline within 1.5 pt). The pre-fix calibration fitted through the tick-LABEL centers, which sit ~0.03 pt off the
ticks (2.3 return units per pt on Shopping); it is kept as `--calibration labels` for before/after comparisons.

    python scripts/paper_curves.py                              -> src/configs/paper_fig3_curves.json (gridlines)
    python scripts/paper_curves.py --calibration labels --out src/configs/paper_fig3_curves_labelcal.json   (pre-fix)
    python scripts/paper_curves.py --fig4                       -> src/configs/paper_fig4_curves.json (HH-4 critic probes)
(needs pymupdf)"""
import argparse
import json
import os
import re

import numpy as np
import pymupdf

# the arXiv v2 PDF (2105.11674v2); set PAPER_PDF to its location
PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.environ.get("PAPER_PDF", os.path.join(PAPER, "external", "2105.11674v2.pdf"))
RUNS = os.path.join(PAPER, "src", "configs")
COL = {(0.12, 0.47, 0.71): "a2c_asym_hs", (1.0, 0.5, 0.05): "a2c_asym_s", (0.17, 0.63, 0.17): "a2c",
       (0.84, 0.15, 0.16): "a2c_react_2", (0.58, 0.4, 0.74): "a2c_react_4"}
# (env, x-range of the panel incl. y labels, plot y-range, x-tick label y-band)
PANELS = [("heavenhell_3", (55, 185), (95, 175), (172, 192)), ("heavenhell_4", (180, 310), (95, 175), (172, 192)),
          ("shopping_5", (305, 435), (95, 175), (172, 192)), ("shopping_6", (430, 560), (95, 175), (172, 192)),
          ("car_flag", (55, 185), (200, 282), (279, 299)), ("cleaner", (180, 310), (200, 282), (279, 299))]
# Figure 4 (HH-4 critic probes, same page): panel -> (x-range incl. y labels, plot y-range, x-tick label y-band)
FIG4 = {"a": ((45, 175), (375, 455), (446, 458)), "b": ((175, 310), (375, 455), (446, 458)),
        "c": ((45, 175), (480, 560), (551, 563)), "d": ((175, 310), (480, 560), (551, 563))}
FIG4_NAMES = {"a": "(a) map0 no-priest", "b": "(b) map1 no-priest", "c": "(c) map0 priest", "d": "(d) map1 priest"}


def num(s):
    s = s.replace("−", "-")
    return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def x_tick_labels(words, x0, x1, tb0, tb1):
    """x tick labels: number (+ unit word 'M'/'k' right after it) -> list of (label-center x, value)."""
    xw = sorted([w for w in words if x0 < w[0] < x1 and tb0 < w[1] < tb1 and w[4] != "Timestep"], key=lambda w: w[0])
    xt, i = [], 0
    while i < len(xw):
        v = num(xw[i][4])
        if v is None:
            i += 1
            continue
        left, right, mult = xw[i][0], xw[i][2], 1.0
        if i + 1 < len(xw) and xw[i + 1][4] in ("M", "k") and xw[i + 1][0] - xw[i][2] < 4:
            mult = 1e6 if xw[i + 1][4] == "M" else 1e3
            right = xw[i + 1][2]
            i += 1
        xt.append(((left + right) / 2, v * mult))
        i += 1
    return xt


def gridlines(drawings, x0, x1, y0, y1):
    """White stroked gridlines inside a panel: (sorted horizontal y positions, sorted vertical x positions)."""
    hs, vs = set(), set()
    for g in drawings:
        r = g["rect"]
        if not (x0 - 5 < r.x0 and r.x1 < x1 + 10 and y0 - 15 < r.y0 and r.y1 < y1 + 15):
            continue
        c = tuple(round(v, 2) for v in (g.get("color") or ()))
        if c != (1.0, 1.0, 1.0) or g["type"] != "s":
            continue
        for it in g["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.y - b.y) < 1e-3 and abs(a.x - b.x) > 20:
                    hs.add(round(a.y, 4))
                if abs(a.x - b.x) < 1e-3 and abs(a.y - b.y) > 20:
                    vs.add(round(a.x, 4))
    return sorted(hs), sorted(vs)


def calibrate(labels, lines, how, tol=1.5):
    """Linear map position -> value. how='gridlines': each label is moved to its nearest gridline (must be within
    `tol` pt, else the label is dropped; >= 2 matches required, never a silent fallback to the label fit)."""
    if how == "labels":
        pairs = list(labels)
    else:
        pairs = []
        for p, v in labels:
            if not lines:
                break
            j = int(np.argmin(np.abs(np.asarray(lines) - p)))
            if abs(lines[j] - p) <= tol:
                pairs.append((lines[j], v))
        if len({p for p, _ in pairs}) < 2:
            raise RuntimeError(f"gridline calibration matched {len(pairs)} of {len(labels)} labels")
    fit = np.polyfit([p for p, _ in pairs], [v for _, v in pairs], 1)
    resid = float(np.max(np.abs(np.polyval(fit, [p for p, _ in pairs]) - np.asarray([v for _, v in pairs]))))
    return fit, pairs, resid


def curves_in(drawings, x0, x1, y0, y1, X, Y, cols, clipped_bands=False):
    """clipped_bands: also accept band polygons that extend below the panel (drawn under a clip path; Fig. 4's
    asym-hs bands reach far below the -0.2 axis floor early in training)."""
    panel = {}
    for g in drawings:
        r = g["rect"]
        inside = x0 < r.x0 and r.x1 < x1 + 5 and y0 - 10 < r.y0 and r.y1 < y1 + 10
        if clipped_bands and g["type"] == "fs":
            inside = x0 < r.x0 and r.x1 < x1 + 5 and y0 - 10 < r.y0 < y1
        if not inside:
            continue
        c = tuple(round(v, 2) for v in (g.get("color") or ()))
        m = cols.get(c)
        if m is None:
            continue
        pts = []
        for it in g["items"]:
            if it[0] == "l":
                pts += [it[1], it[2]]
        pts = np.array([(p.x, p.y) for p in pts])
        if g["type"] == "s" and len(g["items"]) >= 50:      # mean curve polyline
            xs, idx = np.unique(pts[:, 0], return_index=True)
            panel.setdefault(m, {})["t"] = X(xs).tolist()
            panel[m]["mean"] = Y(pts[idx, 1]).tolist()
        elif g["type"] == "fs" and len(g["items"]) >= 100:  # s.e.m. band polygon
            segs = [(it[1], it[2]) for it in g["items"] if it[0] == "l"]
            panel.setdefault(m, {})["band_poly"] = [(float(X(a.x)), float(Y(a.y))) for a, _ in segs]
    return panel


def fig3(how):
    page = pymupdf.open(PDF)[7]
    words = page.get_text("words")
    drawings = page.get_drawings()
    out = {"_calibration": how}
    for env, (x0, x1), (y0, y1), (tb0, tb1) in PANELS:
        xt = x_tick_labels(words, x0, x1, tb0, tb1)
        # y ticks: numeric words left of the plot area
        yt = [((w[1] + w[3]) / 2, num(w[4])) for w in words
              if x0 < w[0] < x0 + 20 and y0 - 5 < w[1] < tb0 and num(w[4]) is not None]   # not the x-tick '0'
        if env.startswith("shopping"):
            # Shopping y labels are -150..0; the minus sign is a separate glyph path, not text (checked on the
            # 500-dpi render), so the extracted numbers are magnitudes
            yt = [(p, -abs(v)) for p, v in yt]
        hs, vs = gridlines(drawings, x0, x1, y0, y1)
        ax, xpairs, xres = calibrate(xt, vs, how)
        ay, ypairs, yres = calibrate(yt, hs, how)
        X = lambda px: np.polyval(ax, px)  # noqa: E731
        Y = lambda py: np.polyval(ay, py)  # noqa: E731
        panel = dict(x_ticks=xt, y_ticks=yt, x_calibration_points=xpairs, y_calibration_points=ypairs,
                     calibration_max_residual=dict(x=xres, y=yres))
        panel.update(curves_in(drawings, x0, x1, y0, y1, X, Y, COL))
        for m in COL.values():
            if m in panel and "t" in panel[m]:
                t = np.asarray(panel[m]["t"])
                y = np.asarray(panel[m]["mean"])
                T = t.max()
                band = np.asarray(panel[m].get("band_poly", []))
                half = None
                if len(band):
                    sel = band[band[:, 0] >= 0.95 * T]
                    half = float((sel[:, 1].max() - sel[:, 1].min()) / 2) if len(sel) else None
                panel[m]["final_last5pct"] = float(y[t >= 0.95 * T].mean())
                panel[m]["band_halfwidth_last5pct"] = half
        out[env] = panel
        print(env, {m: (round(panel[m]["final_last5pct"], 3), panel[m]["band_halfwidth_last5pct"] and
                        round(panel[m]["band_halfwidth_last5pct"], 3)) for m in COL.values() if m in panel},
              "max calib residual x/y", round(xres, 1), round(yres, 6))
    return out


def fig4(how):
    """Figure 4: HH-4 critic values at the four fork histories (a)-(d), mean over 20 runs per arm (+ band)."""
    page = pymupdf.open(PDF)[7]
    words = page.get_text("words")
    drawings = page.get_drawings()
    cols = {c: m for c, m in COL.items() if m in ("a2c_asym_hs", "a2c_asym_s", "a2c")}
    out = {"_calibration": how, "_names": FIG4_NAMES}
    for k, ((x0, x1), (y0, y1), (tb0, tb1)) in FIG4.items():
        xt = x_tick_labels(words, x0 + 15, x1, tb0, tb1)
        hs, vs = gridlines(drawings, x0 + 15, x1, y0, y1)
        # y labels: numeric words entirely left of the plot frame (vs[0]) and above the x-tick label band
        yt = [((w[1] + w[3]) / 2, num(w[4])) for w in words
              if x0 < w[0] and w[2] < vs[0] and y0 - 5 < w[1] < tb0 - 4 and num(w[4]) is not None]
        # the lowest y label is -0.2: its minus sign is a separate glyph path (labels run -0.2, 0.0, ..., 0.8)
        yt = sorted(yt, key=lambda q: -q[0])
        yt[0] = (yt[0][0], -abs(yt[0][1]))
        ax, xpairs, xres = calibrate(xt, vs, how)
        ay, ypairs, yres = calibrate(yt, hs, how)
        X = lambda px: np.polyval(ax, px)  # noqa: E731
        Y = lambda py: np.polyval(ay, py)  # noqa: E731
        panel = dict(x_ticks=xt, y_ticks=yt, x_calibration_points=xpairs, y_calibration_points=ypairs,
                     calibration_max_residual=dict(x=xres, y=yres))
        panel.update(curves_in(drawings, x0, x1, y0, y1 + 3, X, Y, cols, clipped_bands=True))
        out[k] = panel
        print(k, {m: (len(panel[m].get("t", [])), round(panel[m]["t"][0]), round(panel[m]["t"][-1]),
                      round(panel[m]["mean"][-1], 3), "band_poly" in panel[m]) for m in cols.values() if m in panel},
              "max calib residual x/y", round(xres, 1), round(yres, 6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", choices=["gridlines", "labels"], default="gridlines")
    ap.add_argument("--fig4", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.fig4:
        out, path = fig4(a.calibration), a.out or os.path.join(RUNS, "paper_fig4_curves.json")
    else:
        out, path = fig3(a.calibration), a.out or os.path.join(RUNS, "paper_fig3_curves.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
