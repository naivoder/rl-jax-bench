"""Exact read-offs of paper Figures 3 and 6 from the PDF's vector paths.

The figures are matplotlib vector graphics: each curve is a polyline and each standard-error band a filled polygon.
Tick marks are short black lines at known data values (y: labelled ticks; x: the first/last ticks at the panel edges
are 0 and the panel's training budget, whose labels are left/right-aligned to the edges). This replaces the 300-dpi
visual read-off (±4 % of the panel range) with the plotted coordinates themselves.
Per curve it reports the mean over the paper's last 10 % of training (the same window as our E4 metric), the mean SE
half-band there, and the last plotted point. Recheck: the plotted points are means over 30 equal bins of the
training budget, so the last 10 % is exactly the last 3 points (`last10_mean`); the review-stage rule (points with
x >= 0.9 * x_max) picked a 4th point on curves whose bin-27 mean episode end jittered past 0.9 * x_max and is kept as
`last10_xrule_mean` (original values: runs/review/figure_readoff_xrule.json). R2 check: `band_center_offset_max` =
max |band midpoint - line y| over the points (in data units) -- ~0 means each band is centered on its paired line.
Usage: uv run python scripts/review/figure_vector_readoff.py  -> runs/review/figure_readoff.json
"""
import json
import os

import numpy as np
import pymupdf

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.environ.get("PAPER_PDF", os.path.join(R, "external", "2018-time-limits-in-reinforcement-learning.pdf"))  # arXiv:1712.00378v4
F3_BUDGET = {"Ant": 3e6, "HalfCheetah": 2e6, "Hopper": 2e6, "Humanoid": 5e6, "InvertedDoublePendulum": 4e5,
             "InvertedPendulum": 1e5, "Reacher": 2e5, "Swimmer": 1e6, "Walker2d": 2e6}   # Figure 3 x-axis labels
F6_BUDGET = {"Hopper": 3e6, "Walker2d": 3e6, "InfiniteCubePusher": 3e6}                  # Figure 6 x-axis labels
WHITE, BLACK = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)


def polyline(d):
    P = []
    for it in d["items"]:
        if it[0] == "l":
            if not P:
                P.append(tuple(it[1]))
            P.append(tuple(it[2]))
    return np.array(P)


def num(s):
    try:
        return float(s.replace("−", "-"))
    except ValueError:
        return None


def panels(page):
    dr = page.get_drawings()
    return dr, [(i, d["rect"]) for i, d in enumerate(dr) if d.get("fill") == WHITE and d["items"]
                and d["items"][0][0] == "re" and 60 < d["rect"].width < 100]


def yticks(dr, words, R_):
    out = []
    for d in dr:
        if d.get("color") == BLACK and len(d["items"]) == 1 and d["items"][0][0] == "l":
            a, b = d["items"][0][1], d["items"][0][2]
            if abs(a.y - b.y) < 1e-3 and abs(max(a.x, b.x) - R_.x0) < 0.05 and abs(a.x - b.x) < 4 \
                    and R_.y0 - 0.5 <= a.y <= R_.y1 + 0.5:
                lab = [w for w in words if R_.x0 - 30 < w[2] < R_.x0 and abs((w[1] + w[3]) / 2 - a.y) < 3.5
                       and num(w[4]) is not None]
                if lab:
                    out.append((a.y, num(lab[0][4])))
    return sorted(set(out))


def curves_after(dr, pi, n_curves):
    """(band index, line index) pairs following panel pi in drawing order: a filled band then its line."""
    order, j = [], pi + 1
    while j < len(dr) and len(order) < 2 * n_curves:
        d = dr[j]
        if d.get("fill") == WHITE and d["items"] and d["items"][0][0] == "re":
            break  # next panel
        if d.get("color") is None and d.get("fill") not in (None, WHITE, BLACK):
            order.append(("band", j))
        elif d.get("color") not in (None, BLACK) and (d.get("width") or 0) > 0.6 and len(d["items"]) > 3:
            order.append(("line", j))
        j += 1
    return [(b[1], l[1]) for b, l in zip(order[0::2], order[1::2])]


def summarise(dr, band, line, fx, fy, xb):
    L = polyline(dr[line])
    X, Y = fx(L[:, 0]), fy(L[:, 1])
    B = polyline(dr[band])
    bx, by = fx(B[:, 0]), fy(B[:, 1])
    se = []
    for x in X:
        m = np.abs(bx - x) < 1e-3 * xb
        se.append((by[m].max() - by[m].min()) / 2 if m.sum() >= 2 else np.nan)
    se = np.array(se)
    mid = []
    for x in X:
        m = np.abs(bx - x) < 1e-3 * xb
        mid.append((by[m].max() + by[m].min()) / 2 if m.sum() >= 2 else np.nan)
    off = np.abs(np.array(mid) - Y)
    wx = X >= 0.9 * X.max()                                    # review-stage rule (superseded)
    w = np.zeros(len(X), bool)
    w[-3:] = True if len(X) == 30 else False                  # 30 equal bins: the last 10 % = the last 3 points
    if len(X) != 30:
        w = wx
    return dict(last10_mean=float(Y[w].mean()), last10_se=float(np.nanmean(se[w])) if np.isfinite(se[w]).any() else None,
                n_pts_window=int(w.sum()), last10_xrule_mean=float(Y[wx].mean()), n_pts_xrule=int(wx.sum()),
                band_center_offset_max=float(np.nanmax(off)) if np.isfinite(off).any() else None,
                n_pts=len(X), last_point=float(Y[-1]), x_max=float(X.max()),
                dashed="[]" not in str(dr[line].get("dashes")), color=[round(c, 3) for c in dr[line]["color"]],
                x=X.round(0).tolist(), y=Y.round(2).tolist(), se=np.round(se, 2).tolist())


def figure(page_no, budget, names_by_color):
    page = pymupdf.open(PDF)[page_no]
    dr, P = panels(page)
    words = page.get_text("words")
    out = {}
    for k, (pi, R_) in enumerate(P):
        title = [w[4] for w in words if R_.x0 < (w[0] + w[2]) / 2 < R_.x1 and R_.y0 - 15 < w[1] < R_.y0
                 and w[4].replace("ﬁ", "fi") in budget][0].replace("ﬁ", "fi")
        yt = yticks(dr, words, R_)
        (y0p, y0v), (y1p, y1v) = yt[0], yt[-1]
        xb = budget[title]
        fx = lambda x, R_=R_, xb=xb: (x - R_.x0) / (R_.x1 - R_.x0) * xb  # noqa: E731
        fy = lambda y, a=y0p, av=y0v, b=y1p, bv=y1v: av + (y - a) / (b - a) * (bv - av)  # noqa: E731
        cs = {}
        for band, line in curves_after(dr, pi, 5):
            s = summarise(dr, band, line, fx, fy, xb)
            # legend color nearest to the curve color (the PDF stores slightly different RGB for curve and legend)
            name = min(names_by_color.items(), key=lambda kv: np.abs(np.subtract(kv[0][0], s["color"])).sum()
                       + 10 * (kv[0][1] != s["dashed"]))[1]
            cs[name] = s
        out[f"{title}|panel{k}"] = dict(title=title, yticks=yt, budget=xb, curves=cs)
    return out


if __name__ == "__main__":
    BLUE, ORANGE = (0.122, 0.467, 0.706), (1.0, 0.498, 0.055)
    f3 = figure(4, F3_BUDGET, {(BLUE, False): "TA-PPO", (BLUE, True): "TA-PPO 10 workers",
                               (ORANGE, False): "PPO", (ORANGE, True): "PPO 10 workers"})
    # panels 0-8 = Figure 3a (gamma 0.99), 9-17 = Figure 3b (gamma 1), in drawing order
    f3 = {(k.split("|")[0] + ("|g0.99" if int(k.split("panel")[1]) < 9 else "|g1")): v for k, v in f3.items()}
    f6 = figure(6, F6_BUDGET, {((1.0, 0.671, 0.29), False): "PPO (T=50)", ((0.988, 0.365, 0.18), False): "PPO (T=300)",
                               ((0.831, 0.063, 0.129), False): "PPO (T=1000)",
                               ((0.475, 0.773, 0.475), False): "PEB-PPO (T=50)",
                               ((0.082, 0.475, 0.243), False): "PEB-PPO (T=300)"})
    os.makedirs(os.path.join(R, "runs", "review"), exist_ok=True)
    json.dump(dict(figure3=f3, figure6=f6), open(os.path.join(R, "runs", "review", "figure_readoff.json"), "w"))
    for fig, d in (("Figure 3", f3), ("Figure 6", f6)):
        print(f"\n## {fig}: last 10 % of plotted training (mean over points; ± mean SE half-band), last point")
        for k, v in d.items():
            row = "; ".join(f"{n}: {c['last10_mean']:.1f} ± {c['last10_se'] or float('nan'):.1f} (last {c['last_point']:.1f})"
                            for n, c in sorted(v["curves"].items()))
            print(f"{k:30s} {row}")
