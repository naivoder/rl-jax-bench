"""Exact curve data from the vector figures shipped in the arXiv TeX source of 2312.07858v1 (figs/*.pdf).
The paper's plots are vector PDFs, so the polyline vertices are recovered EXACTLY (no eyeballing): axes are calibrated from the tick marks
(short axis-aligned 0.5 pt lines on the left/bottom frame) paired with the tick-label words; curves are the 1.5 pt coloured polylines.
Usage: extract_fig_vectors.py <src_dir_with_figs> <out.json>      (needs pymupdf: uv sync --extra pdf; fixture tool, not part of the reproduction pipeline)
"""
import json, os, sys
import pymupdf

COL = {(0.0, 0.0, 1.0): "blue", (1.0, 0.0, 1.0): "magenta", (0.0, 1.0, 0.0): "green", (1.0, 0.0, 0.0): "red", (0.0, 0.0, 0.0): "black",
       (0.0, 1.0, 1.0): "cyan", (1.0, 1.0, 0.0): "yellow"}


def axis_fit(ticks, words):
    """ticks: list of coordinate positions; words: list of (center, float value). Pair each tick with the nearest numeric label; least squares line."""
    pairs = []
    for t in ticks:
        w = min(words, key=lambda w: abs(w[0] - t))
        if abs(w[0] - t) < 4.0:
            pairs.append((t, w[1]))
    pairs = sorted(set(pairs))
    if len(pairs) < 2:
        return None
    n = len(pairs); sx = sum(p[0] for p in pairs); sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] ** 2 for p in pairs); sxy = sum(p[0] * p[1] for p in pairs)
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx); b = (sy - a * sx) / n
    resid = max(abs(a * p[0] + b - p[1]) for p in pairs)
    return a, b, resid, pairs


def num(s):
    try:
        return float(s.replace("−", "-"))
    except ValueError:
        return None


def extract(path):
    p = pymupdf.open(path)[0]
    dr = p.get_drawings()
    words = [(w[:4], w[4]) for w in p.get_text("words")]
    lines = [x for x in dr if x.get("color") is not None and x.get("fill") is None and len(x["items"]) == 1 and x["items"][0][0] == "l"
             and abs((x.get("width") or 0) - 0.5) < 0.1]
    # frame = the long lines; ticks = short lines (<= 5 pt)
    xt, yt = [], []
    frame = [x for x in lines if max(x["rect"].width, x["rect"].height) > 20]
    x0 = min((f["rect"].x0 for f in frame), default=None); y1 = max((f["rect"].y1 for f in frame), default=None)
    for x in lines:
        r = x["rect"]
        if r.height < 5 and r.width < 0.01 and abs(r.y1 - y1) < 0.5:
            xt.append(r.x0)                       # bottom-axis tick (vertical short line touching the bottom frame)
        if r.width < 5 and r.height < 0.01 and abs(r.x0 - x0) < 0.5:
            yt.append(r.y0)                       # left-axis tick
    xw = [((b[0] + b[2]) / 2, num(t)) for b, t in words if num(t) is not None and abs((b[1] + b[3]) / 2 - (y1 + 15)) < 25]
    yw = [((b[1] + b[3]) / 2, num(t)) for b, t in words if num(t) is not None and (b[0] + b[2]) / 2 < x0]
    xw = [w for w in xw if w[1] is not None]; yw = [w for w in yw if w[1] is not None]
    fx, fy = axis_fit(sorted(set(round(v, 2) for v in xt)), xw), axis_fit(sorted(set(round(v, 2) for v in yt)), yw)
    curves = []
    for x in dr:
        if x.get("color") is not None and x.get("fill") is None and (x.get("width") or 0) > 1.0 and all(i[0] == "l" for i in x["items"]) and len(x["items"]) > 1:
            pts = [x["items"][0][1]] + [i[2] for i in x["items"]]
            pts = [(pt.x, pt.y) for pt in pts]
            col = COL.get(tuple(round(c, 3) for c in x["color"]), str(x["color"]))
            curves.append({"color": col, "dashes": x.get("dashes"), "width": x.get("width"),
                           "x": [fx[0] * a + fx[1] for a, _ in pts] if fx else None, "y": [fy[0] * b + fy[1] for _, b in pts] if fy else None})
    # markers (small filled shapes / curve items) are not decoded here; the polyline vertices carry the data
    return {"axis_fit_resid": [fx[2] if fx else None, fy[2] if fy else None], "x_ticks": fx[3] if fx else None, "y_ticks": fy[3] if fy else None, "curves": curves,
            "n_other_drawings": len(dr)}


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    res = {}
    for root, _, fs in os.walk(src):
        for f in sorted(fs):
            if f.endswith(".pdf"):
                rel = os.path.relpath(os.path.join(root, f), src)
                try:
                    res[rel] = extract(os.path.join(root, f))
                except Exception as e:  # noqa: BLE001
                    res[rel] = {"error": repr(e)}
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res.items():
        if "error" in v:
            print(k, "ERROR", v["error"]); continue
        print(k, "axis resid", v["axis_fit_resid"], "curves", [(c["color"], len(c["y"] or [])) for c in v["curves"]])
