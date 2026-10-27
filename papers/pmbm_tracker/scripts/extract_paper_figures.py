"""Extract the paper's Fig. 3 (truth trajectories) and Fig. 4 (per-step RMS OSPA curves) from the VECTOR graphics of the paper PDF.
Sec. VII says the experiments use 'the scenario in Figure 3'; both figures are vector plots,
so the data points are recoverable exactly (to ~0.02 m / 0.003 OSPA) from the PDF itself.  Source hierarchy tier 1 (paper).

  python scripts/extract_paper_figures.py [--pdf PATH]   ->  data/fig3_truth.npz, data/fig4_paper_curves.json

Axis calibration is read from the tick labels / gridlines of the PDF (not hard-coded):
  Fig. 3 (PDF page 12): x ticks 110..180 (label centres), y ticks 110..180 (label centres);
  Fig. 4 (PDF page 13): x ticks 10..80 (label centres), y gridlines at 0 and 10 (frame lines).
Fig. 3 markers: crosses at k = 1, circles every 5 steps (k = 6, 11, ...); their polyline indices must be k - 1 (a missing vertex shows up
as a gap != 5 between consecutive markers; the single red vertex the renderer dropped between k = 77 and k = 80 is filled by the midpoint
of the longest segment in that gap -- error << R = 1 m^2).
Fig. 4 curve identities are fixed by an assignment against the Table II (0.9, 10) row (RMS over 81 steps with k = 1 = 10), asserted < 0.01.
"""
import argparse, hashlib, json, os
import numpy as np
import pymupdf
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PDF = os.environ.get("PMBM_PAPER_PDF", "external/paper/1703.04264v4.pdf")   # arXiv 1703.04264v4 (https://arxiv.org/pdf/1703.04264v4)
TABLE2_09_10 = dict(est1=2.23, est2=2.34, est3=2.36, tomb=2.65, momb=2.37, cphd=3.39, phd=7.05)


def _lin(vals, pix):
    a, b = np.polyfit(pix, vals, 1)
    return lambda u: a * np.asarray(u) + b


def fig3(doc):
    p = doc[11]
    words = p.get_text("words")
    xt = sorted((float(w[4]), (w[0] + w[2]) / 2) for w in words if w[4] in {str(v) for v in range(110, 181, 10)} and abs(w[1] - 225.2) < 1.0)
    yt = sorted((float(w[4]), (w[1] + w[3]) / 2) for w in words if w[4] in {str(v) for v in range(110, 181, 10)} and abs(w[0] - 333.0) < 1.0)
    assert len(xt) == 8 and len(yt) == 8, (xt, yt)
    # exact tick positions = the vertical / horizontal grid lines of the axes; each tick label is snapped to its nearest grid line
    gx = sorted({d["rect"].x0 for d in p.get_drawings() if d["color"] is not None and d["rect"].width < 0.01 and d["rect"].height > 100 and 60 < d["rect"].y0 < 230 and d["rect"].x0 > 340})
    gy = sorted({d["rect"].y0 for d in p.get_drawings() if d["color"] is not None and d["rect"].height < 0.01 and d["rect"].width > 150 and d["rect"].x0 > 340 and 60 < d["rect"].y0 < 230})
    gx = [min(gx, key=lambda g: abs(g - u)) for _, u in xt]
    gy = [min(gy, key=lambda g: abs(g - u)) for _, u in yt]
    assert max(abs(g - u) for g, (_, u) in zip(gx, xt)) < 1.0 and max(abs(g - u) for g, (_, u) in zip(gy, yt)) < 1.5
    fx = _lin([v for v, _ in xt], gx)
    fy = _lin([v for v, _ in yt], gy)
    names = {(0.0, 0.0, 1.0): "blue", (1.0, 0.0, 0.0): "red", (0.0, 1.0, 0.0): "green", (0.0, 0.0, 0.0): "black"}
    poly, circ = {}, {n: [] for n in names.values()}
    for d in p.get_drawings():
        c = d.get("color")
        if c not in names:
            continue
        its = d["items"]
        if len(its) > 30 and all(it[0] == "l" for it in its):
            pts = [its[0][1]] + [it[2] for it in its]
            for a, b in zip(its[:-1], its[1:]):
                assert abs(a[2].x - b[1].x) < 1e-6 and abs(a[2].y - b[1].y) < 1e-6      # contiguous polyline
            poly[names[c]] = np.array([[float(fx(q.x)), float(fy(q.y))] for q in pts])
        elif len(its) == 4 and its[0][0] == "c":
            r = d["rect"]
            circ[names[c]].append([float(fx((r.x0 + r.x1) / 2)), float(fy((r.y0 + r.y1) / 2))])
    order = ["blue", "red", "green", "black"]
    tr, notes = [], {}
    for n in order:
        P = poly[n]
        C = np.array(circ[n])
        idx = sorted(int(np.argmin(np.linalg.norm(P - c, axis=1))) for c in C)
        err = max(float(np.min(np.linalg.norm(P - c, axis=1))) for c in C)
        assert err < 0.01, (n, err)                                    # circle marker sits on a polyline vertex
        want = list(range(0, 5 * len(idx), 5))                         # markers at k = 1, 6, 11, ... -> vertex index 0, 5, 10, ...
        gap = None
        if idx != want:
            # one dropped vertex: indices agree up to the gap and are shifted by one afterwards
            j = next(i for i, (a, b) in enumerate(zip(idx, want)) if a != b)
            assert all(a == b - 1 for a, b in zip(idx[j:], want[j:])), (n, idx)
            lo = idx[j - 1] if j else 0
            seg = np.linalg.norm(np.diff(P[lo:idx[j] + 1], axis=0), axis=1)
            m = lo + int(seg.argmax())
            P = np.vstack([P[:m + 1], (P[m] + P[m + 1]) / 2, P[m + 1:]])
            gap = dict(inserted_after_vertex=m, between_k=[lo + 1, idx[j] + 2])
            notes[n] = gap
        tr.append(P)
    K = 81
    pos = np.full((4, K, 2), np.nan)
    alive = np.zeros((4, K), bool)
    for i, P in enumerate(tr):
        pos[i, :len(P)] = P
        alive[i, :len(P)] = True
    assert alive.sum(1).tolist() == [40, 81, 81, 81], alive.sum(1)       # target 1 dies after k = 40 (t_death = 41)
    X = np.zeros((4, K, 4))
    for i in range(4):
        n = int(alive[i].sum())
        v = np.gradient(pos[i, :n], axis=0)
        X[i, :n, 0], X[i, :n, 2] = pos[i, :n, 0], pos[i, :n, 1]
        X[i, :n, 1], X[i, :n, 3] = v[:, 0], v[:, 1]
    return dict(X=X, alive=alive, positions=pos, notes=json.dumps(notes))


def fig4(doc):
    p = doc[12]
    words = p.get_text("words")
    xt = sorted((float(w[4]), (w[0] + w[2]) / 2) for w in words if w[4] in {str(v) for v in range(10, 81, 10)} and abs(w[1] - 403.3) < 1.0)
    assert len(xt) == 8
    fx = _lin([v for v, _ in xt], [u for _, u in xt])
    ys = sorted({round(d["rect"].y0, 2) for d in p.get_drawings() if d["rect"].height < 0.01 and d["rect"].width > 190 and 248 < d["rect"].y0 < 403})
    y0, y10 = max(ys), min(ys)                                          # gridlines y = 0 (bottom) and y = 10 (top)
    fy = lambda v: 10 * (y0 - v) / (y0 - y10)
    curves, phd = [], {}
    for d in p.get_drawings():
        c = d.get("color")
        its = d["items"]
        if c is None or not its or any(it[0] != "l" for it in its):
            continue
        if len(its) == 79 and abs(d["width"] - 0.78) < 1e-3:
            ks = [fx(its[0][1].x)] + [fx(it[2].x) for it in its]
            v = [fy(its[0][1].y)] + [fy(it[2].y) for it in its]
            curves.append(dict(k=np.round(ks).astype(int), v=np.array(v)))
        elif c == (1.0, 0.0, 1.0) and len(its) > 5:                      # PHD, drawn as two pieces around the gap at k = 41-42
            ks = [fx(its[0][1].x)] + [fx(it[2].x) for it in its]
            v = [fy(its[0][1].y)] + [fy(it[2].y) for it in its]
            for kk, vv in zip(np.round(ks).astype(int), v):
                phd[int(kk)] = float(vv)
    assert len(curves) == 6, len(curves)
    algos = ["est1", "est2", "est3", "tomb", "momb", "cphd"]
    rms = np.array([np.sqrt((100.0 + np.sum(c["v"] ** 2)) / 81) for c in curves])
    tgt = np.array([TABLE2_09_10[a] for a in algos])
    r, c_ = linear_sum_assignment(np.abs(rms[:, None] - tgt[None, :]))
    out = {}
    for ci, ai in zip(r, c_):
        assert abs(rms[ci] - tgt[ai]) < 0.01, (algos[ai], rms[ci])
        assert curves[ci]["k"].tolist() == list(range(2, 82))
        out[algos[ai]] = {str(int(k)): float(v) for k, v in zip(curves[ci]["k"], curves[ci]["v"])}
    out["phd"] = {str(k): v for k, v in sorted(phd.items())}
    full = [phd.get(k, 10.0) for k in range(2, 82)]                      # PHD gap = OSPA 10 in every run
    assert abs(np.sqrt((100 + np.sum(np.square(full))) / 81) - 7.05) < 0.02
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF)
    a = ap.parse_args()
    doc = pymupdf.open(a.pdf)
    sha = hashlib.sha256(open(a.pdf, "rb").read()).hexdigest()
    f3 = fig3(doc)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    np.savez(os.path.join(ROOT, "data", "fig3_truth.npz"), X=f3["X"], alive=f3["alive"], positions=f3["positions"], notes=f3["notes"], pdf_sha256=sha)
    f4 = fig4(doc)
    json.dump(dict(pdf_sha256=sha, source="Fig. 4 vector paths, p_d = 0.9, lambda_c = 10; k = 1 is not plotted (OSPA = 10 for every algorithm)", curves=f4),
              open(os.path.join(ROOT, "data", "fig4_paper_curves.json"), "w"), indent=0)
    print("fig3 alive", f3["alive"].sum(1).tolist(), "notes", f3["notes"], "| fig4 curves", {k: len(v) for k, v in f4.items()}, "| pdf sha256", sha[:16])


if __name__ == "__main__":
    main()
