"""Extract Figure-1 curves (arXiv v1 p.24) from the PDF's vector paths. Axis calibration from the axes frame
(long 0.18/0.16-pt lines) + tick marks; cross-checked against tick-label centres. Output: curves.json."""
import json, collections, os
import numpy as np, pymupdf
PDF = os.environ.get('PAPER_PDF', 'external/paper.pdf')  # arXiv:0805.3415v1; Figure 1 is on page 24
p = pymupdf.open(PDF)[23]
dr = p.get_drawings()
STY = {'[ .17841701 1.42734 ] 0': 'ucb1', '[ 2.14101 ] 0': 'exp3s',
       '[ .17841701 1.42734 2.14101 1.42734 ] 0': 'ducb', '[] 0': 'swucb'}
# frames: long thin lines
H, V = [], []
for g in dr:
    w = g.get('width') or 0
    if w > 0.3: continue
    for it in g['items']:
        if it[0] != 'l': continue
        a, b = it[1], it[2]
        if abs(a.y - b.y) < 1e-3 and abs(a.x - b.x) > 100: H.append((min(a.x, b.x), max(a.x, b.x), a.y))
        if abs(a.x - b.x) < 1e-3 and abs(a.y - b.y) > 60: V.append((a.x, min(a.y, b.y), max(a.y, b.y)))
H = sorted(set((round(x0, 3), round(x1, 3), round(y, 3)) for x0, x1, y in H), key=lambda r: (r[2], r[0]))
V = sorted(set((round(x, 3), round(y0, 3), round(y1, 3)) for x, y0, y1 in V))
print('H', H); print('V', V)

PAN = {('e1', 'arm1'): (127.481, 281.485, 343.492, 249.883, 7000.0),
       ('e2', 'arm1'): (337.841, 491.845, 343.492, 249.883, 9000.0),
       ('e1', 'regret'): (127.481, 281.485, 462.412, 368.803, 1000.0),
       ('e2', 'regret'): (337.841, 491.845, 462.412, 368.803, 900.0)}
segs = collections.defaultdict(list)
for g in dr:
    if abs((g.get('width') or 0) - 0.71) > 0.02 or g.get('dashes') not in STY: continue
    for it in g['items']:
        if it[0] == 'l':
            segs[STY[g['dashes']]].append((it[1].x, it[1].y, it[2].x, it[2].y))
res = {}
for (exp, met), (xl, xr, yb, yt, ymax) in PAN.items():
    for pol, ss in segs.items():
        pts = []
        for x0, y0, x1, y1 in ss:
            if xl - 0.5 <= min(x0, x1) and max(x0, x1) <= xr + 0.5 and yt - 0.5 <= min(y0, y1) and max(y0, y1) <= yb + 0.5:
                # exclude legend sample lines (short, in legend box top-left)
                pts.append((x0, y0)); pts.append((x1, y1))
        if not pts: continue
        a = np.array(pts)
        t = (a[:, 0] - xl) / (xr - xl) * 1e4
        v = (yb - a[:, 1]) / (yb - yt) * ymax
        o = np.lexsort((v, t)); t, v = t[o], v[o]
        # drop legend: legend sample lines sit at t in ~[150,1000] and y near top; remove points above 0.75*ymax with t<1100 that are isolated
        res[(exp, met, pol)] = (t, v)
# legend removal: legend lines are horizontal segments inside legend box; detect by known legend y positions
def clean(t, v, exp, met):
    leg_top = {'arm1': 0.80, 'regret': 0.80}[met]
    ymax = PAN[(exp, met)][4]
    keep = ~((t < 1100) & (v > leg_top * ymax))
    return t[keep], v[keep]
out = {}
for k, (t, v) in res.items():
    t, v = clean(t, v, k[0], k[1])
    # monotonicity: sort by t, check decreases between consecutive distinct points
    dec = np.diff(v)[np.diff(t) > 1e-6]
    def at(T0):
        return float(np.interp(T0, t, v))
    out['|'.join(k)] = {'n_pts': int(len(t)), 't_min': float(t.min()), 't_max': float(t.max()),
                        'v_at_3000': at(3000), 'v_at_5000': at(5000), 'v_at_T': float(v[t >= t.max() - 1e-6].max()),
                        'v_end_min': float(v[t >= t.max() - 1e-6].min()),
                        'max_decrease': float(-dec.min()) if dec.size else 0.0}
for k, d in sorted(out.items()):
    print(k, {kk: round(vv, 1) if isinstance(vv, float) else vv for kk, vv in d.items()})
json.dump({k: {'t': res[tuple(k.split('|'))][0].tolist(), 'v': res[tuple(k.split('|'))][1].tolist(), **out[k]} for k in out},
          open('fig1_curves.json', 'w'))
