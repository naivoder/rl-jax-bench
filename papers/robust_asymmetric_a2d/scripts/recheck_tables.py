"""Markdown tables of the recheck runs from runs/recheck/report.json (python scripts/recheck_tables.py > runs/recheck/tables.md)."""
import json, os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'sa', 'report.json')
r = json.load(open(R))
P, O = r['pairs'], r['official']
f2 = lambda v: f'{v:.2f}'
def ms(s): return f"{s['mean_se'][0]:.2f} ± {s['mean_se'][1]:.2f}"
def med(s): return f"{s['median']:.2f} [{s['iqr'][0]:.2f}, {s['iqr'][1]:.2f}]"
def ps(s): return f"{s['paper_stat']['median']:.2f} [{s['paper_stat']['q'][0]:.2f}, {s['paper_stat']['q'][1]:.2f}]"
print('#### Before / after (same seeds; final = per-seed mean of the last 5 evaluations)\n')
print('| cell | n | before: mean ± SE; median [IQR]; conv; fallback iters | after: mean ± SE; median [IQR]; conv; fallback iters | Δ mean ± SE (paired t p) | seeds changed > 0.01 | mean clamped samples / iteration (after) |')
print('|---|---|---|---|---|---|---|')
for k, p in P.items():
    b, a, d = p['before'], p['after'], p['before_after']
    ncl = a.get('n_clamped_mean')
    print(f"| {k} | {d['n']} | {ms(b)}; {med(b)}; {b.get('converged_end')}/{b['n']}; {b['fallback_iters_total']:.0f} | {ms(a)}; {med(a)}; {a.get('converged_end')}/{a['n']}; {a['fallback_iters_total']:.0f} | {d['delta_mean_se'][0]:+.2f} ± {d['delta_mean_se'][1]:.2f} ({d['t_p_two_sided']:.2g}) | {d['n_changed_gt_0.01']}/{d['n']} | {'–' if ncl is None else f'{ncl:.1f}'} |")
print('\n#### Paper statistic (per-x median [IQR] over seeds at the final x) vs the decoded paper figure at the same x, and the authors\' code\n')
print('| cell | x | paper (decoded) median [IQR] | authors\' code (n; finals; converged) | ours pre-fix | ours post-fix | judgment pre → post (paper statistic; shared rule, raw) |')
print('|---|---|---|---|---|---|---|')
OMAP = {'E3 B.1 λ=0': 'E3B1/0', 'E3 B.1 λ=0.25': 'E3B1/0.25', 'E3 B.1 λ=0.75': 'E3B1/0.75', 'E3 B.1 λ=1.0': 'E3B1/1.0',
        'E3 main λ=0.25': 'E3main/0.25', 'E3 main λ=0.5': 'E3main/0.5', 'E3 main λ=1.0': 'E3main/1.0',
        'E2 TD1 a2d_noq': 'E2/TD1/a2d_noq', 'E2 TD1 a2d_q': 'E2/TD1/a2d_q', 'E2 TD3 a2d_noq': 'E2/TD3/a2d_noq',
        'E1 TD A2D(Compact)': 'E1/TD/a2d_compact'}
for k, p in P.items():
    if k.startswith('abl'):
        continue
    b, a = p['before'], p['after']
    pa = a.get('paper_at_x')
    o = O.get(OMAP.get(k, ''), None)
    os_ = f"n {o['n']}: {', '.join(f2(v) for v in o['finals'])}; {o['converged']}/{o['n']}" if o else '–'
    pstr = f"{pa['median']:.2f} [{pa['q'][0]:.2f}, {pa['q'][1]:.2f}]" if pa else '– (no figure)'
    print(f"| {k} | {a['paper_stat']['x']:.0f} | {pstr} | {os_} | {ps(b)} | {ps(a)} | {b.get('verdict_paper_stat', '–')} → {a.get('verdict_paper_stat', '–')} |")
