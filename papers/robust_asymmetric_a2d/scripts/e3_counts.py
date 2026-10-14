"""E3 seed counts at the pre-declared threshold of the follow-up runs: 'within 5 % of the POMDP optimum 14' = return >= 13.3.
full_extra_report.py counted at 14 - 0.05*3 = 13.85 instead. Writes runs/full_extra/e3_counts_predeclared.json (both thresholds)."""
import json, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs')
out = {}
for exp, arm, n in [('full_extra_e3b1', 'a2d_noq', 20), ('full_extra_e3main', 'a2d_compact', 10)]:
    for lam in ['0', '0.25', '0.5', '0.75', '1.0']:
        cur = np.array([json.load(open(f'{R}/{exp}/TD2/{arm}+lam{lam}/seed_{s}/metrics.json'))['curves']['learner_sto/ret']
                        for s in range(n)], float)
        last5 = cur[:, -5:].mean(1)
        row = {'n': n}
        for name, thr in [('predeclared_13.3', 14 * 0.95), ('report_13.85', 14 - 0.05 * 3)]:
            row[name] = {'converged_last5': int(np.sum(last5 >= thr)), 'ever_reached': int(np.sum(cur.max(1) >= thr))}
        out[f'{exp}/lam{lam}'] = row
        print(exp, lam, row)
json.dump(out, open(f'{R}/r35/e3_counts_predeclared.json', 'w'), indent=1)
