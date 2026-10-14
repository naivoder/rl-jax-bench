"""Recheck report: before (stored pre-fix runs) vs after (log-likelihood clamp re-runs, same seeds), the paper's own
statistic against the decoded vector figures, and the authors'-code reference runs. Writes runs/recheck/report.json and
prints markdown rows. Runs that are not finished yet are skipped (reported as missing).

Statistics (pre-declared before the re-runs):
* final (tail statistic, pre-declared since the full-scale runs) = per-seed mean of the last 5 stochastic evaluations;
  mean +- SE, median [IQR], converged = final >= opt - 5 % |opt|; before/after paired two-sided t, n changed > 0.01;
* paper statistic = per-x median [q25, q75] over seeds of the raw evaluation at the final x, and the mean of the median
  curve over its last 5 points; paper value = decoded figure at the same x;
* fallback = number of TRPO iterations that used the KL fallback of the TRPO freeze fix (block mean x 5, summed);
* median-curve drop = max of the median curve - mean of its last 3 points.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats as st

from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, 'runs')
MEAS = os.path.join(REPO, 'data', 'paper_figures')
OFFICIAL_ROOT = os.environ.get('OFFICIAL_ROOT', os.path.join(REPO, 'external', 'official'))
OFF = os.path.join(OFFICIAL_ROOT, 'oracle')
OFF_SA = os.path.join(OFFICIAL_ROOT, 'recheck_logs')
OPT = {'FL': 4.0, 'TD': 2.0, 'TD1': 16.0, 'TD2': 14.0, 'TD3': 12.0}
SUB = {'FL': 11.66, 'TD': 7.0}


def load(exp, env, arm):
    fs = glob.glob(f'{R}/{exp}/{env}/{arm}/seed_*/metrics.json')
    out = {}
    for f in fs:
        m = json.load(open(f))
        out[int(m['seed'])] = m
    return out


def curve(m, key):
    return np.asarray(m['curves'][key], float)


def arm_stats(ms, key='learner_sto/ret', opt=None):
    s = sorted(ms)
    fin = np.array([curve(ms[i], key)[-5:].mean() for i in s])
    Y = np.array([curve(ms[i], key) for i in s])
    x = np.asarray(ms[s[0]]['interactions'], float)
    med = np.median(Y, 0)
    d = {'n': len(s), 'seeds': s, 'finals': fin.round(4).tolist(), 'mean_se': mean_se(fin),
         'median': float(np.median(fin)), 'iqr': np.quantile(fin, [0.25, 0.75]).tolist(),
         'paper_stat': {'x': float(x[-1]), 'median': float(med[-1]),
                        'q': np.quantile(Y[:, -1], [0.25, 0.75]).tolist(), 'last5_of_median': float(med[-5:].mean())},
         'median_curve_drop': float(med.max() - med[-3:].mean()), 'median_curve_peak_x': float(x[int(med.argmax())])}
    if opt is not None:
        thr = opt - 0.05 * abs(opt)
        d['converged_end'] = int((fin >= thr).sum())
        d['converged_ever'] = int((Y.max(1) >= thr).sum())
        i = np.where(med >= thr)[0]
        d['median_first_x_95pct'] = float(x[i[0]]) if len(i) else None
    fb = [np.nansum(np.nan_to_num(curve(ms[i], 'train/kl_fallback'), nan=0.0)) * 5
          if 'train/kl_fallback' in ms[i]['curves'] else np.nan for i in s]
    d['fallback_iters_total'] = float(np.nansum(fb))
    d['fallback_seeds'] = int(np.sum(np.asarray(fb) > 0))
    d['iters_total'] = int(len(s) * ms[s[0]]['iters'][-1])
    if 'train/n_clamped' in ms[s[0]]['curves']:
        d['n_clamped_mean'] = float(np.nanmean([np.nanmean(curve(ms[i], 'train/n_clamped')) for i in s]))
    if 'proj_loss' in ms[s[0]]['curves']:
        F = np.array([curve(ms[i], 'proj_loss') for i in s])
        Fl5 = np.nanmean(F[:, -5:], 1)
        d['F_final_mean'] = float(np.nanmean(Fl5))
        d['F_paper_stat'] = {'median': float(np.nanmedian(F[:, -1])), 'q': np.nanquantile(F[:, -1], [.25, .75]).tolist()}
        d['F_finals'] = Fl5.tolist()
    d['_Y'] = Y
    d['_x'] = x
    return d


def before_after(b, a):
    s = sorted(set(b['seeds']) & set(a['seeds']))
    fb = np.array([b['finals'][b['seeds'].index(i)] for i in s])
    fa = np.array([a['finals'][a['seeds'].index(i)] for i in s])
    d = fa - fb
    if np.allclose(d, 0):
        t_p = 1.0
        diff = (0.0, 0.0)
    else:
        t_p = float(st.ttest_rel(fa, fb).pvalue)
        diff = mean_se(d)
    return {'n': len(s), 'delta_mean_se': diff, 't_p_two_sided': t_p, 'n_changed_gt_0.01': int((np.abs(d) > 0.01).sum()),
            'n_bit_identical': int((d == 0).sum()), 'delta_per_seed': d.round(3).tolist()}


# --------------------------------------------------------------------------------------------- paper (decoded figures)
def _pc(c):
    m = np.array(c['median'][0])
    e = [np.array(x) for x in c.get('edge', []) if len(x) == len(m)]
    return m, e


def paper_at(fig, env, name, x, kind='reward'):
    """(median, q_lo, q_hi) of the decoded paper curve at interaction count x (raw scale)."""
    if fig in ('fig4', 'fig5'):
        o = json.load(open(f"{MEAS}/{fig}_{'IceLake' if env == 'FL' else 'TigerDoor'}.json"))['curves'][name]
        m, e = _pc(o)
        px, ys = m[:, 0], [m[:, 1]] + [ee[:, 1] for ee in e]
        if fig == 'fig4':
            ys = [y + SUB[env] for y in ys]
    elif fig == 'figB3':
        o = json.load(open(f'{MEAS}/figB3.json'))[env]['curves'][name]
        m, e = _pc(o)
        px, ys = 10 ** m[:, 0], [y + SUB[env] for y in [m[:, 1]] + [ee[:, 1] for ee in e]]
    elif fig == 'figB1':
        o = json.load(open(f'{MEAS}/figB1.json'))[env][kind]['curves'][name]
        m, e = _pc(o)
        px, ys = m[:, 0], [m[:, 1]] + [ee[:, 1] for ee in e]
    elif fig == 'figB2':
        o = json.load(open(f'{MEAS}/figB2.json'))[kind]['curves'][name]
        m, e = _pc(o)
        px = m[:, 0]
        ys = [m[:, 1]] + [ee[:, 1] for ee in e]
        if kind == 'reward':
            ys = [y + 15 for y in ys]
    v = [float(np.interp(x, px, y)) for y in ys]
    q = sorted(v[1:3]) if len(v) >= 3 else [np.nan, np.nan]
    full = {'x': px.tolist(), 'median': ys[0].tolist()}
    return {'median': v[0], 'q': q, 'last5_of_median': float(np.mean([np.interp(x - k * 1e4, px, ys[0]) for k in range(5)])),
            'curve': full}


def paper_drop(name):
    p = paper_at('figB2', 'TD2', name, 5e5)
    y = np.asarray(p['curve']['median'])
    return float(y.max() - y[-3:].mean()), float(p['curve']['x'][int(y.argmax())])


def verdict(ours, paper, q, rel_scale=None):
    """shared rule: MATCH within 5 % relative or inside the paper's IQR; CLOSE within 20 %; else DIVERGENT."""
    if q is not None and not np.isnan(q[0]) and q[0] - 1e-9 <= ours <= q[1] + 1e-9:
        return 'MATCH (inside IQR)'
    den = abs(paper) if rel_scale is None else abs(rel_scale)
    r = abs(ours - paper) / den
    return ('MATCH' if r <= 0.05 else 'CLOSE' if r <= 0.2 else 'DIVERGENT') + f' ({100 * (ours - paper) / den:+.1f} %)'


def verdict_F(ours, paper, q):
    if q[0] <= ours <= q[1]:
        return 'MATCH (inside IQR)'
    r = ours / paper
    return ('CLOSE' if 0.5 <= r <= 2 else 'DIVERGENT') + f' ({np.log10(r):+.2f} decade)'


# --------------------------------------------------------------------------------------------- official logs
EP = re.compile(r'Ep:\s*(\d+)/\d+.*?Exp-s\s+([-\d.]+),\s*Lea-s\s+([-\d.]+),\s*Div:\s*([\w./]+)')


def parse_off(path):
    rows = []
    for line in open(path, errors='ignore'):
        mm = EP.search(line)
        if mm:
            div = float(mm.group(4)) if mm.group(4) not in ('N/A',) else np.nan
            rows.append((int(mm.group(1)), float(mm.group(2)), float(mm.group(3)), div))
    return np.array(rows) if rows else None


def off_runs(paths, iters, opt):
    out = []
    for p in sorted(paths):
        r = parse_off(p)
        if r is None or r[-1, 0] < iters:
            continue
        out.append({'log': os.path.relpath(p, os.path.dirname(OFFICIAL_ROOT)), 'final': float(r[-5:, 2].mean()),
                    'end': float(r[-1, 2]), 'F_final': float(np.nanmean(r[-5:, 3])), 'curve': r[:, 2].tolist(),
                    'converged': bool(r[-5:, 2].mean() >= opt - 0.05 * abs(opt))})
    return out


def off_summary(runs):
    if not runs:
        return None
    f = np.array([r['final'] for r in runs])
    Y = np.array([r['curve'] for r in runs])
    med = np.median(Y, 0)
    return {'n': len(runs), 'finals': f.round(2).tolist(), 'mean_se': mean_se(f), 'converged': int(sum(r['converged'] for r in runs)),
            'median_curve_drop': float(med.max() - med[-3:].mean()), 'logs': [r['log'] for r in runs]}


OFFICIAL = {
    'E3B1/0': lambda: off_runs(glob.glob(f'{OFF_SA}/ref_b1lam0_*.log'), 250, 14),
    'E3B1/0.25': lambda: off_runs(glob.glob(f'{OFF}/runs3/b1lam0.25_*.log'), 250, 14),
    'E3B1/0.75': lambda: off_runs(glob.glob(f'{OFF_SA}/ref_b1lam0.75_*.log'), 250, 14),
    'E3B1/1.0': lambda: off_runs(glob.glob(f'{OFF}/runs3/b1lam1.0_*.log') + glob.glob(f'{OFF_SA}/ref_b1lam1.0_*.log'), 250, 14),
    'E3main/0.25': lambda: off_runs(glob.glob(f'{OFF}/runs3/lam0.25_*.log'), 250, 14),
    'E3main/0.5': lambda: off_runs(glob.glob(f'{OFF}/runs3/lam0.5_*.log'), 250, 14),
    'E3main/1.0': lambda: off_runs(glob.glob(f'{OFF}/runs3/lam1.0_*.log'), 250, 14),
    'E2/TD1/a2d_noq': lambda: off_runs(glob.glob(f'{OFF}/runs3/noq_*v1_*.log'), 200, 16),
    'E2/TD1/a2d_q': lambda: off_runs(glob.glob(f'{OFF}/runs3/q_*v1_*.log'), 200, 16),
    'E2/TD3/a2d_noq': lambda: off_runs(glob.glob(f'{OFF_SA}/ref_b1noq_*.log'), 200, 12),
    'E1/TD/a2d_compact': lambda: off_runs(glob.glob(f'{OFF_SA}/ref_a2dmain_MiniGrid*.log'), 200, 2),
    'E1/TD/a2d_compact lam0.9': lambda: off_runs(glob.glob(f'{OFF_SA}/ref_a2dmain_lam09_*.log'), 200, 2),
}


def main():
    rep = {'pairs': {}, 'variants': {}, 'official': {}, 'tests': {}}
    rows = []

    def pair(label, env, before, after, key='learner_sto/ret', paper=None):
        b, a = load(*before[:1], env, before[1]), load(*after[:1], env, after[1])
        if not a or not b:
            print(f'MISSING {label}: before {len(b)} after {len(a)}')
            return None
        sb, sa = arm_stats(b, key, OPT[env]), arm_stats(a, key, OPT[env])
        out = {'before': {k: v for k, v in sb.items() if not k.startswith('_')},
               'after': {k: v for k, v in sa.items() if not k.startswith('_')}, 'before_after': before_after(sb, sa),
               'before_src': '/'.join([before[0], env, before[1]]), 'after_src': '/'.join([after[0], env, after[1]])}
        if paper:
            fig, name = paper
            for side, s_ in (('before', sb), ('after', sa)):
                p = paper_at(fig, env, name, s_['paper_stat']['x'])
                out[side]['paper_at_x'] = {k: v for k, v in p.items() if k != 'curve'}
                out[side]['verdict_paper_stat'] = verdict(s_['paper_stat']['median'], p['median'], p['q'])
                out[side]['verdict_paper_stat_last5'] = verdict(s_['paper_stat']['last5_of_median'], p['last5_of_median'], None)
                out[side]['verdict_mean'] = verdict(s_['mean_se'][0], p['median'], None)
        rep['pairs'][label] = out
        ba = out['before_after']
        f = lambda s: f"{s['mean_se'][0]:.2f} ± {s['mean_se'][1]:.2f}; {s['median']:.2f} [{s['iqr'][0]:.2f}, {s['iqr'][1]:.2f}]; conv {s.get('converged_end')}/{s['n']}; fb {s['fallback_iters_total']:.0f}"
        rows.append(f"| {label} | {f(out['before'])} | {f(out['after'])} | {ba['delta_mean_se'][0]:+.2f} ± {ba['delta_mean_se'][1]:.2f} (p {ba['t_p_two_sided']:.2g}; {ba['n_changed_gt_0.01']}/{ba['n']} changed) |")
        return out

    # E1
    pair('E1 FL A2D(Compact)', 'FL', ('full_f64', 'a2d_compact'), ('recheck_main', 'a2d_compact'), paper=('fig4', 'A2D(Compact)'))
    pair('E1 TD A2D(Compact)', 'TD', ('full_extra_main', 'a2d_compact'), ('recheck_main', 'a2d_compact'), paper=('fig4', 'A2D(Compact)'))
    pair('E1 FL RL(Compact)', 'FL', ('full_f64', 'rl_compact'), ('recheck_rl', 'rl_compact'), paper=('figB3', 'RL(Compact)'))
    pair('E1 TD RL(Compact)', 'TD', ('full_f64', 'rl_compact'), ('recheck_rl', 'rl_compact'), paper=('figB3', 'RL(Compact)'))
    pair('E1 FL A2D(Image) pilot', 'FL', ('full', 'a2d_image'), ('recheck_img', 'a2d_image'), paper=('fig4', 'A2D(Image)'))
    pair('E1 TD A2D(Image) pilot', 'TD', ('full', 'a2d_image'), ('recheck_img', 'a2d_image'), paper=('fig4', 'A2D(Image)'))
    # E2
    for env, bexp in (('TD1', 'full_f64_e2'), ('TD2', 'full_extra_e2'), ('TD3', 'full_extra_e2')):
        for arm, col in (('a2d_noq', 'brown'), ('a2d_q', 'pink')):
            pair(f'E2 {env} {arm}', env, (bexp, arm), ('recheck_e2', arm), paper=('figB1', col))
    # E3
    for lm, pl in (('0', 'lam0.00'), ('0.25', 'lam0.25'), ('0.5', 'lam0.50'), ('0.75', 'lam0.75'), ('1.0', 'lam1.00')):
        pair(f'E3 B.1 λ={lm}', 'TD2', ('full_extra_e3b1', f'a2d_noq+lam{lm}'), ('recheck_e3b1', f'a2d_noq+lam{lm}'), paper=('figB2', pl))
        pair(f'E3 main λ={lm}', 'TD2', ('full_extra_e3main', f'a2d_compact+lam{lm}'), ('recheck_e3main', f'a2d_compact+lam{lm}'))
    # ablations / new arms
    for env in ('FL', 'TD'):
        for lab in ('beta0', 'beta095', 'critic_state', 'critic_mix05', 'paperest', 'own', 'critic_mix05own'):
            pair(f'abl {env} {lab}', env, ('full_extra_abl', f'a2d_compact+{lab}'), ('recheck_abl', f'a2d_compact+{lab}'))

    # paper-statistic F(psi) and median-curve drop
    for label, env, exp, arm, fig, name, x in (('F E1 FL A2D(Compact)', 'FL', 'recheck_main', 'a2d_compact', 'fig5', 'A2D(Compact)', None),
                                               ('F E1 TD A2D(Compact)', 'TD', 'recheck_main', 'a2d_compact', 'fig5', 'A2D(Compact)', None),
                                               ('F E1 FL A2D(Compact) pre-fix', 'FL', 'full_f64', 'a2d_compact', 'fig5', 'A2D(Compact)', None),
                                               ('F E1 TD A2D(Compact) pre-fix', 'TD', 'full_extra_main', 'a2d_compact', 'fig5', 'A2D(Compact)', None),
                                               ('F E1 FL A2D(Image) pilot', 'FL', 'recheck_img', 'a2d_image', 'fig5', 'A2D(Image)', None),
                                               ('F E1 TD A2D(Image) pilot', 'TD', 'recheck_img', 'a2d_image', 'fig5', 'A2D(Image)', None)):
        ms = load(exp, env, arm)
        if not ms:
            continue
        s_ = arm_stats(ms, 'learner_sto/ret', OPT[env])
        p = paper_at(fig, env, name, s_['paper_stat']['x'])
        rep['tests'][label] = {'ours': s_['F_paper_stat'], 'paper': {k: v for k, v in p.items() if k != 'curve'},
                               'verdict': verdict_F(s_['F_paper_stat']['median'], p['median'], p['q'])}
    for lm, pl in (('0', 'lam0.00'), ('0.25', 'lam0.25'), ('0.5', 'lam0.50'), ('0.75', 'lam0.75'), ('1.0', 'lam1.00')):
        drop, px = paper_drop(pl)
        for exp in ('full_extra_e3b1', 'recheck_e3b1'):
            ms = load(exp, 'TD2', f'a2d_noq+lam{lm}')
            if ms:
                s_ = arm_stats(ms, 'learner_sto/ret', 14)
                rep['tests'][f'median-curve drop λ={lm} {exp}'] = {'ours_drop': s_['median_curve_drop'], 'ours_peak_x': s_['median_curve_peak_x'],
                                                         'paper_drop': drop, 'paper_peak_x': px,
                                                         'F_paper_stat': s_['F_paper_stat']}
        pF = paper_at('figB2', 'TD2', pl, 5e5, 'kl')
        rep['tests'][f'figB2 F λ={lm} paper'] = {k: v for k, v in pF.items() if k != 'curve'}

    # pre-declared E3 tests (post-fix) for both configurations
    for exp, pre, arm in (('recheck_e3b1', 'full_extra_e3b1', 'a2d_noq'), ('recheck_e3main', 'full_extra_e3main', 'a2d_compact')):
        for tag in (exp, pre):
            F, L = [], []
            ok = True
            per = {}
            for lm in ('0.25', '0.5', '0.75', '1.0'):
                ms = load(tag, 'TD2', f'{arm}+lam{lm}')
                if not ms:
                    ok = False
                    break
                s_ = arm_stats(ms, 'learner_sto/ret', 14)
                per[lm] = s_
                F += s_['F_finals']
                L += [float(lm)] * len(s_['F_finals'])
            if not ok:
                continue
            rho = st.spearmanr(L, F).correlation
            rng = np.random.default_rng(0)
            perm = np.array([st.spearmanr(rng.permutation(L), F).correlation for _ in range(10000)])
            pc = paired_compare(per['0.5']['finals'], per['1.0']['finals'])
            p1 = pc['t_p'] / 2 if pc['t'] > 0 else 1 - pc['t_p'] / 2
            rep['tests'][f'E3 tests {tag}'] = {'spearman_rho': float(rho), 'perm_p_one_sided': float((perm >= rho).mean()),
                                               'lam05_minus_lam1': pc['diff_mean'], 'se': pc['diff_se'], 'wins': pc['wins'],
                                               'p_one_sided': float(p1), 'n': pc['n']}

    # E2 family (post-fix): Q - noQ per env, Holm over 3
    E2 = {}
    for env in ('TD1', 'TD2', 'TD3'):
        q, nq = load('recheck_e2', env, 'a2d_q'), load('recheck_e2', env, 'a2d_noq')
        if q and nq:
            sq, sn = arm_stats(q, 'learner_sto/ret', OPT[env]), arm_stats(nq, 'learner_sto/ret', OPT[env])
            s = sorted(set(sq['seeds']) & set(sn['seeds']))
            E2[env] = paired_compare([sq['finals'][sq['seeds'].index(i)] for i in s], [sn['finals'][sn['seeds'].index(i)] for i in s])
    if E2:
        h = holm({k: v['t_p'] for k, v in E2.items()})
        for k in E2:
            E2[k]['p_holm'] = h[k]['p_holm']
        rep['tests']['E2 family Q-noQ (post-fix)'] = {k: {kk: E2[k][kk] for kk in ('diff_mean', 'diff_se', 't95', 'wins', 'n', 't_p', 'p_holm')} for k in E2}

    # family C (post-fix): A2D(Compact) - AIL(Compact), one-sided, Holm over 2
    C = {}
    for env, aexp in (('FL', 'recheck_main'), ('TD', 'recheck_main')):
        a, b = load(aexp, env, 'a2d_compact'), load('full_f64', env, 'ail_compact')
        if a and b:
            sa_, sb_ = arm_stats(a), arm_stats(b)
            s = sorted(set(sa_['seeds']) & set(sb_['seeds']))
            pc = paired_compare([sa_['finals'][sa_['seeds'].index(i)] for i in s], [sb_['finals'][sb_['seeds'].index(i)] for i in s])
            pc['p_one_sided'] = pc['t_p'] / 2 if pc['t'] > 0 else 1 - pc['t_p'] / 2
            C[env] = pc
    if C:
        h = holm({k: v['p_one_sided'] for k, v in C.items()})
        rep['tests']['family C (post-fix)'] = {k: {'diff': C[k]['diff_mean'], 'se': C[k]['diff_se'], 'wins': C[k]['wins'],
                                                   'n': C[k]['n'], 'p_holm': h[k]['p_holm']} for k in C}
    # family P (post-fix): A2D(Image) - {AIL(Image), RL(Image), RL(Asym)}, n = 3, one-sided, Holm over 6
    P = {}
    for env in ('FL', 'TD'):
        a = load('recheck_img', env, 'a2d_image')
        if not a:
            continue
        for bexp, barm in (('small_ail_rerun', 'ail_image'), ('full', 'rl_image'), ('full', 'rl_asym')):
            b = load(bexp, env, barm)
            sa_, sb_ = arm_stats(a), arm_stats(b)
            s = sorted(set(sa_['seeds']) & set(sb_['seeds']))
            pc = paired_compare([sa_['finals'][sa_['seeds'].index(i)] for i in s], [sb_['finals'][sb_['seeds'].index(i)] for i in s])
            pc['p_one_sided'] = pc['t_p'] / 2 if pc['t'] > 0 else 1 - pc['t_p'] / 2
            P[f'{env}: A2D(Image) - {barm}'] = pc
    if P:
        h = holm({k: v['p_one_sided'] for k, v in P.items()})
        rep['tests']['family P (post-fix)'] = {k: {'diff': P[k]['diff_mean'], 'se': P[k]['diff_se'], 'wins': P[k]['wins'],
                                                   'n': P[k]['n'], 'p_one_sided': P[k]['p_one_sided'],
                                                   'p_holm': h[k]['p_holm']} for k in P}
    # ablation family A (post-fix): (a)-(d) x {FL, TD} vs main seeds 0-9, two-sided, Holm over 8; new arms unadjusted
    A, NEW = {}, {}
    for env in ('FL', 'TD'):
        main_ = load('recheck_main', env, 'a2d_compact')
        if not main_:
            continue
        sm = arm_stats({k: v for k, v in main_.items() if k < 10}, 'learner_sto/ret', OPT[env])
        for lab in ('beta0', 'beta095', 'critic_state', 'critic_mix05', 'paperest', 'own', 'critic_mix05own'):
            ab = load('recheck_abl', env, f'a2d_compact+{lab}')
            if not ab:
                continue
            sab = arm_stats(ab, 'learner_sto/ret', OPT[env])
            s = sorted(set(sab['seeds']) & set(sm['seeds']))
            pc = paired_compare([sab['finals'][sab['seeds'].index(i)] for i in s], [sm['finals'][sm['seeds'].index(i)] for i in s])
            row = {'diff': pc['diff_mean'], 'se': pc['diff_se'], 'wins': pc['wins'], 'n': pc['n'], 't_p': pc['t_p'],
                   'arm_median': sab['median'], 'arm_mean': sab['mean_se'][0], 'converged': sab['converged_end'],
                   'fallback_iters': sab['fallback_iters_total']}
            (A if lab in ('beta0', 'beta095', 'critic_state', 'critic_mix05') else NEW)[f'{env} {lab}'] = row
    if A:
        h = holm({k: v['t_p'] for k, v in A.items()})
        for k in A:
            A[k]['p_holm'] = h[k]['p_holm']
    rep['tests']['ablation family A (post-fix)'] = A
    rep['tests']['new arms (post-fix, unadjusted)'] = NEW

    # labeled variants
    for label, env, exp, arm, ref in (('lam09 (shipped TD launcher λ_a2d)', 'TD', 'recheck_var', 'a2d_compact+lam09', ('recheck_main', 'a2d_compact')),
                                      ('bufseed (clamp + official buffer seed)', 'TD', 'recheck_var', 'a2d_compact+bufseed', ('recheck_main', 'a2d_compact')),
                                      ('lam09 without clamp', 'TD', 'recheck_var', 'a2d_compact+lam09_noclamp', ('recheck_var', 'a2d_compact+lam09')),
                                      ('launch_q TD3 Q', 'TD3', 'recheck_var', 'a2d_compact+launchq_q1', None),
                                      ('launch_q TD3 noQ', 'TD3', 'recheck_var', 'a2d_compact+launchq_q0', None),
                                      ('B.1 λ=0.25 state critic', 'TD2', 'recheck_var', 'a2d_noq+lam0.25+critic_state', ('recheck_e3b1', 'a2d_noq+lam0.25')),
                                      ('B.1 λ=0.25 mix05own', 'TD2', 'recheck_var', 'a2d_noq+lam0.25+critic_mix05own', ('recheck_e3b1', 'a2d_noq+lam0.25')),
                                      ('B.1 λ=0.5 state critic', 'TD2', 'recheck_var', 'a2d_noq+lam0.5+critic_state', ('recheck_e3b1', 'a2d_noq+lam0.5')),
                                      ('B.1 λ=0.5 mix05own', 'TD2', 'recheck_var', 'a2d_noq+lam0.5+critic_mix05own', ('recheck_e3b1', 'a2d_noq+lam0.5'))):
        ms = load(exp, env, arm)
        if not ms:
            continue
        s_ = arm_stats(ms, 'learner_sto/ret', OPT[env])
        d = {k: v for k, v in s_.items() if not k.startswith('_')}
        if env == 'TD3':
            p = paper_at('figB1', 'TD3', 'brown' if arm.endswith('q0') else 'pink', s_['paper_stat']['x'])
            d['paper_at_x'] = {k: v for k, v in p.items() if k != 'curve'}
            d['median_at_1e5'] = float(np.median(s_['_Y'][:, list(s_['_x']).index(1e5)])) if 1e5 in s_['_x'] else None
        if ref:
            r_ = load(*ref[:1], env, ref[1])
            if r_:
                sr = arm_stats({k: v for k, v in r_.items() if k in ms}, 'learner_sto/ret', OPT[env])
                s = sorted(set(s_['seeds']) & set(sr['seeds']))
                pc = paired_compare([s_['finals'][s_['seeds'].index(i)] for i in s], [sr['finals'][sr['seeds'].index(i)] for i in s])
                d['vs_ref'] = {'ref': '/'.join([ref[0], env, ref[1]]), 'ref_median': sr['median'], 'ref_converged': sr['converged_end'],
                               'diff': pc['diff_mean'], 'se': pc['diff_se'], 't_p': pc['t_p'], 'n': pc['n'],
                               'fisher_p_converged': float(st.fisher_exact([[d['converged_end'], d['n'] - d['converged_end']],
                                                                           [sr['converged_end'], sr['n'] - sr['converged_end']]])[1])}
        rep['variants'][label] = d

    # TD3 no-Q speed (C5 row): median curve at 1e5 / 1.5e5 vs paper
    for tag, exp in (('pre-fix', 'full_extra_e2'), ('post-fix', 'recheck_e2')):
        ms = load(exp, 'TD3', 'a2d_noq')
        if ms:
            s_ = arm_stats(ms, 'learner_sto/ret', 12)
            x = list(s_['_x'])
            med = np.median(s_['_Y'], 0)
            rep['tests'][f'TD3 noQ median curve {tag}'] = {f'{int(v)}': float(med[x.index(v)]) for v in (4e4, 1e5, 1.5e5, 2e5, 3e5, 4e5) if v in x}
    pB = paper_at('figB1', 'TD3', 'brown', 1e5)
    rep['tests']['TD3 noQ paper median curve'] = {f'{int(v)}': float(np.interp(v, pB['curve']['x'], pB['curve']['median'])) for v in (4e4, 1e5, 1.5e5, 2e5, 3e5, 4e5)}

    # official
    for k, f in OFFICIAL.items():
        rep['official'][k] = off_summary(f())
    # main args: pooled converged fraction ours vs official, lambda in {0.25, 0.5, 1}
    oc = [rep['official'].get(f'E3main/{lm}') for lm in ('0.25', '0.5', '1.0')]
    ours = [rep['pairs'].get(f'E3 main λ={lm}') for lm in ('0.25', '0.5', '1.0')]
    if all(oc) and all(ours):
        oconv, on = sum(o['converged'] for o in oc), sum(o['n'] for o in oc)
        uconv, un = sum(u['after']['converged_end'] for u in ours), sum(u['after']['n'] for u in ours)
        rep['tests']['main-args pooled converged (λ .25/.5/1)'] = {'ours': [uconv, un], 'official': [oconv, on],
                                                             'fisher_p': float(st.fisher_exact([[uconv, un - uconv], [oconv, on - oconv]])[1])}
    # TD3 noQ ours vs official converged
    o = rep['official'].get('E2/TD3/a2d_noq')
    u = rep['pairs'].get('E2 TD3 a2d_noq')
    if o and u:
        rep['tests']['TD3 noQ converged ours vs official'] = {
            'ours_after': [u['after']['converged_end'], u['after']['n']], 'ours_before': [u['before']['converged_end'], u['before']['n']],
            'official': [o['converged'], o['n']],
            'fisher_p_after': float(st.fisher_exact([[u['after']['converged_end'], u['after']['n'] - u['after']['converged_end']],
                                                     [o['converged'], o['n'] - o['converged']]])[1])}

    os.makedirs(f'{R}/sa', exist_ok=True)
    json.dump(rep, open(f'{R}/sa/report.json', 'w'), indent=1, default=float)
    print('| cell | before: mean ± SE; median [IQR]; converged; fallback iters | after | Δ (after − before) |')
    print('|---|---|---|---|')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
