"""Full-scale report: finals per arm (mean of the last 5 stochastic evals per seed; mean +- s.e. over seeds), the
pre-declared tests (families P, C, E2, E3, A; see README), paper and authors'-code reference
numbers, and the per-seed index runs/seed_<n>/metrics.json rebuilt from every per-run metrics file (the two CPU queues
and the GPU driver write the index concurrently, so it is rebuilt here). Writes runs/full/report.json."""
import glob, json, os, sys
import numpy as np
import scipy.stats as st
from rl_jax_bench.stats import mean_se, paired_compare, holm  # noqa: E402

OPT = {'FL': (32 / 3, 4.0), 'TD': (6.0, 2.0), 'TD1': (18.0, 16.0), 'TD2': (16.0, 14.0), 'TD3': (14.0, 12.0)}
# Paper numbers (raw undiscounted stochastic return; paper text values exact, plot reads +-15 % of |R_norm|).
PAPER = {('FL', 'a2d_compact'): 4.0, ('TD', 'a2d_compact'): 2.0, ('FL', 'a2d_image'): 4.0, ('TD', 'a2d_image'): 2.0,
         ('FL', 'rl_compact'): 4.0, ('TD', 'rl_compact'): 2.0, ('FL', 'rl_mdp'): 32 / 3 - 0.5, ('TD', 'rl_mdp'): 6.0,
         ('FL', 'ail_compact'): -30.0, ('TD', 'ail_compact'): -56.0, ('FL', 'ail_image'): -30.0,
         ('TD', 'ail_image'): -56.0, ('FL', 'rl_image'): -29.0, ('TD', 'rl_image'): -14.0,
         ('FL', 'rl_asym'): -29.0, ('TD', 'rl_asym'): -14.0,
         ('TD1', 'a2d_q'): 16.0, ('TD1', 'a2d_noq'): -42.0, ('TD2', 'a2d_q'): 14.0, ('TD2', 'a2d_noq'): 14.0,
         ('TD3', 'a2d_q'): 12.0, ('TD3', 'a2d_noq'): 12.0}
PAPER_TEXT = {('FL', 'ail'): -26.6, ('TD', 'ail'): -54.0}
OFFICIAL = {('FL', 'a2d_compact'): (3.92, 0.05, 3), ('TD', 'a2d_compact'): (1.78, 0.12, 3),
            ('FL', 'ail_compact'): (-29.68, 0.78, 3), ('TD', 'ail_compact'): (-54.09, 1.10, 3),
            ('FL', 'rl_mdp'): (9.41, 0.86, 3), ('TD', 'rl_mdp'): (5.96, 0.02, 3),
            ('FL', 'a2d_image'): (3.89, None, 1)}   # official A2D(Image) FL: one run, value at iteration 30


def load(exp, env, arm):
    out = {}
    for f in glob.glob(f'runs/{exp}/{env}/{arm}/seed_*/metrics.json'):
        m = json.load(open(f))
        out[m['seed']] = m
    return out


def key_main(arm):
    return 'expert_sto/ret' if arm == 'rl_mdp' else 'learner_sto/ret'


def fin(ms, key, seeds=None):
    seeds = sorted(ms) if seeds is None else seeds
    return np.array([np.nanmean(np.asarray(ms[s]['curves'][key][-5:], float)) for s in seeds])


def judge(ours, paper, env):
    if paper is None:
        return None
    off = OPT[env][0] + 1.0
    rr = abs(ours - paper) / max(abs(paper), 1e-9)
    rn = abs((ours - off) - (paper - off)) / abs(paper - off)
    j = lambda r: 'MATCH' if r <= 0.05 else ('CLOSE' if r <= 0.20 else 'DIVERGENT')
    return {'paper': paper, 'delta_raw_pct': 100 * (ours - paper) / max(abs(paper), 1e-9),
            'delta_norm_pct': 100 * ((ours - off) - (paper - off)) / abs(paper - off),
            'judge_raw': j(rr), 'judge_norm': j(rn)}


def first_reach(ms, key, env, seeds):
    """First interaction count at which the across-seed median is within 5 % (normalized scale) of the POMDP optimum."""
    mdp, pomdp = OPT[env]
    thr = pomdp - 0.05 * abs(pomdp - (mdp + 1))
    cur = np.array([ms[s]['curves'][key] for s in seeds], float)
    med = np.median(cur, 0)
    inter = ms[seeds[0]]['interactions']
    hit = np.where(med >= thr)[0]
    return int(inter[hit[0]]) if len(hit) else None


def summarise(exp, env, arm):
    ms = load(exp, env, arm)
    if not ms:
        return None
    seeds = sorted(ms)
    main = key_main(arm)
    r = fin(ms, main, seeds)
    d = {'exp': exp, 'seeds': seeds, 'n': len(seeds), 'iters': ms[seeds[0]]['iters'][-1],
         'ret_last5_per_seed': r.round(4).tolist(), 'ret': mean_se(r), 'median': float(np.median(r)),
         'iqr': [float(np.percentile(r, 25)), float(np.percentile(r, 75))],
         'ret_norm': float(r.mean() - (OPT[env][0] + 1)) if env in OPT else None,
         'finite': all(np.all(np.isfinite(np.asarray(ms[s]['curves'][main], float))) for s in seeds),
         'device': 'cpu' if ms[seeds[0]]['config']['obs'] != 'image' and exp != 'small_ail_rerun' else 'gpu',
         'dtype': ms[seeds[0]]['config'].get('dtype', 'float32'),
         'n_stuck': int(np.sum(r < OPT[env][1] - 0.5 * abs(OPT[env][1] - OPT[env][0] - 1))) if env in OPT else None,
         'timing': ms[seeds[0]]['timing']}
    for k in ['learner_sto/kl_clean', 'proj_loss', 'learner_det/ret', 'learner_sto/haz_rate', 'learner_sto/press_rate',
              'expert_sto/ret']:
        if k in ms[seeds[0]]['curves'] and k != main:
            d[k] = mean_se(fin(ms, k, seeds))
    if env in OPT:
        d['first_reach_opt'] = first_reach(ms, main, env, seeds)
        d['judge'] = judge(float(r.mean()), PAPER.get((env, arm.split('+')[0])), env)
    return d


def one_sided(pc):
    return pc['t_p'] / 2 if pc['t'] > 0 else 1 - pc['t_p'] / 2


def paired(expa, enva, arma, expb, envb, armb, seeds=None):
    a, b = load(expa, enva, arma), load(expb, envb, armb)
    s = sorted(set(a) & set(b)) if seeds is None else [x for x in seeds if x in a and x in b]
    if len(s) < 2:
        return None
    pc = paired_compare(fin(a, key_main(arma), s), fin(b, key_main(armb), s))
    pc['seeds'] = s
    pc['p_one_sided'] = one_sided(pc)
    return pc


CX = os.environ.get('FULL_COMPACT_EXP', 'full')        # compact E1 exp: full (float32) | full_f64 (float64 re-run)
E2X = os.environ.get('FULL_E2_EXP', 'full_e2')          # E2 exp: full_e2 (float32) | full_f64_e2
OUTF = os.environ.get('FULL_REPORT', 'runs/full/report.json')


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rep = {'arms': {}, 'compact_exp': CX, 'e2_exp': E2X}
    ARMS_E1 = [(CX, a) for a in ['rl_mdp', 'a2d_compact', 'rl_compact', 'ail_compact']] + \
        [('full', a) for a in ['a2d_image', 'rl_asym', 'rl_image']] + [('small_ail_rerun', 'ail_image')]
    for env in ['FL', 'TD']:
        for exp, arm in ARMS_E1:
            d = summarise(exp, env, arm)
            if d is None and arm in ('rl_image', 'rl_asym', 'a2d_image'):
                d = summarise('small', env, arm)          # skipped by the budget line -> small-scale 60-it value, labeled
                if d:
                    d['FALLBACK'] = 'small 60-iteration run (full-scale run skipped by the budget line)'
            if d:
                d['official'] = OFFICIAL.get((env, arm))
                rep['arms'][f'{env}/{arm}'] = d
    for env in ['TD1', 'TD2', 'TD3']:
        for arm in ['a2d_q', 'a2d_noq']:
            d = summarise(E2X, env, arm)
            if d:
                rep['arms'][f'{env}/{arm}'] = d
    for lm in ['0', '0.25', '0.5', '0.75', '1.0']:
        d = summarise('full_e3', 'TD2', f'a2d_compact+lam{lm}')
        if d:
            ms = load('full_e3', 'TD2', f'a2d_compact+lam{lm}')
            cur = np.array([ms[s]['curves']['learner_sto/ret'] for s in sorted(ms)], float)
            d['peak_minus_last5'] = mean_se(fin(ms, 'learner_sto/ret') - np.nanmax(cur, 1))
            d['ret_norm_E3'] = float(d['ret'][0] - 15.0)       # Fig. B.2 scale: POMDP optimum -> -1
            rep['arms'][f'TD2/a2d_compact+lam{lm}'] = d
    for env in ['FL', 'TD']:
        for v in ['beta0', 'beta095', 'critic_state', 'critic_mix05']:
            d = summarise('full_abl', env, f'a2d_compact+{v}')
            if d:
                rep['arms'][f'{env}/a2d_compact+{v}'] = d

    # ---- family P: image, n = 3, Holm over 6 (one-sided)
    P = {}
    for env in ['FL', 'TD']:
        P[f'{env}: A2D(Image) - AIL(Image)'] = paired('full', env, 'a2d_image', 'small_ail_rerun', env, 'ail_image')
        for arm in ['rl_image', 'rl_asym']:
            exp_b = 'full' if load('full', env, arm) else 'small'
            P[f'{env}: A2D(Image) - {arm}' + ('' if exp_b == 'full' else ' [small 60-it fallback]')] = \
                paired('full', env, 'a2d_image', exp_b, env, arm)
    P = {k: v for k, v in P.items() if v}
    if P:
        h = holm({k: v['p_one_sided'] for k, v in P.items()})
        for k in P:
            P[k]['p_holm'] = h[k]['p_holm']
            P[k]['reject_H0'] = h[k]['p_holm'] < 0.05
    rep['family_P'] = P
    # ---- family C: compact, n = 20, Holm over 2
    C = {f'{env}: A2D(Compact) - AIL(Compact)': paired(CX, env, 'a2d_compact', CX, env, 'ail_compact')
         for env in ['FL', 'TD']}
    C = {k: v for k, v in C.items() if v}
    if C:
        h = holm({k: v['p_one_sided'] for k, v in C.items()})
        for k in C:
            C[k]['p_holm'] = h[k]['p_holm']
            C[k]['reject_H0'] = h[k]['p_holm'] < 0.05
    rep['family_C'] = C
    # ---- secondary
    S = {}
    for env in ['FL', 'TD']:
        pc = paired(CX, env, 'a2d_compact', CX, env, 'rl_compact')
        if pc:
            S[f'{env}: A2D(Compact) - RL(Compact) (two-sided)'] = pc
        for arm, ex in [('a2d_compact', CX), ('a2d_image', 'full')]:
            ms = load(ex, env, arm)
            if ms:
                r = fin(ms, 'learner_sto/ret')
                t = st.ttest_1samp(r, OPT[env][1])
                S[f'{env}: {arm} - POMDP optimum (one-sample)'] = {
                    'diff_mean': float(r.mean() - OPT[env][1]), 't': float(t.statistic), 'p_two_sided': float(t.pvalue),
                    'median': float(np.median(r)), 'n_at_opt_within_5pct':
                        int(np.sum(r >= OPT[env][1] - 0.05 * abs(OPT[env][1] - OPT[env][0] - 1))), 'n': len(r)}
        for arm, exp in [('a2d_compact', CX), ('ail_compact', CX), ('a2d_image', 'full'),
                         ('ail_image', 'small_ail_rerun')]:
            ms = load(exp, env, arm)
            if ms:
                S[f'{env}: {arm} KL clean / F(psi)'] = {'kl_clean': mean_se(fin(ms, 'learner_sto/kl_clean')),
                                                        'F_psi': mean_se(fin(ms, 'proj_loss'))}
    rep['secondary'] = S
    # ---- E2: Holm over 3
    E2 = {}
    for env in ['TD1', 'TD2', 'TD3']:
        pc = paired(E2X, env, 'a2d_q', E2X, env, 'a2d_noq')
        if pc:
            pc['p_test'] = pc['p_one_sided'] if env == 'TD1' else pc['t_p']
            pc['equiv_margin'] = 0.10 * OPT[env][1]
            pc['equivalent_within_10pct'] = bool(pc['t95'][0] > -pc['equiv_margin'] and pc['t95'][1] < pc['equiv_margin'])
            E2[env] = pc
    if E2:
        h = holm({k: v['p_test'] for k, v in E2.items()})
        for k in E2:
            E2[k]['p_holm'] = h[k]['p_holm']
    rep['E2'] = E2
    # ---- E3: Spearman lambda vs F(psi) over 0.25..1.0, one-sided permutation
    lam, F, R = [], [], {}
    for lm in ['0.25', '0.5', '0.75', '1.0']:
        ms = load('full_e3', 'TD2', f'a2d_compact+lam{lm}')
        if ms:
            f = fin(ms, 'proj_loss')
            lam += [float(lm)] * len(f)
            F += f.tolist()
    if len(set(lam)) >= 2:
        rho = st.spearmanr(lam, F).statistic
        rng = np.random.default_rng(0)
        perm = np.array([st.spearmanr(rng.permutation(lam), F).statistic for _ in range(10000)])
        rep['E3_spearman'] = {'rho': float(rho), 'p_one_sided_perm': float((np.sum(perm >= rho) + 1) / 10001),
                              'n': len(F)}
    pc = paired('full_e3', 'TD2', 'a2d_compact+lam0.5', 'full_e3', 'TD2', 'a2d_compact+lam1.0')
    if pc:
        rep['E3_lam05_minus_lam1'] = pc
    # ---- ablation family A: Holm over 8 (two-sided), paired with E1 A2D(Compact) seeds 0-9
    A = {}
    for env in ['FL', 'TD']:
        for v in ['beta0', 'beta095', 'critic_state', 'critic_mix05']:
            pc = paired('full_abl', env, f'a2d_compact+{v}', 'full', env, 'a2d_compact', seeds=list(range(10)))
            if pc:
                A[f'{env}: {v} - main'] = pc
        pc = paired('full', env, 'ail_compact', 'full', env, 'a2d_compact', seeds=list(range(10)))
        if pc:
            rep.setdefault('ablation_expert_off', {})[f'{env}: expert-off(AIL) - main'] = pc
    for v in ['beta095', 'critic_state']:                  # float64 re-runs, paired with the float64 main arm
        pc = paired('full_f64_abl', 'TD', f'a2d_compact+{v}', 'full_f64', 'TD', 'a2d_compact', seeds=list(range(10)))
        if pc:
            rep.setdefault('ablation_f64', {})[f'TD: {v} - main (float64)'] = pc
            rep['arms'][f'TD/a2d_compact+{v} (float64)'] = summarise('full_f64_abl', 'TD', f'a2d_compact+{v}')
    if A:
        h = holm({k: v['t_p'] for k, v in A.items()})
        for k in A:
            A[k]['p_holm'] = h[k]['p_holm']
    rep['family_A'] = A
    json.dump(rep, open(OUTF, 'w'), indent=1, default=float)

    # ---- rebuild the per-seed index runs/seed_<n>/metrics.json from every per-run file
    idx = {}
    for f in sorted(glob.glob('runs/*/*/*/seed_*/metrics.json')):
        m = json.load(open(f))
        e = f.split('/')
        idx.setdefault(m['seed'], {})[f"{e[1]}/{e[2]}/{e[3]}"] = {
            'final': m['final'], 'config_hash': m['config_hash'], 'detail': f}
    for sd, d in idx.items():
        os.makedirs(f'runs/seed_{sd}', exist_ok=True)
        json.dump(d, open(f'runs/seed_{sd}/metrics.json', 'w'), indent=1)
    print(json.dumps({k: (v['ret'], v['n'], v['iters']) for k, v in rep['arms'].items()}, indent=0))
    for fam in ['family_P', 'family_C', 'E2', 'family_A']:
        for k, v in rep[fam].items():
            print(fam, k, 'diff %.3f +- %.3f' % (v['diff_mean'], v['diff_se']), 'p1 %.2e' % v['p_one_sided'],
                  'p_holm %.2e' % v.get('p_holm', np.nan), 'wins %d/%d' % (v['wins'], v['n']))
    print('E3', rep.get('E3_spearman'))


if __name__ == '__main__':
    main()
