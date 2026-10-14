"""Report of the follow-up runs (scripts/full_extra.sh).
E3 in the App. B.1 configuration (headline) and the main-args variant; float64 ablation family A; the labeled
paper-estimator / own-target arms; E2 at 20 seeds; E1 TD A2D(Compact) with the TRPO freeze fix and the recomputed family C / secondaries. Also counts, per run,
TRPO freezes (train/trpo_ok) and fallback events (train/kl_fallback). Writes runs/full_extra/report.json and rebuilds the
per-seed index runs/seed_<n>/metrics.json (four lanes write it concurrently)."""
import glob, json, os, sys
import numpy as np
import scipy.stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_jax_bench.stats import mean_se, holm  # noqa: E402
from full_report import load, fin, summarise, paired, OPT  # noqa: E402

LAMS = ['0', '0.25', '0.5', '0.75', '1.0']
# Fig. B.2 median reads on the raw scale (normalized R - 15): lambda 0 degrades to about -10 normalized
# (raw 5) late; 0.25 / 0.5 / 0.75 reach -1 (raw 14); lambda 1 stays flat near -25 (raw -10).
PAPER_E3 = {'0': 5.0, '0.25': 14.0, '0.5': 14.0, '0.75': 14.0, '1.0': -10.0}


def verdict(ours, paper):
    r = abs(ours - paper) / abs(paper)
    return ('MATCH' if r <= 0.05 else 'CLOSE' if r <= 0.20 else 'DIVERGENT'), 100 * (ours - paper) / abs(paper)


def freeze_stats(ms):
    out = {'frozen_seeds': [], 'fallback_iter_frac': None, 'late_trpo_ok': None}
    fb, late = [], []
    for s in sorted(ms):
        c = ms[s]['curves']
        ok = np.asarray(c.get('train/trpo_ok', []), float)
        its = ms[s]['iters']
        idx = [i for i in range(1, len(ok)) if np.all(ok[i:] < 0.5)]
        if idx:
            out['frozen_seeds'].append([s, its[idx[0]]])
        if len(ok):
            late.append(np.nanmean(ok[-5:]))
        if 'train/kl_fallback' in c:
            fb.append(np.nanmean(np.asarray(c['train/kl_fallback'], float)[1:]))
    out['late_trpo_ok'] = float(np.mean(late)) if late else None
    out['fallback_iter_frac'] = float(np.mean(fb)) if fb else None
    out['seeds_with_fallback'] = int(sum(f > 0 for f in fb)) if fb else None
    return out


def e3(exp, arm_base):
    rows, lam, F = {}, [], []
    for lm in LAMS:
        arm = f'{arm_base}+lam{lm}'
        ms = load(exp, 'TD2', arm)
        if not ms:
            continue
        d = summarise(exp, 'TD2', arm)
        r = fin(ms, 'learner_sto/ret')
        cur = np.array([ms[s]['curves']['learner_sto/ret'] for s in sorted(ms)], float)
        its = np.array(ms[sorted(ms)[0]]['iters'])
        late = (its >= 230)
        d['peak_minus_late'] = mean_se(cur[:, late].mean(1) - np.nanmax(cur, 1))
        d['n_converged_last5'] = int(np.sum(r >= 14 - 0.05 * 3))
        d['n_ever_reached'] = int(np.sum(np.nanmax(cur, 1) >= 14 - 0.05 * 3))
        d['ret_norm_E3_mean'] = float(r.mean() - 15.0)
        d['ret_norm_E3_median'] = float(np.median(r) - 15.0)
        d['F_psi_median'] = float(np.median(fin(ms, 'proj_loss')))
        d['verdict_mean'] = verdict(float(r.mean()), PAPER_E3[lm])
        d['verdict_median'] = verdict(float(np.median(r)), PAPER_E3[lm])
        d['freeze'] = freeze_stats(ms)
        rows[lm] = d
        if lm != '0':
            f = fin(ms, 'proj_loss')
            lam += [float(lm)] * len(f)
            F += f.tolist()
    out = {'rows': rows}
    if len(set(lam)) >= 2:
        rho = st.spearmanr(lam, F).statistic
        rng = np.random.default_rng(0)
        perm = np.array([st.spearmanr(rng.permutation(lam), F).statistic for _ in range(10000)])
        out['spearman_lambda_F'] = {'rho': float(rho), 'p_one_sided_perm': float((np.sum(perm >= rho) + 1) / 10001),
                                    'n': len(F)}
    pc = paired(exp, 'TD2', f'{arm_base}+lam0.5', exp, 'TD2', f'{arm_base}+lam1.0')
    if pc:
        out['lam05_minus_lam1'] = pc
    return out


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rep = {'E3_B1': e3('full_extra_e3b1', 'a2d_noq'), 'E3_main_variant': e3('full_extra_e3main', 'a2d_compact')}
    # ---- E1 TD A2D(Compact) with the fix vs the stored float64 run
    a, b = load('full_extra_main', 'TD', 'a2d_compact'), load('full_f64', 'TD', 'a2d_compact')
    if a:
        same = [s for s in sorted(a) if s in b and a[s]['curves']['learner_sto/ret'] == b[s]['curves']['learner_sto/ret']]
        rep['E1_TD_a2d_fix'] = {'summary': summarise('full_extra_main', 'TD', 'a2d_compact'),
                                'seeds_bitwise_identical_to_full_f64': same,
                                'seeds_changed': [s for s in sorted(a) if s not in same],
                                'changed_finals_old_new': {s: [b[s]['final']['ret_sto_last5'], a[s]['final']['ret_sto_last5']]
                                                           for s in sorted(a) if s not in same and s in b},
                                'freeze': freeze_stats(a)}
        one = fin(a, 'learner_sto/ret')
        t = st.ttest_1samp(one, 2.0)
        rep['E1_TD_a2d_fix']['vs_optimum'] = {'diff_mean': float(one.mean() - 2.0), 'p_two_sided': float(t.pvalue),
                                              'median': float(np.median(one)),
                                              'n_within_5pct': int(np.sum(one >= 2.0 - 0.05 * 5)), 'n': len(one)}
        C = {'FL': paired('full_f64', 'FL', 'a2d_compact', 'full_f64', 'FL', 'ail_compact'),
             'TD': paired('full_extra_main', 'TD', 'a2d_compact', 'full_f64', 'TD', 'ail_compact')}
        h = holm({k: v['p_one_sided'] for k, v in C.items()})
        for k in C:
            C[k]['p_holm'] = h[k]['p_holm']
        rep['family_C'] = C
        rep['TD_a2d_minus_rl_compact'] = paired('full_extra_main', 'TD', 'a2d_compact', 'full_f64', 'TD', 'rl_compact')
    # ---- ablation family A (float64), Holm over 8; extra arms not in the family
    main_ref = {'FL': ('full_f64', 'FL'), 'TD': ('full_extra_main', 'TD')}
    A, X, arms = {}, {}, {}
    for env in ['FL', 'TD']:
        mexp, menv = main_ref[env]
        for v in ['beta0', 'beta095', 'critic_state', 'critic_mix05', 'paperest', 'own', 'critic_mix05own']:
            arm = f'a2d_compact+{v}'
            if not load('full_extra_abl', env, arm):
                continue
            pc = paired('full_extra_abl', env, arm, mexp, menv, 'a2d_compact', seeds=list(range(10)))
            d = summarise('full_extra_abl', env, arm)
            d['freeze'] = freeze_stats(load('full_extra_abl', env, arm))
            arms[f'{env}/{v}'] = d
            (A if v in ('beta0', 'beta095', 'critic_state', 'critic_mix05') else X)[f'{env}: {v} - main'] = pc
        mm = load(mexp, menv, 'a2d_compact')
        arms[f'{env}/main seeds 0-9'] = {'median': float(np.median(fin(mm, 'learner_sto/ret', list(range(10))))),
                                         'ret': mean_se(fin(mm, 'learner_sto/ret', list(range(10)))),
                                         'kl': mean_se(fin(mm, 'learner_sto/kl_clean', list(range(10))))}
    if A:
        h = holm({k: v['t_p'] for k, v in A.items()})
        for k in A:
            A[k]['p_holm'] = h[k]['p_holm']
    rep['family_A_f64'], rep['extra_arms'], rep['ablation_arms'] = A, X, arms
    # ---- E2 at 20 seeds (TD2, TD3) + TD1 from full_f64_e2, Holm over 3
    E2 = {}
    for env, exp in [('TD1', 'full_f64_e2'), ('TD2', 'full_extra_e2'), ('TD3', 'full_extra_e2')]:
        pc = paired(exp, env, 'a2d_q', exp, env, 'a2d_noq')
        if pc:
            pc['p_test'] = pc['p_one_sided'] if env == 'TD1' else pc['t_p']
            pc['equiv_margin'] = 0.10 * OPT[env][1]
            pc['equivalent_within_10pct'] = bool(pc['t95'][0] > -pc['equiv_margin'] and
                                                 pc['t95'][1] < pc['equiv_margin'])
            E2[env] = pc
            for arm in ['a2d_q', 'a2d_noq']:
                rep.setdefault('E2_arms', {})[f'{env}/{arm}'] = summarise(exp, env, arm)
    if E2:
        h = holm({k: v['p_test'] for k, v in E2.items()})
        for k in E2:
            E2[k]['p_holm'] = h[k]['p_holm']
    rep['E2'] = E2
    # ---- E2 TD1 no-Q transient: median curve peak and where it is
    ms = load('full_f64_e2', 'TD1', 'a2d_noq')
    if ms:
        cur = np.array([ms[s]['curves']['learner_sto/ret'] for s in sorted(ms)], float)
        med = np.median(cur, 0)
        it = ms[sorted(ms)[0]]['interactions']
        k = int(np.argmax(med[1:]) + 1)
        rep['TD1_noq_median_curve'] = {'peak': float(med[k]), 'peak_interactions': it[k], 'final': float(med[-1]),
                                       'at_1e5': float(med[it.index(100000)]) if 100000 in it else None}
    os.makedirs('runs/r35', exist_ok=True)
    json.dump(rep, open('runs/full_extra/report.json', 'w'), indent=1, default=float)
    # ---- rebuild the per-seed index (skipped with NO_INDEX=1 while lanes are still writing it)
    idx = {}
    for f in ([] if os.environ.get('NO_INDEX') else sorted(glob.glob('runs/*/*/*/seed_*/metrics.json'))):
        m = json.load(open(f))
        e = f.split('/')
        idx.setdefault(m['seed'], {})[f"{e[1]}/{e[2]}/{e[3]}"] = {
            'final': m['final'], 'config_hash': m['config_hash'], 'detail': f}
    for sd, d in idx.items():
        os.makedirs(f'runs/seed_{sd}', exist_ok=True)
        json.dump(d, open(f'runs/seed_{sd}/metrics.json', 'w'), indent=1)
    # ---- print
    for name in ['E3_B1', 'E3_main_variant']:
        for lm, d in rep[name]['rows'].items():
            print(name, lm, 'mean %.2f+-%.2f med %.2f conv %d/%d ever %d F %.3g' % (
                d['ret'][0], d['ret'][1], d['median'], d['n_converged_last5'], d['n'], d['n_ever_reached'],
                d['proj_loss'][0]), d['verdict_mean'], d['verdict_median'], d['freeze'])
        print(name, rep[name].get('spearman_lambda_F'),
              {k: rep[name]['lam05_minus_lam1'][k] for k in ('diff_mean', 'diff_se', 'p_one_sided')}
              if 'lam05_minus_lam1' in rep[name] else None)
    for fam in ['family_A_f64', 'extra_arms']:
        for k, v in rep[fam].items():
            print(fam, k, 'diff %.3f +- %.3f p %.3g holm %s wins %d/%d' % (v['diff_mean'], v['diff_se'], v['t_p'],
                                                                        v.get('p_holm'), v['wins'], v['n']))
    for k, v in rep['ablation_arms'].items():
        print('arm', k, v.get('ret'), v.get('median'), v.get('freeze'))
    if 'E1_TD_a2d_fix' in rep:
        e = rep['E1_TD_a2d_fix']
        print('E1 TD fix', e['summary']['ret'], e['summary']['median'], 'changed', e['seeds_changed'],
              e['changed_finals_old_new'], e['freeze'], e['vs_optimum'])
        for k, v in rep['family_C'].items():
            print('C', k, v['diff_mean'], v['diff_se'], v['p_holm'], v['wins'])
    for k, v in rep['E2'].items():
        print('E2', k, v['diff_mean'], v['diff_se'], v['t95'], v['p_test'], v['p_holm'], v['equivalent_within_10pct'])
    for k, v in rep.get('E2_arms', {}).items():
        print('E2arm', k, v['ret'], v['median'], v['iqr'], v['n_stuck'])
    print('TD1 noQ', rep.get('TD1_noq_median_curve'))


if __name__ == '__main__':
    main()
