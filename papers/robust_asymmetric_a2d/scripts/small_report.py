"""Small-scale report: per-arm finals (mean of the last 5 evals per seed; mean +- s.e. over seeds), the pre-declared
PASS criteria (see README), and the official-code reference runs. Writes runs/small/report.json."""
import glob, json, os, sys
import numpy as np
from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

R = 'runs/small'
OPT = {'FL': (32 / 3, 4.0), 'TD': (6.0, 2.0), 'TD1': (18.0, 16.0)}


def load(env, arm):
    out = {}
    for f in glob.glob(f'{R}/{env}/{arm}/seed_*/metrics.json'):
        m = json.load(open(f))
        out[m['seed']] = m
    return out


def fin(ms, key):
    seeds = sorted(ms)
    return seeds, np.array([np.nanmean(ms[s]['curves'][key][-5:]) for s in seeds])


rep, rows = {}, []
for env in ['FL', 'TD', 'TD1']:
    for arm in ['rl_mdp', 'a2d_compact', 'rl_compact', 'ail_compact', 'a2d_image', 'ail_image', 'rl_asym', 'rl_image',
                'a2d_q', 'a2d_noq']:
        ms = load(env, arm)
        if not ms:
            continue
        main = 'expert_sto/ret' if arm == 'rl_mdp' else 'learner_sto/ret'
        seeds, r = fin(ms, main)
        d = {'seeds': seeds, 'ret_last5': r.tolist(), 'ret': mean_se(r), 'n_iters': ms[seeds[0]]['iters'][-1]}
        for k in ['learner_sto/kl_clean', 'proj_loss', 'learner_det/ret', 'learner_sto/haz_rate', 'learner_sto/press_rate']:
            if k in ms[seeds[0]]['curves']:
                d[k] = mean_se(fin(ms, k)[1])
        nan = any(not np.all(np.isfinite(np.asarray(ms[s]['curves'][main], float))) for s in seeds)
        d['finite'] = not nan
        rep.setdefault(env, {})[arm] = d
        mdp, pomdp = OPT[env]
        kl = d.get('learner_sto/kl_clean', (np.nan, None))[0]
        pl = d.get('proj_loss', (np.nan, None))[0]
        rows.append(f"| {env} | {arm} | {d['n_iters']} | {len(seeds)} | {d['ret'][0]:.2f} ± {d['ret'][1] or 0:.2f} | "
                    f"{d['ret'][0] - (mdp + 1):.2f} | {kl:.2e} | {pl:.2e} |")

# official reference
ref = json.load(open('runs/official_ref/ref_curves.json')) if os.path.exists('runs/official_ref/ref_curves.json') else {}
refsum = {}
for env, dd in ref.items():
    for mode, runs in dd.items():
        key = 'mdp_sto' if mode.startswith('RL_state') else 'pomdp_sto'
        vals = [np.mean(c[key][-5:]) for c in runs.values() if c['it'] and c['it'][-1] >= 60]
        projs = [np.mean(c['proj'][-5:]) for c in runs.values() if c['it'] and c['it'][-1] >= 60]
        refsum[f'{env}/{mode}'] = {'final': vals, 'proj': projs, 'n': len(vals),
                                   'last_it': [c['it'][-1] for c in runs.values()]}
rep['official_ref'] = refsum


def g(env, arm, k='ret'):
    return rep.get(env, {}).get(arm, {}).get(k, (np.nan, None))[0]


crit = {}
crit['1_all_finite'] = all(v['finite'] for e in ['FL', 'TD', 'TD1'] for v in rep.get(e, {}).values())
crit['2_a2d_compact_near_opt'] = bool(g('FL', 'a2d_compact') >= 2.47 and g('TD', 'a2d_compact') >= 1.0)
crit['2_a2d_gt_ail'] = bool(g('FL', 'a2d_compact') > g('FL', 'ail_compact') and g('TD', 'a2d_compact') > g('TD', 'ail_compact'))
crit['2_a2d_kl_lt_ail'] = bool(g('FL', 'a2d_compact', 'learner_sto/kl_clean') < g('FL', 'ail_compact', 'learner_sto/kl_clean')
                               and g('TD', 'a2d_compact', 'learner_sto/kl_clean') < g('TD', 'ail_compact', 'learner_sto/kl_clean'))
crit['3_ail_order'] = bool(-60 <= g('FL', 'ail_compact') <= -15 and -112 <= g('TD', 'ail_compact') <= -28)
crit['4_a2d_image_gt_ail_image'] = bool(g('FL', 'a2d_image') > g('FL', 'ail_image') and g('TD', 'a2d_image') > g('TD', 'ail_image'))
crit['4_rl_image_below_opt'] = bool(all(g(e, a) < OPT[e][1] - 0.05 * abs(OPT[e][1]) for e in ['FL', 'TD'] for a in ['rl_image', 'rl_asym']))
crit['5_td1_q_gt_noq'] = bool(g('TD1', 'a2d_q') > g('TD1', 'a2d_noq'))
crit['5_td1_noq_near_-42'] = bool(abs(g('TD1', 'a2d_noq') + 42) <= 0.2 * 42)
for env, arm, mode in [('FL', 'a2d_compact', 'A2D_partial_state'), ('TD', 'a2d_compact', 'A2D_partial_state'),
                       ('FL', 'ail_compact', 'AD_partial_state'), ('TD', 'ail_compact', 'AD_partial_state')]:
    rs = refsum.get(f'{env}/{mode}', {}).get('final', [])
    if rs:
        ours = g(env, arm)
        lo, hi = min(rs), max(rs)
        crit[f'6_{env}_{arm}_vs_official'] = bool((lo <= ours <= hi) or abs(ours - np.mean(rs)) <= 0.2 * abs(np.mean(rs)))
rep['criteria'] = crit
rep['PASS'] = bool(all(crit.values()))
# paired directions (descriptive at small scale)
pairs = {}
for env in ['FL', 'TD']:
    for a, b in [('a2d_compact', 'ail_compact'), ('a2d_image', 'ail_image'), ('a2d_image', 'rl_image'), ('a2d_image', 'rl_asym')]:
        A, B = rep.get(env, {}).get(a), rep.get(env, {}).get(b)
        if A and B:
            common = [s for s in A['seeds'] if s in B['seeds']]
            xa = [A['ret_last5'][A['seeds'].index(s)] for s in common]
            xb = [B['ret_last5'][B['seeds'].index(s)] for s in common]
            if len(common) >= 2:
                pc = paired_compare(xa, xb)
                pairs[f'{env}: {a} - {b}'] = {k: pc[k] for k in ('diff_mean', 'diff_se', 't95', 't_p', 'wins', 'n')}
rep['paired'] = pairs
json.dump(rep, open(f'{R}/report.json', 'w'), indent=1, default=float)
print('| env | arm | iters | seeds | final raw (mean ± s.e.) | R_norm | KL clean | F(ψ) train |')
print('|---|---|---|---|---|---|---|---|')
print('\n'.join(rows))
print(json.dumps({'criteria': crit, 'PASS': rep['PASS']}, indent=1))
print(json.dumps(pairs, indent=1, default=float))
print(json.dumps(refsum, indent=1, default=float))
