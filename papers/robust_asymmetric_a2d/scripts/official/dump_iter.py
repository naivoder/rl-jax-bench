"""Oracle dump #2: WHOLE training iterations of the OFFICIAL code (plai-group/a2d @368aaf5), per method,
plus the evaluation metric on identical (deterministic) rollouts.

Every case builds the official AdpAsymDagger exactly as tests/A2D/util.py (do_rl / do_a2d / do_advanced_rl) would,
with freshly initialized (then output-perturbed) networks, and calls the official `step()` (rollout -> RL steps ->
projection) for a few iterations. Nothing inside the official code is replaced; its randomness is only RECORDED:
  * the rollouts (states, observations, actions, rewards, masks) as returned by the official get_data();
  * every np.random.shuffle of value_step / q_step / ReplayBuffer._update_epoch_sampling_order (index permutations);
  * every np.random.choice made by trpo_step (the 10 % Fisher subsample).
The JAX side (tests/test_official_parity.py) replays those exact rollouts / permutations / subsamples through its own
Trainer.iteration and compares every network's parameters after each iteration.

Run from repo/a2d_gridworld with the isolated venv:
    PYTHONPATH=$PWD $OFFICIAL_ROOT/venv/bin/python <paper>/scripts/official/dump_iter.py <out.npz>
then compact_fixture.py <out.npz> tests/data/official_iter.npz
"""
import sys
import copy

OUT_PATH = sys.argv[1]
sys.argv = [sys.argv[0], '--verbose', '1']            # official --verbose 1 (otherwise the trainee never steps)

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

torch.set_default_dtype(torch.float64)
import a2d.environments.minigrid.gym_minigrid  # noqa: E402,F401  registers the envs
import a2d.A2Dagger as A2Dagger                         # noqa: E402
from a2d.models.models import DiscPolicy, ValueDenseNet  # noqa: E402
from a2d.util.sampler import sample_batch               # noqa: E402
from tests.A2D.A2Dagger_arguments import get_args       # noqa: E402

BASE_ARGS = get_args()
OUT = {}


def put(k, v):
    OUT[k] = np.array(v.detach().numpy() if torch.is_tensor(v) else v, copy=True)   # copy: params are updated in place


def net_params(net, prefix):
    for name, p in net.named_parameters():
        put(prefix + '/' + name, p)


# ------------------------------------------------------------------ recording hooks (record, never replace)
REC = {'vq': [], 'vq_tag': [], 'rb': [], 'sel': []}
_np_shuffle, _np_choice = np.random.shuffle, np.random.choice


def _shuffle(x):
    _np_shuffle(x)
    caller = sys._getframe(1).f_code.co_name
    if caller in ('value_step', 'q_step'):
        REC['vq'].append(np.array(x)); REC['vq_tag'].append(caller)
    elif caller == '_update_epoch_sampling_order':
        REC['rb'].append(np.array(x))
    else:
        raise RuntimeError('unexpected shuffle from ' + caller)


def _choice(*a, **k):
    out = _np_choice(*a, **k)
    if sys._getframe(1).f_code.co_name == 'trpo_step':
        REC['sel'].append(np.array(out))
    return out


np.random.shuffle, np.random.choice = _shuffle, _choice


def reset_rec():
    for k in REC:
        REC[k] = []


# ------------------------------------------------------------------ case construction (mirrors tests/A2D/util.py)
def make_args(env, mode, render_type, overrides):
    a = copy.deepcopy(BASE_ARGS)
    a.env_name = env
    for k, v in overrides.items():
        setattr(a, k, v)
    a.mode = mode
    a.render_type = render_type
    a.frame_stack = 1
    if mode == 'RL' or mode == 'ARL':
        a.entropy_reg = a.rl_entropy_reg; a.cotrain = False
        a.lambda_ = a.lambda_rl
    if mode == 'A2D':
        a.cotrain = True
        a.entropy_reg = a.a2d_entropy_reg
        a.lambda_ = a.lambda_a2d
    if mode == 'AD':
        a.cotrain = False
    a.max_kl_final = a.max_kl
    return a


def perturb(net, gain):
    with torch.no_grad():
        net.head[-1].weight.mul_(gain)


def build(case, env, mode, render_type, overrides, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    args = make_args(env, mode, render_type, overrides)
    tr = A2Dagger.AdpAsymDagger(args, img_stack=1, img_size=args.resize)
    obs_shape = tr.env.my_render().shape
    st_dim = tr.env.state.shape[0]
    ep = ev = lp = lv = None
    if mode in ('A2D', 'AD') or (mode == 'RL' and render_type == 'state'):
        if mode == 'RL':                                 # do_rl: DiscPolicy(observation_space.shape, ...)
            ep, ev = DiscPolicy(obs_shape, 4), ValueDenseNet(obs_shape)
        else:                                            # get_expert_network -> init_mdp_net (int input size)
            ep, ev = DiscPolicy(st_dim, 4), ValueDenseNet(st_dim)
        perturb(ep, 60.0); perturb(ev, 300.0)
    if render_type != 'state':
        lp = DiscPolicy(obs_shape, 4)
        lv = ValueDenseNet(st_dim) if mode == 'ARL' else ValueDenseNet(obs_shape)
        if mode == 'ARL':
            lv.ASYMMETRIC = True
        perturb(lp, 40.0); perturb(lv, 300.0)
    tr.define_networks(ep, ev, lp, lv)
    tr.logging = lambda *a, **k: None
    for nm in ('expert_pol', 'expert_val', 'learner_pol', 'learner_val', 'expert_q', 'learner_q'):
        net = getattr(tr, nm, None)
        if net is not None:
            if nm.endswith('_q'):
                perturb(net, 3.0)
            net_params(net, f'{case}/init/{nm}')
    put(f'{case}/obs_dim', int(np.prod(obs_shape))); put(f'{case}/state_dim', st_dim)
    return tr


def record_batches(tr):
    orig = tr.get_data
    store = []

    def get_data(*a, **k):
        b = orig(*a, **k)
        store.append(dict(S=np.asarray(b.state), O=torch.stack(b.stored_observe).reshape(len(b.state), -1).numpy(),
                          A=np.asarray(b.action).reshape(-1), R=np.asarray(b.reward, dtype=np.float64),
                          ND=np.asarray(b.mask)))
        return b
    tr.get_data = get_data
    return store


def dump_iters(case, tr, store, n_pre, it_args_list, betas_before):
    """it_args_list: list of step() kwargs, one per iteration; betas recorded before each step."""
    for i, kw in enumerate(it_args_list):
        reset_rec()
        put(f'{case}/it{i}/beta', float(tr.beta))
        tr.step(i + 1, **kw)
        b = store[n_pre + i]
        for k, v in b.items():
            put(f'{case}/it{i}/batch/{k}', v)
        put(f'{case}/it{i}/vq', np.stack(REC['vq']) if REC['vq'] else np.zeros((0, 2000), int))
        put(f'{case}/it{i}/vq_tag', np.array([t == 'q_step' for t in REC['vq_tag']], dtype=np.int8))
        put(f'{case}/it{i}/rb_n', len(REC['rb']))
        for j, r in enumerate(REC['rb']):
            put(f'{case}/it{i}/rb{j}', r)
        put(f'{case}/it{i}/sel', np.stack(REC['sel']) if REC['sel'] else np.zeros((0, 200), int))
        put(f'{case}/it{i}/proj_loss', float(tr.current_projection_loss))
        for nm in ('expert_pol', 'expert_val', 'learner_pol', 'learner_val', 'expert_q', 'learner_q'):
            net = getattr(tr, nm, None)
            if net is not None:
                net_params(net, f'{case}/it{i}/{nm}')
        # the official A2D() loop's beta update (applied after step); AD() zeroes beta
        tr.beta = betas_before[i + 1] if i + 1 < len(betas_before) else tr.beta


FL, TD, TD1 = 'MiniGrid-LavaGapS7-v0', 'MiniGrid-TigerDoorEnv-v0', 'MiniGrid-TigerDoorEnvOSC-v1'
A2D_KW = dict(sampling_dist='mixture', networks_to_update=['expert', 'learner_val'], verbose=1)


def beta_seq(decay, n):
    b, out = (1.0 if decay > 0 else 0.0), []
    for _ in range(n + 1):
        out.append(b)
        b *= decay
        if b < 0.001:
            b = 0.0
    return out


CASES = []

# 1. A2D, FL, main args, 3 iterations from beta = 1 (exercises beta-mixture, IS weights, FIFO wrap at 6000 > 5000)
CASES.append(('a2d_fl', FL, 'A2D', 'partial_state', dict(lambda_a2d=0.9), 11, [A2D_KW] * 3, beta_seq(0.8, 3)))
# 2. A2D, TD, beta = 0 from the start (beta_decay 0): pure trainee-critic advantage
CASES.append(('a2d_td_b0', TD, 'A2D', 'partial_state', dict(lambda_a2d=0.95, beta_decay=0.0), 12, [A2D_KW] * 2,
              beta_seq(0.0, 2)))
# 3. A2D + Q, TD1, official Q_arguments (beta 0, lambda .5, KL 1e-3, entropy .02 on the surrogate, MC targets)
QARGS = dict(USE_Q=1, value_calc='time', regularize_adv=0, a2d_entropy_reg=0.02, rl_entropy_reg=0.02, max_kl=0.001,
             lambda_a2d=0.5, beta_decay=0.0)
CASES.append(('a2dq_td1', TD1, 'A2D', 'partial_state', QARGS, 13, [A2D_KW] * 2, beta_seq(0.0, 2)))
# 4. A2D + Q with a beta-mixture (beta 1 -> 0.8): exercises beta Q_m + (1 - beta) Q_p - V^beta
CASES.append(('a2dq_td1_mix', TD1, 'A2D', 'partial_state', dict(QARGS, beta_decay=0.8), 14, [A2D_KW] * 2,
              beta_seq(0.8, 2)))
# 5. RL(MDP) on TD
RLM_KW = dict(proj_update=False, sampling_dist='expert', store_=False, networks_to_update=['expert'], verbose=1)
CASES.append(('rlmdp_td', TD, 'RL', 'state', {}, 15, [RLM_KW] * 2, [1.0, 1.0, 1.0]))
# 6. RL on the compact partial observation (symmetric), FL
RLP_KW = dict(proj_update=False, sampling_dist='learner', store_=False, networks_to_update=['learner'], verbose=1)
CASES.append(('rlpo_fl', FL, 'RL', 'partial_state', {}, 16, [RLP_KW] * 2, [0.0, 0.0, 0.0]))
# 7. ARL (asymmetric value on s), compact partial observation, TD
CASES.append(('arl_td', TD, 'ARL', 'partial_state', {}, 17, [RLP_KW] * 2, [0.0, 0.0, 0.0]))

for case, env, mode, rt, ov, seed, kws, betas in CASES:
    tr = build(case, env, mode, rt, ov, seed)
    if mode == 'RL':
        if rt == 'state':
            delattr(tr, 'learner_pol'); delattr(tr, 'learner_val')
        else:
            delattr(tr, 'expert_pol'); delattr(tr, 'expert_val')
    if mode == 'ARL':
        delattr(tr, 'expert_pol'); delattr(tr, 'expert_val')
    tr.beta = betas[0]
    store = record_batches(tr)
    dump_iters(case, tr, store, 0, kws, betas)
    print('case', case, 'done; final beta', tr.beta, flush=True)

# 8. AIL (official AD with cotrain=False): pre-training step (5000 expert samples + 1 projection epoch), then two
#    iterations (beta = 1, then 0), as AD() does.
case = 'ail_fl'
tr = build(case, FL, 'AD', 'partial_state', {}, 18, )
store = record_batches(tr)
reset_rec()
tr.beta = torch.tensor(1.)
tr.pretrain(True)
b = store[0]
for k, v in b.items():
    put(f'{case}/pre/batch/{k}', v)
put(f'{case}/pre/rb_n', len(REC['rb']))
for j, r in enumerate(REC['rb']):
    put(f'{case}/pre/rb{j}', r)
net_params(tr.learner_pol, f'{case}/pre/learner_pol')
AD_KW = dict(rl_update=False, sampling_dist='mixture', verbose=1)
tr.beta = 1.0
dump_iters(case, tr, store, 1, [AD_KW] * 2, [1.0, 0.0, 0.0])
print('case ail_fl done', flush=True)

# ------------------------------------------------------------------ evaluation metric on identical rollouts
# Official logging(): sample_batch(dist, eval_=True (argmax), full_trajectory=True, batch_size=eval_batch_size).
# Deterministic policies => the rollout is a function of the per-episode reset draws only; we store the trajectory
# (store=True does not change the rewards) so the JAX side can force the same resets.
EXPERTS = {'eval_fl': 'tests/results/AdaptAsymDagger/MiniGrid-LavaGapS7-v0/ref_ail_compact_1/RL_state/1/expert.pt',
           'eval_td': 'tests/results/AdaptAsymDagger/MiniGrid-TigerDoorEnv-v0/ref_ail_compact_1/RL_state/1/expert.pt'}
for case, env, seed in [('eval_fl', FL, 21), ('eval_td', TD, 22)]:
    tr = build(case, env, 'A2D', 'partial_state', dict(lambda_a2d=0.9), seed)
    # a TRAINED official RL(MDP) expert (from our official reference runs) so episodes end at goal/lava with varied
    # lengths; the learner keeps its random (perturbed) init (mostly 100-step time-outs).
    tr.expert_pol.load_state_dict(torch.load(EXPERTS[case])['expert_network'])
    net_params(tr.expert_pol, f'{case}/init/expert_pol')
    a = copy.deepcopy(tr.params); a.batch_size = a.eval_batch_size
    for lane, dist in [('expert', tr.expert_sampling_dist), ('learner', tr.learner_sampling_dist)]:
        with torch.no_grad():
            rew, batch, _ = sample_batch(dist, tr.env, a, eval_=True, store=True, full_trajectory=True)
        put(f'{case}/{lane}/ret', rew)
        put(f'{case}/{lane}/S', np.asarray(batch.state)); put(f'{case}/{lane}/ND', np.asarray(batch.mask))
        put(f'{case}/{lane}/R', np.asarray(batch.reward, dtype=np.float64))
        put(f'{case}/{lane}/A', np.asarray(batch.action).reshape(-1))
    # stochastic evaluation (the column the figures plot): 30 independent official evaluations of the same expert
    sto = []
    for _ in range(30):
        with torch.no_grad():
            sto.append(sample_batch(tr.expert_sampling_dist, tr.env, a, eval_=False, store=False,
                                    full_trajectory=True)[0])
    put(f'{case}/expert_sto/rets', np.array(sto))
    print('case', case, 'done', np.mean(sto), flush=True)

np.random.shuffle, np.random.choice = _np_shuffle, _np_choice
np.savez_compressed(OUT_PATH, **OUT)
print('wrote', len(OUT), 'arrays to', OUT_PATH)
