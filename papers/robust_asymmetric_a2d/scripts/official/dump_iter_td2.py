"""Oracle dump #3 (TD2 main-args A2D; derived from dump_iter.py): WHOLE training iterations of the OFFICIAL code (plai-group/a2d @368aaf5), per method,
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
    PYTHONPATH=$PWD $OFFICIAL_ROOT/venv/bin/python <paper>/scripts/official/dump_iter_td2.py <out.npz>
then compact_fixture.py <out.npz> tests/data/official_iter_g3.npz
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
TD2 = 'MiniGrid-TigerDoorEnvOSC-v2'
# A2D main args on a TD-OSC layout (Fig. B.2 lambda sweep), 3 iterations from beta = 1, for
# lambda = 0.25 and 1.0 (official launch_lambda_sweep.sh: --lambda-rl L --lambda-a2d L).
CASES.append(('a2d_td2_l025', TD2, 'A2D', 'partial_state', dict(lambda_a2d=0.25, lambda_rl=0.25), 31, [A2D_KW] * 3,
              beta_seq(0.8, 3)))
CASES.append(('a2d_td2_l100', TD2, 'A2D', 'partial_state', dict(lambda_a2d=1.0, lambda_rl=1.0), 32, [A2D_KW] * 3,
              beta_seq(0.8, 3)))

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

np.savez_compressed(OUT_PATH, **OUT)
print('saved', OUT_PATH, len(OUT), flush=True)
