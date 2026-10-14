"""Record OFFICIAL whole iterations N_PRE+1 .. N_PRE+K from a TRAINED mid-run TD2
state (main args, lambda L), with every random draw recorded (dump_iter.py hooks: value shuffles, replay-buffer epoch
orders, TRPO Fisher subsample) and every rollout batch, plus all parameters after each iteration. The pre-phase is the
official do_a2d run (deterministic). Official source untouched. The fixture tests/data/official_trained_td2_lam05.npz
was recorded with SEED 1, LAM 0.5, N_PRE 20, K 4 and trimmed losslessly to uint8 / int16 (trimming step not shipped).
Usage (from repo/a2d_gridworld): PYTHONPATH=$PWD $OFFICIAL_ROOT/venv/bin/python <this> SEED LAM N_PRE K OUT.npz"""
import sys
import copy
seed, lam, n_pre, n_rec, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
env = sys.argv[6] if len(sys.argv) > 6 else 'MiniGrid-TigerDoorEnvOSC-v2'
EXTRA = sys.argv[7:]
sys.argv = [sys.argv[0], '--iters', str(n_pre + n_rec + 1), '--env-name', env, '--seed', str(seed),
            '--lambda-a2d', lam, '--lambda-rl', lam, '--folder-extension', f'recheck_rp_{lam}_{seed}_{env[-2:]}',
            '--verbose', '1'] + EXTRA
import numpy as np   # noqa: E402
import torch         # noqa: E402
torch.set_num_threads(1)
np.random.seed(seed); torch.manual_seed(seed)
from tests.A2D.A2Dagger_arguments import get_args   # noqa: E402
from tests.A2D.util import make_logging, do_rl, do_a2d   # noqa: E402
import a2d.A2Dagger as A2Dagger   # noqa: E402

STEP_KW = {'sampling_dist': 'mixture', 'networks_to_update': ['expert', 'learner_val'], 'verbose': 1}
OUT = {}
REC = {'vq': [], 'rb': [], 'sel': []}
_shuf, _choice = np.random.shuffle, np.random.choice
RECORDING = [False]


def put(k, v):
    OUT[k] = np.array(v.detach().numpy() if torch.is_tensor(v) else v, copy=True)


def shuffle(x):
    _shuf(x)
    if RECORDING[0]:
        c = sys._getframe(1).f_code.co_name
        if c in ('value_step', 'q_step'):
            REC['vq'].append(np.array(x))
        elif c == '_update_epoch_sampling_order':
            REC['rb'].append(np.array(x))
        else:
            raise RuntimeError(c)


def choice(*a, **k):
    o = _choice(*a, **k)
    if RECORDING[0] and sys._getframe(1).f_code.co_name == 'trpo_step':
        REC['sel'].append(np.array(o))
    return o


np.random.shuffle, np.random.choice = shuffle, choice


def nets(tr, prefix):
    for k in ('expert_pol', 'expert_val', 'learner_pol', 'learner_val'):
        for name, p in getattr(tr, k).named_parameters():
            put(f'{prefix}/{k}/{name}', p)


def my_a2d(self, verbose=True, log_it=True):
    self.beta = 1.0 if self.beta_decay != 0 else 0.0
    for i in range(1, n_pre + 1):
        self.step(i, **STEP_KW)
        self.beta *= self.beta_decay
        if self.beta < 0.001:
            self.beta = 0.0
    nets(self, 'pre')
    for k, o in self.optims.items():
        if o is None:
            continue
        st = o.state_dict()['state']
        for i, (name, p) in enumerate(getattr(self, k).named_parameters()):
            if i not in st:
                continue
            put(f'pre/opt/{k}/{name}/m', st[i]['exp_avg']); put(f'pre/opt/{k}/{name}/v', st[i]['exp_avg_sq'])
            put(f'pre/opt/{k}/step', float(st[i]['step']))
    rb = self.replay_buffer
    put('pre/buf/states', rb.states); put('pre/buf/obs', rb.obs)
    put('pre/buf/active_length', int(rb.active_length)); put('pre/buf/counter', int(rb.counter))
    put('pre/damping', float(self.params.damping))
    orig = self.get_data
    store = []

    def get_data(*a, **k):
        b = orig(*a, **k)
        store.append(dict(S=np.asarray(b.state), O=torch.stack(b.stored_observe).reshape(len(b.state), -1).numpy(),
                          A=np.asarray(b.action).reshape(-1), R=np.asarray(b.reward, dtype=np.float64),
                          ND=np.asarray(b.mask)))
        return b
    self.get_data = get_data
    RECORDING[0] = True
    for j in range(n_rec):
        for kk in REC:
            REC[kk] = []
        put(f'it{j}/beta', float(self.beta))
        self.step(n_pre + 1 + j, **STEP_KW)
        for kk, v in store[j].items():
            put(f'it{j}/batch/{kk}', v)
        put(f'it{j}/vq', np.stack(REC['vq']))
        put(f'it{j}/rb_n', len(REC['rb']))
        for q, r in enumerate(REC['rb']):
            put(f'it{j}/rb{q}', r)
        put(f'it{j}/sel', np.stack(REC['sel']))
        put(f'it{j}/proj_loss', float(self.current_projection_loss))
        put(f'it{j}/damping', float(self.params.damping))
        nets(self, f'it{j}')
        self.beta *= self.beta_decay
        if self.beta < 0.001:
            self.beta = 0.0
    np.savez(out, **OUT)
    print('REPLAY_DUMP_DONE')


A2Dagger.AdpAsymDagger.A2D = my_a2d
args = get_args()
args, tee = make_logging(args)
expert_args, _ = do_rl(args, _render_type='state', _return_type='state', _frame_stack=1, _run_rl=False)
args.expert_location = expert_args.log_dir
expert_args.expert_location = args.expert_location
do_a2d(args=args, _mode='A2D', _render_type='partial_state', _frame_stack=1, _cotrain=args.cotrain)
