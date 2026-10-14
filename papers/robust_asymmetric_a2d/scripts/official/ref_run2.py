"""Reference-run driver #2 for the OFFICIAL code. Calls the official tests/A2D/util.py
entry points exactly as tests/A2D/RunA2DExperiments.py does, with three VARIANTS:

  shipped : every official default untouched (incl. --verbose = int(not cluster) = 0 on Linux)
  fixed   : official defaults + --verbose 1 (the trainee optimizer steps; what the paper's figures must have used)
  paper   : 'fixed' + the two paper-text settings the code does not implement:
            T = 200 episode cap (paper App. D.1; code max_steps = 100), and L2 1e-3 "applied to all networks"
            (paper App. D.1; --l2-reg parsed but never used) realised as Adam weight_decay = 1e-3 on every Adam
            optimizer (TRPO-updated policies have no optimizer to carry it). Both are monkeypatches applied here,
            the official source is not edited.

Usage (from repo/a2d_gridworld):
  PYTHONPATH=$PWD python <paper>/scripts/official/ref_run2.py MODE ENV ITERS SEED VARIANT [EXPERT_DIR]
  MODE: a2d_compact | ail_compact (trains its RL(MDP) expert first) | ail_reuse (expert from EXPERT_DIR)
        | a2d_image | ail_image_reuse | rl_image | arl_image_reuse
"""
import sys
mode, env, iters, seed, variant = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
expert_dir = sys.argv[6] if len(sys.argv) > 6 else None
assert variant in ('shipped', 'fixed', 'paper')
lam = '0.9' if 'Lava' in env else '0.95'
argv = [sys.argv[0], '--iters', str(iters + 1), '--env-name', env, '--seed', str(seed), '--lambda-a2d', lam,
        '--folder-extension', f'ref2_{variant}_{mode}_{seed}']
if variant != 'shipped':
    argv += ['--verbose', '1']
sys.argv = argv
import shutil                                            # noqa: E402
import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
torch.set_num_threads(1)
np.random.seed(seed); torch.manual_seed(seed)            # the official code never seeds; we do, for our record only

if variant == 'paper':
    import gym
    _make = gym.make

    def make_T200(*a, **k):
        e = _make(*a, **k)
        for obj in (e, e.unwrapped):
            obj.max_steps = 200
            obj._max_episode_steps = 200
        return e
    gym.make = make_T200
    _Adam = torch.optim.Adam

    class AdamL2(_Adam):
        def __init__(self, params, **kw):
            kw.setdefault('weight_decay', 1e-3)
            super().__init__(params, **kw)
    torch.optim.Adam = AdamL2

from tests.A2D.A2Dagger_arguments import get_args        # noqa: E402
from tests.A2D.util import make_logging, do_rl, do_a2d, do_advanced_rl   # noqa: E402
args = get_args()
print('[ref_run2] variant', variant, 'mode', mode, 'verbose', args.verbose, flush=True)
args, tee = make_logging(args)
expert_args, _ = do_rl(args, _render_type='state', _return_type='state', _frame_stack=1, _run_rl=False)
args.expert_location = expert_args.log_dir
expert_args.expert_location = args.expert_location
if mode == 'ail_compact':
    expert_args.save_as_expert = True
    do_rl(args=expert_args, _render_type='state', _return_type='state', _frame_stack=1)
elif mode.endswith('_reuse'):
    shutil.copy(expert_dir + '/expert.pt', args.expert_location + '/expert.pt')
    print('[ref_run2] expert copied from', expert_dir)

if mode == 'a2d_compact':
    do_a2d(args=args, _mode='A2D', _render_type='partial_state', _frame_stack=1, _cotrain=args.cotrain)
elif mode in ('ail_compact', 'ail_reuse'):
    do_a2d(args=args, _mode='AD', _render_type='partial_state', _frame_stack=1, _cotrain=False)
elif mode == 'a2d_image':
    do_a2d(args=args, _mode='A2D', _render_type='observe', _frame_stack=args.frame_stack, _cotrain=args.cotrain)
elif mode == 'ail_image_reuse':
    do_a2d(args=args, _mode='AD', _render_type='observe', _frame_stack=args.frame_stack, _cotrain=False)
elif mode == 'rl_image':
    do_rl(args=args, _render_type='observe', _return_type='observe', _frame_stack=args.frame_stack)
elif mode == 'arl_image_reuse':
    do_advanced_rl(args=args, _render_type='observe', _frame_stack=args.frame_stack, _return_type='observe',
                   _ete=False)
print('REF_DONE')
