"""Reference-curve driver for the OFFICIAL code. Calls the official tests/A2D/util.py entry points
(do_rl / do_a2d) with the official default arguments, overriding only: --verbose 1 (otherwise the
trainee optimizer never steps on Linux), --iters, --env-name, --seed, --folder-extension.
Usage (from repo/a2d_gridworld): PYTHONPATH=$PWD python <paper>/scripts/official/ref_run.py MODE ENV ITERS SEED
  MODE in {a2d_compact, rl_mdp, ail_compact}; ail_compact trains its RL(MDP) expert first (as RunA2DExperiments).
"""
import sys, os
mode, env, iters, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lam = '0.9' if 'Lava' in env else '0.95'
sys.argv = [sys.argv[0], '--verbose', '1', '--iters', str(iters + 1), '--env-name', env, '--seed', str(seed),
            '--lambda-a2d', lam, '--folder-extension', f'ref_{mode}_{seed}']
import numpy as np, torch
np.random.seed(seed); torch.manual_seed(seed)          # the official code never seeds; we do, for our record only
from tests.A2D.A2Dagger_arguments import get_args
from tests.A2D.util import make_logging, do_rl, do_a2d
args = get_args()
args, tee = make_logging(args)
expert_args, _ = do_rl(args, _render_type='state', _return_type='state', _frame_stack=1, _run_rl=False)
args.expert_location = expert_args.log_dir
expert_args.expert_location = args.expert_location
if mode in ('rl_mdp', 'ail_compact'):
    expert_args.save_as_expert = True
    do_rl(args=expert_args, _render_type='state', _return_type='state', _frame_stack=1)
if mode == 'a2d_compact':
    do_a2d(args=args, _mode='A2D', _render_type='partial_state', _frame_stack=1, _cotrain=args.cotrain)
if mode == 'ail_compact':
    do_a2d(args=args, _mode='AD', _render_type='partial_state', _frame_stack=1, _cotrain=False)
print('REF_DONE')
