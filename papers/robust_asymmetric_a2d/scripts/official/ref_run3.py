"""Reference-run driver #3 for the OFFICIAL code: A2D(Compact) through the official util.do_a2d entry point
with EXTRA official command-line flags passed through unchanged (lambda sweep of Fig. B.2 = official
launch_lambda_sweep.sh: --lambda-rl L --lambda-a2d L on MiniGrid-TigerDoorEnvOSC-v2; Q experiment of Fig. B.1 = the
Q_arguments values as flags). Variant 'fixed' only (--verbose 1). Source untouched.
TAG 'rlmdp*' runs RL(MDP) alone (stuck-seed frequency check on Tiger Door).
Usage (from repo/a2d_gridworld): PYTHONPATH=$PWD python <paper>/scripts/official/ref_run3.py TAG ENV ITERS SEED [official flags...]"""
import sys
tag, env, iters, seed, extra = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
lam = '0.9' if 'Lava' in env else '0.95'
sys.argv = [sys.argv[0], '--iters', str(iters + 1), '--env-name', env, '--seed', str(seed), '--lambda-a2d', lam,
            '--folder-extension', f'ref3_{tag}_{seed}', '--verbose', '1'] + extra
import numpy as np   # noqa: E402
import torch         # noqa: E402
torch.set_num_threads(1)
np.random.seed(seed); torch.manual_seed(seed)
from tests.A2D.A2Dagger_arguments import get_args   # noqa: E402
from tests.A2D.util import make_logging, do_rl, do_a2d   # noqa: E402
args = get_args()
print('[ref_run3] tag', tag, 'extra', extra, 'lambda_a2d', args.lambda_a2d, 'lambda_rl', args.lambda_rl,
      'USE_Q', args.USE_Q, 'verbose', args.verbose, flush=True)
args, tee = make_logging(args)
expert_args, _ = do_rl(args, _render_type='state', _return_type='state', _frame_stack=1, _run_rl=False)
args.expert_location = expert_args.log_dir
expert_args.expert_location = args.expert_location
if tag.startswith('rlmdp'):          # RL(MDP) only (TRPO on the omniscient state), official do_rl entry point
    do_rl(args=expert_args, _render_type='state', _return_type='state', _frame_stack=1)
    print('REF_DONE')
    sys.exit(0)
do_a2d(args=args, _mode='A2D', _render_type='partial_state', _frame_stack=1, _cotrain=args.cotrain)
print('REF_DONE')
