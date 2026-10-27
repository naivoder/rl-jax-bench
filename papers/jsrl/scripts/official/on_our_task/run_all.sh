#!/bin/sh
# Third-party JSRL on our task, sequentially (CPU). Needs `sh scripts/official/setup.sh` first.
# Work directory: runs/on_our_task (inputs and outputs of every script in this folder live there).
# PY = project venv, REF = reference venv with their package on PYTHONPATH.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PAPER=$(cd "$HERE/../../.." && pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PAPER/external/official}
PY=${PY:-"uv run python"}
REF=${REF:-"$OFFICIAL_ROOT/.venv/bin/python"}
export OFFICIAL_ROOT PYTHONPATH=$OFFICIAL_ROOT/jumpstart-rl/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$PAPER/runs/on_our_task/ref" && cd "$PAPER/runs/on_our_task"

JAX_PLATFORMS=cpu $PY "$HERE/export_guide.py"        # guide_breakout_np.npz, guide_ref_logits.npz
$REF "$HERE/check_guide_port.py"                     # guide port: logits and true mean on the original MinAtar
JAX_PLATFORMS=cpu $PY "$HERE/nstep_ours.py"          # nstep_stream.npz
$REF "$HERE/nstep_theirs.py"                         # nstep_parity.json
$REF "$HERE/probe_counter.py"                        # issues 16, 17, 18
# OR-1 (as-is, 10-step returns): 5 seeds x 3 arms x 200 k steps; then the stock 1-step stack, 3 seeds x 100 k
for s in 0 1 2 3 4; do for a in jsrl_paper scratch jsrl_default; do $REF "$HERE/ref_run.py" $a $s 2e5 ref 10; done; done
for s in 0 1 2; do for a in jsrl_paper scratch; do $REF "$HERE/ref_run.py" $a $s 1e5 ref 1; done; done
# P-A: roll-in from env step 0, first 40 k steps (evaluation every 2,000 steps)
for s in 0 1 2 3 4; do EVAL_EVERY=2000 $REF "$HERE/ref_run.py" jsrl_paper_g0 $s 4e4 ref 10; done
# P-B: our code with a 6,000-step guide-off warm-up (5 seeds from 0, 20 chunks)
JAX_PLATFORMS=cpu $PY "$HERE/probe_ours_warmup.py" 5 0 20 6000 probe_ours_warmup.json
$PY "$HERE/analyze.py"                               # analysis.json
