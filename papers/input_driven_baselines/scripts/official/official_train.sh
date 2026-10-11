#!/bin/sh
# Run one of the authors' unmodified trainers (load_balance_actor_critic_train.py = state baseline,
# load_balance_actor_multi_critic_train.py = 10 value networks) through the TF compat shim, in the authors' venv.
#   sh scripts/official/official_train.sh <trainer.py> <outdir> [extra trainer flags ...]
# The reference run used seed 42 for 2,500 iterations with the paper's environment (pareto 1.5 / 100 sizes,
# interval 55, 500 jobs); its remaining hyperparameter flags were not recorded. The README-config runs used
# --seed <s> --num_ep 1001 and the trainer defaults. The authors' master process does not exit by itself: stop it
# once <outdir>/m/test_performance.npy holds the tests you need.
OFFICIAL_ROOT=${OFFICIAL_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)/external/official"}
S=$1; O=$2; shift 2
mkdir -p "$O/m"
R="0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05"
"$OFFICIAL_ROOT/.venv/bin/python" -u "$(dirname "$0")/run_official.py" "$S" \
  --num_workers 10 --service_rates $R --result_folder "$O/" --model_folder "$O/m/" "$@" > "$O/log.txt" 2>&1
