#!/bin/sh
# Reference runs of the authors' library (Tonic @0e20c89, torch PPO) in its own venv, CPU only, one after another.
# Run from the paper folder; OFFICIAL_ROOT (default external/official) holds the Tonic clone and .venv-ref.
# CONFIGS defaults to "tonic paper paper_vn"; outputs go to $OFFICIAL_ROOT/ref_runs/.
export OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
CONFIGS=${CONFIGS:-"tonic paper paper_vn"}
REFPY="$OFFICIAL_ROOT/.venv-ref/bin/python"
mkdir -p "$OFFICIAL_ROOT/ref_runs"
run() {  # exp arm config seed steps
  f="$OFFICIAL_ROOT/ref_runs/$1_$2_$3_s$4.json"
  [ -f "$f" ] || OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$REFPY" -I -W ignore scripts/official/ref_run.py \
    --exp "$1" --arm "$2" --config "$3" --seed "$4" --steps "$5" --out "$f"
}
for c in $CONFIGS; do
  for a in peb_ppo_T300 ppo_T300 ppo_T1000; do for s in 0 1000 2000 3000 4000; do run hopper $a $c $s 301056; done; done
  for a in ppo_g099 ta_ppo_g099 ppo_g1 ta_ppo_g1; do for s in 0 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
    run ip $a $c $s 100000; done; done
done
