#!/bin/sh
# Every official-code reference run behind results/ref*_summary.json, results/recheck_report.json and the README's
# "authors' code" column, sequential (the original ran up to 6 at a time, 1 thread each). Must use --verbose 1 (set by
# the drivers): with the shipped default the trainee optimizer never steps on Linux. Then collect with
# scripts/parse_official_ref.py, parse_official_ref2.py and parse_official_ref3.py.
HERE=$(cd "$(dirname "$0")" && pwd)
PAPER=$(cd "$HERE/../.." && pwd)
export OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PAPER/external/official}
R="sh $HERE/run_ref.sh"
LAVA=MiniGrid-LavaGapS7-v0; TIGER=MiniGrid-TigerDoorEnv-v0

# 1. ref_run.py: official defaults + --verbose 1 (logs in oracle/runs)
for S in 1 2 3; do
  REF_LOGS=$OFFICIAL_ROOT/oracle/runs $R ref_run.py a2d_compact_${LAVA}_s$S a2d_compact $LAVA 200 $S
  REF_LOGS=$OFFICIAL_ROOT/oracle/runs $R ref_run.py a2d_compact_${TIGER}_s$S a2d_compact $TIGER 200 $S
done
for J in "$LAVA 1" "$LAVA 2" "$TIGER 1"; do
  set -- $J
  REF_LOGS=$OFFICIAL_ROOT/oracle/runs $R ref_run.py ail_compact_$1_s$2 ail_compact $1 200 $2
done

# 2. ref_run2.py: variants shipped / fixed / paper (logs in oracle/runs2); EXPERT_DIR reuses a ref_run.py expert
while read MODE ENV IT S VAR EXP; do
  REF_LOGS=$OFFICIAL_ROOT/oracle/runs2 $R ref_run2.py ${VAR}_${MODE}_${ENV}_s$S $MODE $ENV $IT $S $VAR $EXP
done <<JOBS
a2d_compact MiniGrid-LavaGapS7-v0 200 1 paper
a2d_compact MiniGrid-LavaGapS7-v0 200 1 shipped
a2d_compact MiniGrid-LavaGapS7-v0 200 2 paper
a2d_compact MiniGrid-LavaGapS7-v0 200 2 shipped
a2d_compact MiniGrid-LavaGapS7-v0 200 3 paper
a2d_compact MiniGrid-LavaGapS7-v0 200 3 shipped
a2d_compact MiniGrid-LavaGapS7-v0 200 4 fixed
a2d_compact MiniGrid-TigerDoorEnv-v0 200 1 paper
a2d_compact MiniGrid-TigerDoorEnv-v0 200 1 shipped
a2d_compact MiniGrid-TigerDoorEnv-v0 200 2 paper
a2d_compact MiniGrid-TigerDoorEnv-v0 200 2 shipped
a2d_compact MiniGrid-TigerDoorEnv-v0 200 3 paper
a2d_compact MiniGrid-TigerDoorEnv-v0 200 3 shipped
ail_compact MiniGrid-LavaGapS7-v0 200 3 fixed
ail_compact MiniGrid-TigerDoorEnv-v0 200 2 fixed
ail_compact MiniGrid-TigerDoorEnv-v0 200 3 fixed
ail_reuse MiniGrid-LavaGapS7-v0 200 1 paper tests/results/AdaptAsymDagger/MiniGrid-LavaGapS7-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-LavaGapS7-v0 200 1 shipped tests/results/AdaptAsymDagger/MiniGrid-LavaGapS7-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-LavaGapS7-v0 200 2 paper tests/results/AdaptAsymDagger/MiniGrid-LavaGapS7-v0/ref_ail_compact_2/RL_state/2
ail_reuse MiniGrid-LavaGapS7-v0 200 2 shipped tests/results/AdaptAsymDagger/MiniGrid-LavaGapS7-v0/ref_ail_compact_2/RL_state/2
ail_reuse MiniGrid-LavaGapS7-v0 200 3 paper tests/results/AdaptAsymDagger/MiniGrid-LavaGapS7-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-LavaGapS7-v0 200 3 shipped tests/results/AdaptAsymDagger/MiniGrid-LavaGapS7-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-TigerDoorEnv-v0 200 1 paper tests/results/AdaptAsymDagger/MiniGrid-TigerDoorEnv-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-TigerDoorEnv-v0 200 1 shipped tests/results/AdaptAsymDagger/MiniGrid-TigerDoorEnv-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-TigerDoorEnv-v0 200 2 paper tests/results/AdaptAsymDagger/MiniGrid-TigerDoorEnv-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-TigerDoorEnv-v0 200 2 shipped tests/results/AdaptAsymDagger/MiniGrid-TigerDoorEnv-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-TigerDoorEnv-v0 200 3 paper tests/results/AdaptAsymDagger/MiniGrid-TigerDoorEnv-v0/ref_ail_compact_1/RL_state/1
ail_reuse MiniGrid-TigerDoorEnv-v0 200 3 shipped tests/results/AdaptAsymDagger/MiniGrid-TigerDoorEnv-v0/ref_ail_compact_1/RL_state/1
JOBS
# image reference (GPU venv venv_cuda in the original, torch 1.13.1+cu117): one A2D(Image) FL run, 30 iterations
REF_LOGS=$OFFICIAL_ROOT/oracle/runs2 $R ref_run2.py fixed_a2d_image_FL_s1 a2d_image $LAVA 30 1 fixed

# 3. ref_run3.py: extra official flags passed through (logs in oracle/runs3)
Q="--value-calc time --regularize-adv 0 --a2d-entropy-reg 0.02 --rl-entropy-reg 0.02 --max-kl 0.001"
for S in 1 2; do
  for LM in 1.0 0.25 0.5; do
    $R ref_run3.py lam${LM}_MiniGrid-TigerDoorEnvOSC-v2_s$S lam$LM MiniGrid-TigerDoorEnvOSC-v2 250 $S --lambda-rl $LM --lambda-a2d $LM
  done
  $R ref_run3.py q_MiniGrid-TigerDoorEnvOSC-v1_s$S q MiniGrid-TigerDoorEnvOSC-v1 200 $S --USE-Q 1 $Q --lambda-a2d 0.5 --beta-decay 0.0
  $R ref_run3.py noq_MiniGrid-TigerDoorEnvOSC-v1_s$S noq MiniGrid-TigerDoorEnvOSC-v1 200 $S --USE-Q 0 $Q --lambda-a2d 0.5 --beta-decay 0.0
done
for S in 3 4 5 6 7 8 9 10 11; do
  $R ref_run3.py rlmdp_${TIGER}_s$S rlmdp $TIGER 200 $S
done
B1="--USE-Q 0 $Q --beta-decay 0.0"
for J in "1.0 1" "0.25 1" "1.0 2" "0.25 2"; do
  set -- $J
  $R ref_run3.py b1lam$1_MiniGrid-TigerDoorEnvOSC-v2_s$2 b1lam$1 MiniGrid-TigerDoorEnvOSC-v2 250 $2 $B1 --lambda-rl $1 --lambda-a2d $1
done

# 4. after the log-likelihood clamp fix (logs in recheck_logs, names prefixed ref_)
export REF_LOGS=$OFFICIAL_ROOT/recheck_logs
for S in 1 2 3 4; do
  $R ref_run3.py ref_b1lam0.75_MiniGrid-TigerDoorEnvOSC-v2_s$S b1lam0.75 MiniGrid-TigerDoorEnvOSC-v2 250 $S $B1 --lambda-rl 0.75 --lambda-a2d 0.75
done
for S in 3 4; do
  $R ref_run3.py ref_b1lam1.0_MiniGrid-TigerDoorEnvOSC-v2_s$S b1lam1.0 MiniGrid-TigerDoorEnvOSC-v2 250 $S $B1 --lambda-rl 1.0 --lambda-a2d 1.0
done
for S in 1 2 3 4 5 6 7 8 9 10; do
  $R ref_run3.py ref_b1noq_MiniGrid-TigerDoorEnvOSC-v3_s$S b1noq MiniGrid-TigerDoorEnvOSC-v3 200 $S $B1 --lambda-a2d 0.5
done
for S in 1 2 3; do
  $R ref_run3.py ref_b1lam0_MiniGrid-TigerDoorEnvOSC-v2_s$S b1lam0 MiniGrid-TigerDoorEnvOSC-v2 250 $S $B1 --lambda-rl 0 --lambda-a2d 0
done
for S in 4 5 6 7; do
  $R ref_run3.py ref_a2dmain_${TIGER}_s$S a2dmain $TIGER 200 $S
done
# shipped launcher's lambda_a2d = 0.9 (ref_run3.py sets 0.95 for TD; the trailing flag overrides it)
for S in 1 2 3 4 5 6; do
  $R ref_run3.py ref_a2dmain_lam09_${TIGER}_s$S a2dmain_lam09 $TIGER 200 $S --lambda-a2d 0.9
done
