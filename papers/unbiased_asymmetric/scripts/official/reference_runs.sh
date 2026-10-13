#!/bin/sh
# Authors'-code reference curves (2M steps, CPU), run one after another. Paper settings (Table 2) via
# run_official.sh; the official argparse defaults via run_official_defaults.sh. Logs go to $OFFICIAL_ROOT/runs/<out>.log,
# which src/eval.py reads. The 3rd seed (0) of the paper-setting runs and the long Heaven-Hell-3 asym-s runs use the
# same scripts (see README, "Checked against").
HERE=$(cd "$(dirname "$0")" && pwd)
R=${OFFICIAL_ROOT:-$HERE/../../external/official}/runs
mkdir -p "$R"
S5=POMDP-shopping_5-episodic-v1; H3=POMDP-heavenhell_3-episodic-v0
while read -r line; do
  out=$(echo "$line" | awk '{print $NF}')
  if [ -s "$R/$out.log" ]; then echo "skip $out (log exists)"; continue; fi
  sh $HERE/$line > "$R/$out.log" 2>&1
done <<J
run_official.sh $S5 asym-a2c 0.001 0.0003 3.0 1 2000000 shopping5_asym-a2c_s1
run_official.sh $S5 asym-a2c-state 0.001 0.001 10.0 1 2000000 shopping5_asym-a2c-state_s1
run_official.sh $S5 a2c 0.001 0.0003 3.0 1 2000000 shopping5_a2c_s1
run_official.sh $S5 asym-a2c 0.001 0.0003 3.0 2 2000000 shopping5_asym-a2c_s2
run_official.sh $S5 asym-a2c-state 0.001 0.001 10.0 2 2000000 shopping5_asym-a2c-state_s2
run_official.sh $S5 a2c 0.001 0.0003 3.0 2 2000000 shopping5_a2c_s2
run_official.sh $H3 asym-a2c 0.001 0.001 0.1 1 2000000 heavenhell3_asym-a2c_s1
run_official.sh $H3 asym-a2c-state 0.001 0.001 1.0 1 2000000 heavenhell3_asym-a2c-state_s1
run_official.sh $H3 asym-a2c 0.001 0.001 0.1 2 2000000 heavenhell3_asym-a2c_s2
run_official.sh $H3 asym-a2c-state 0.001 0.001 1.0 2 2000000 heavenhell3_asym-a2c-state_s2
run_official_defaults.sh $S5 asym-a2c 0 2000000 defaults_shopping5_asym-a2c_s0
run_official_defaults.sh $S5 asym-a2c-state 0 2000000 defaults_shopping5_asym-a2c-state_s0
J
