"""full-scale investigation: CleanRL's LSTM logic on masked CartPole with detail L3 OFF (no (h, c) reset at done, in
rollout and training alike, since both go through get_states). Mechanical substitution on
harness/cleanrl_ppo_lstm_maskedcartpole.py: '(1.0 - d).view(1, -1, 1) * lstm_state[i]' -> 'lstm_state[i]'.
Prints sha256 of input and output."""
import hashlib
import os

H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
src = open(os.path.join(H, "harness", "cleanrl_ppo_lstm_maskedcartpole.py")).read()
out = src
for i in (0, 1):
    old = f"(1.0 - d).view(1, -1, 1) * lstm_state[{i}]"
    assert out.count(old) == 1, old
    out = out.replace(old, f"lstm_state[{i}]")
p = os.path.join(H, "harness", "cleanrl_ppo_lstm_maskedcartpole_noreset.py")
open(p, "w").write(out)
print("cleanrl_ppo_lstm_maskedcartpole.py", hashlib.sha256(src.encode()).hexdigest())
print("cleanrl_ppo_lstm_maskedcartpole_noreset.py", hashlib.sha256(out.encode()).hexdigest())
