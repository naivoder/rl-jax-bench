"""Build harness/cleanrl_ppo_shared.py = CleanRL@fe8d8a0 ppo.py with ONLY its Agent class replaced by the Agent class of
the post's ppo_shared.py@fbef824 (the post's own file targets gym 0.21 and does not run on gymnasium 0.29.1;
 fallback, labeled). Mechanical text substitution; prints the sha256 of inputs and output."""
import hashlib
import os

H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
src = open(os.path.join(H, "cleanrl", "cleanrl", "ppo.py")).read()
sh = open(os.path.join(H, "ppo-implementation-details", "ppo_shared.py")).read()


def agent_block(text):
    i = text.index("class Agent(nn.Module):")
    j = text.index('if __name__ == "__main__":')
    return text[i:j]


out = src.replace(agent_block(src), agent_block(sh))
assert out != src
p = os.path.join(H, "harness", "cleanrl_ppo_shared.py")
open(p, "w").write(out)
for name, t in [("ppo.py", src), ("ppo_shared.py", sh), ("cleanrl_ppo_shared.py", out)]:
    print(name, hashlib.sha256(t.encode()).hexdigest())
