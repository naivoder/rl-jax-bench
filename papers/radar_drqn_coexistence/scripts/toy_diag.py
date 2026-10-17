"""toy-check follow-up (our-code-first rule): which Table-I factor breaks DDRQN on the one-hot toy? lr x buffer, R_lin p=0.4.
Writes runs/toys/toy_diag.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import toys as G  # bootstraps the GPU
import train as TR
cfg = TR.load_cfg(os.path.join(G.ROOT, "src", "configs", "default.yaml"))
rows = []
for arm, lr, buf in [("ddrqn", 1e-3, 20000), ("ddrqn", 1e-4, 2000), ("ddqn_tr10", 1e-3, 2000)]:
    L = G.arm_config(arm, cfg).trace_len
    r = G.core_on_toy(arm, cfg, "lin", n_steps=100_000, lr=lr, buffer_traces=buf // L)
    r.update(lr=lr, buffer_transitions=buf)
    rows.append(r)
    G.metric("toy diag", arm=arm, lr=lr, buffer=buf, act_agree=round(r["act_agree_mean"], 3),
             final_policy=str(set(r["final_policy"])), adv_err=round(max(r["final_adv_err"]), 3))
json.dump(rows, open(os.path.join(G.ROOT, "runs", "toys", "toy_diag.json"), "w"), indent=1, default=float)
