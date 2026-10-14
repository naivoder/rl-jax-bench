"""Analytic FLOP estimate per training iteration per seed for each arm (dominant terms only; +-30 %).
XLA's HLO cost analysis was tried first and rejected: it counts a while/scan body once, not once per trip, so the
25-epoch value fit, the CG loop and the rollout scan vanish (it reported 0.08 GFLOP/iteration for A2D(Compact)).
Counts are forward-pass equivalents (fwd+bwd = 3 fwd; a Fisher-vector product = 5 fwd) times MACs per forward
(src/model.py shapes), x2 for FLOPs. Writes runs/full/flops.json."""
import json

N, BUF, VAL_EP, PROJ_EP, CG, FISH, LS = 2000, 5000, 25, 2, 10, 200, 2   # official/paper settings; LS = typical
# line-search trials (accepted at the first or second trial in our logs)


def mlp(d, out, h=64):
    return d * h + h * h + h * out


conv = 20 * 20 * 32 * 27 + (18 * 18 + 16 * 16 + 14 * 14) * 32 * 288 + 14 * 14 * 32 * 64   # PixelEncoder MACs
IMG = conv + mlp(64, 4)
S, B = mlp(30, 4), mlp(30, 4)      # TD expert state (30-d) / compact-partial (30-d); FL (34/25-d) is within 10 %
EVAL = 2 * N / 5                   # two eval lanes (sto, det) of >= 2000 steps every 5 iterations


def trpo(f):
    return f * (3 * N + CG * FISH * 5 + LS * N)


def value(f):
    return f * VAL_EP * N * 3


def proj(f_l, f_e):
    return PROJ_EP * BUF * (3 * f_l + f_e)


arms = {
    'rl_mdp': N * S + value(S) + trpo(S) + EVAL * S,
    'rl_compact': N * B + value(B) + trpo(B) + EVAL * B,
    'rl_image': N * IMG + value(IMG) + trpo(IMG) + EVAL * IMG,
    'rl_asym': N * IMG + value(S) + trpo(IMG) + EVAL * IMG,
    'ail_compact': N * (B + S) + proj(B, S) + EVAL * (B + S),
    'ail_image': N * (IMG + S) + proj(IMG, S) + EVAL * (IMG + S),
    'a2d_compact': N * (B + S) + 2 * N * (B + S) + value(S) + value(B) + trpo(S) + N * B + proj(B, S) + 2 * EVAL * (B + S),
    'a2d_image': N * (IMG + S) + 2 * N * (IMG + S) + value(S) + value(IMG) + trpo(S) + N * IMG + proj(IMG, S)
                 + 2 * EVAL * (IMG + S),
}
arms['a2d_q'] = arms['a2d_compact'] + value(mlp(34, 1)) * 2
out = {k: {'gflop_per_iter_per_seed': round(2 * v / 1e9, 3)} for k, v in arms.items()}
out['_macs_per_forward'] = {'image_net': IMG, 'flat_net': S}
json.dump(out, open('runs/full/flops.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
