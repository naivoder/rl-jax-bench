"""Sensitivity (F-S) and hyper-parameter parity (F-T) arms, pre-declared as P4 / P5 (README 'Results').
`python scripts/arms.py jobs` prints the job lines used by scripts/run_all.sh; report.py imports ARMS."""
import sys

# id -> (algorithms, list of key=value overrides)
FS = {
    "S1_R2": (["MCTS3-GD"], ["R_var=2.0"]),
    "S2_rb003": (["MCTS3-GD"], ["r_b=0.03"]),
    "S3_capalways": (["MCTS3-GD"], ["merge_cap=always"]),
    "S4_noDoublePredict": (["MCTS3-GD"], ["root_double_predict=false"]),
    "S5_phProduct": (["MCTS3-GD"], ["ph_mode=product"]),
    "S6_expRandom": (["MCTS3-GD"], ["expansion=random"]),
    "S7_deltaFirst": (["MCTS3-GD"], ["delta_selected=first"]),
    "S8_NCVplanner": (["MCTS3-GD"], ["plan_nx=4"]),
    "S9_gospaPos": (["MCTS3-GD"], ["gospa_space=pos"]),
    "S10_truthSmooth": (["MCTS3-GD", "Myopic-GD"], ["truth_pd=smooth"]),
    "S11_filtSmooth": (["MCTS3-GD", "Myopic-GD"], ["filt_pd=smooth"]),
    "S12_bothSmooth": (["MCTS3-GD", "Myopic-GD"], ["truth_pd=smooth", "filt_pd=smooth"]),
    "S13_clutterDisc": (["MCTS3-GD"], ["clutter_gen=disc"]),
    "S14_lbpConverged": (["MCTS3-GD"], ["lbp_mode=converged", "lbp_tol=1e-9", "lbp_max_iter=1000"]),
    "K1_kldNoSum": (["MCTS3-KLD"], ["kld_mode=official_nosum"]),
    "K2_kldBernoulli": (["MCTS3-KLD"], ["kld_mode=bernoulli"]),
}
FT = {
    "lam0.5": ["lam=0.5"], "lam1.0": ["lam=1.0"], "eps0": ["uct_eps=0"], "eps300": ["uct_eps=300"],
}
FT_ALGS = ["MCTS3-GD", "MCTS3-KLD"]


def cfg_name(base, sets):
    """the name train.py gives a flipped config, and its shard-file-safe form."""
    n = base + "+" + ",".join(sets)
    return n, n.replace("+", "_").replace("=", "-").replace(",", "_")


def jobs():
    out = []
    for sid, (algs, sets) in FS.items():
        for a in algs:
            out.append(f"CFG-O e1s {a} --set {' '.join(sets)}")
    for fid, sets in FT.items():
        for a in FT_ALGS:
            out.append(f"CFG-O e1s {a} --set {' '.join(sets)}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "jobs":
        print("\n".join(jobs()))
