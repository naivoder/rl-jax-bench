"""Configuration for the radar_search_track_seq reproduction (Ewers et al. 2025, arXiv 2502.13584).

Every value carries its source tag inline. The canonical per-key table is sources.yaml (source tags
are summarized in the README).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 0.05                 # paper §2.1 (50 ms); companion Table IV 5 ms -> conditional switch,
    T: int = 1000                    # INFERRED from Fig. 1a/1d Static (README),
    n_tgt: int = 3                   # INFERRED (c^2 N = 3e8 -> N = 3 at c = 10 km),
    n_a: int = 19                    # companion eq. (1): ceil(2*60/(9/sqrt2)) = 19,
    for_half_deg: float = 60.0       # companion eq. (2),
    beam_r_deg: float = 4.5          # paper §2.1: 9 deg field of view -> radius
    pd: float = 0.9                  # ASSUMED (not given),
    sig_el_deg: float = 0.05         # ASSUMED (monopulse fraction of a 9 deg beam),
    sig_az_deg: float = 0.05         # ASSUMED,
    sig_rng_m: float = 10.0          # ASSUMED,
    q_truth: float = 0.1             # ASSUMED (near-CV truth, m^2/s^3),
    q_filt: float = 1.0              # ASSUMED (filter CV noise, m^2/s^3),
    rng_lo_m: float = 10_000.0       # ASSUMED,
    rng_hi_m: float = 30_000.0       # ASSUMED,
    az_lim_deg: float = 50.0         # ASSUMED,
    el_lim_deg: float = 20.0         # ASSUMED,
    spd_lo: float = 100.0            # ASSUMED (horizontal speed m/s),
    spd_hi: float = 250.0            # ASSUMED, (used only when truth_vel = "uniform_speed")
    truth_vel: str = "prior"         # AMENDED 2026-10-17 (README): horizontal velocity ~ N(0, init_vel_sigma^2)
                                     # per axis, vz = 0 -- the initiator prior is then consistent with the truth.
                                     # "uniform_speed" = the original reading (speed U(100,250) m/s): tracks cannot be
                                     # held by any rule because the filter's velocity prior (bounded by the trace-5000
                                     # deleter, README) excludes the truth (README small-scale, scene audit)
    n_sv: int = 100                  # ASSUMED, frozen before any run,
    gamma_sv: float = 0.01           # ASSUMED, frozen before any run,
    raster: int = 48                 # companion eq. (5),
    k_max: int = 17                  # ASSUMED (7*17 = 119 ~ Flat's 1x120, paper §3),
    init_vel_sigma: float = 30.0     # ASSUMED (bounded by the trace-5000 deleter, README),
    truth_vel_sigma: float | None = None  # recheck: truth horizontal velocity SD per axis when
                                     # truth_vel = "prior". None = tied to init_vel_sigma (primary, unchanged);
                                     # a number decouples the truth from the filter prior (filter-only diagnostics)
    del_thr: float = 5000.0          # companion Table IV; trace statistic = library CovarianceBasedDeleter,
    del_stat: str = "trace"          # companion (library CovarianceBasedDeleter: trace),. "frob" = THIS paper's text
                                     # (§2.1 "Frobenius norm of the target covariance above a set threshold"), co-primary
                                     # scene since the review; the threshold 5000 was published only with trace
    n_fa_slots: int = 0              # ASSUMED 0 = no clutter (primary). >0: per-look false alarms, Binomial(n_fa_slots,
                                     # clutter_rate / n_fa_slots), uniform in the beam disc and in range ( sensitivity)
    clutter_rate: float = 0.0        # ASSUMED mean false alarms per look (paper §2.1 motivates the MTT with "clutter or
                                     # anomalies" but gives no rate); used only when n_fa_slots > 0
    gate: float = 3.0                # ASSUMED (Stone Soup tutorial convention),
    missed: float = 3.0              # ASSUMED,
    ukf_alpha: float = 0.5           # library (Stone Soup 1.9.1 UnscentedKalmanUpdater default),
    ukf_beta: float = 2.0            # library,
    ukf_kappa: float = -3.0          # library (3 - ndim_state),
    init_jacobian: str = "fd"        # library: SimpleMeasurementInitiator uses Stone Soup's forward-difference Jacobian
                                     # (its 1e-8 step floor makes C0 carry ~1e-4 relative round-off); "analytic" = exact
                                     # (jax.jacfwd), used to isolate that round-off in the oracle tests
    associator: str = "nn"           # companion Table II (greedy NearestNeighbour); "gnn" = paper §2.1 Munkres,
    gospa_c: float = 10_000.0        # INFERRED (README),
    gospa_p: int = 2                 # INFERRED, (alpha = 2 fixed, library)
    scale_range: float = 30_000.0    # ASSUMED obs scaling,
    scale_angle_deg: float = 60.0    # companion eq. (2) FoR half-width,
    scale_vel: float = 300.0         # ASSUMED,
    scale_cov: float = 5000.0        # ASSUMED (deleter threshold),
    gospa_metrics: bool = True       # compute GOSPA inside the step (eval only; training sets False for speed)
    gospa_lib: bool = False          # also compute the LIBRARY-reading GOSPA (Stone Soup tutorial pipeline: a track deleted
                                     # at t counts with its last state at t; up to LIB_KILL_SLOTS per step),. Off for
                                     # training and for the full-scale runs (they predate it)

    sigma0_mode: str = "radial"      # ASSUMED: "radial" = 95 % of the 2-D mass in the beam disc, 4.5/sqrt(2 ln 20)
                                     # = 1.84 deg; "axis" = per-axis two-sided 95 %, 4.5/1.96 = 2.30 deg ( sensitivity)
    ssv_age0: bool = False           # ASSUMED: False = SSV evaluated BEFORE the look is pushed (ages 1..N_SV);
                                     # True = after the push (ages 0..N_SV-1, the range p_max sums over), sensitivity
    ssv_norm: str = "fifo"           # ASSUMED (; recheck): T in p_max,T is undefined in the paper.
                                     # "fifo" = T = N_SV-1 (sum over the FIFO; primary, frozen); "single" = T = 0, the peak
                                     # of one fresh scan phi_0(0). Scales r_SV, the SR metric AND the raster (one p_max)

    @property
    def sigma0_deg(self) -> float:
        """95 % of the 2-D Gaussian mass inside the beam disc (README): 4.5 / sqrt(2 ln 20).
        sigma0_mode="axis": per-axis two-sided 95 % reading, 4.5 / 1.959964."""
        if self.sigma0_mode == "axis":
            return self.beam_r_deg / 1.959963984540054
        if self.sigma0_mode != "radial":
            raise ValueError(self.sigma0_mode)
        return self.beam_r_deg / math.sqrt(2.0 * math.log(20.0))


@dataclass(frozen=True)
class NetConfig:
    encoder: str = "mhsa"            # flat | bigru | mhsa  (paper §2.6)
    masked: bool = True              # companion §III-D "masked upon input" (primary),
    cnn_out: int = 128               # paper §3
    mlp: tuple = (128, 128)          # paper §3 ("2 x 128")
    bigru_hidden: int = 64           # paper §3
    bigru_out: int = 64              # paper §3 (Dense 128->64 ASSUMED)
    mhsa_gru_hidden: int = 32        # paper §3
    mhsa_heads: int = 4              # ASSUMED,
    mhsa_d: int = 64                 # ASSUMED,
    init_style: str = "torch"        # official toolchain (SB3 2.4.0a8 fork + torch 2.3.1, fixture P8): Linear layers in
                                     # the extractor orthogonal sqrt 2 / zero bias, GRU + MHA in_proj torch defaults,
                                     # LayerNorm eps 1e-5, AE decoder torch Linear default. "flax" = the small-scale runs,


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4                 # library (SB3 default),
    gamma: float = 0.99              # library
    gae_lambda: float = 0.95         # library
    clip: float = 0.2                # library
    n_epochs: int = 10               # library
    minibatch: int = 64              # library
    n_steps: int = 2048              # library,
    n_envs: int = 8                  # ASSUMED (8-core i7-9700, paper §3),
    ent_coef: float = 0.0            # library
    vf_coef: float = 0.5             # library
    max_grad_norm: float = 0.5       # library
    adam_eps: float = 1e-5           # library
    total_steps: int = 2_000_000     # ASSUMED cap, (fallback 1e6). NOT what ran: small/full-scale ran 114,688 steps per lane
                                     # (amendment, hard GPU ceiling), ~1.1 % of the paper's 1e7 (review)
    curve_every: int = 100_000       # README (learning curves every 1e5 env steps)
    curve_scenarios: int = 16        # README (validation set)


@dataclass(frozen=True)
class PretrainConfig:
    ae_steps: int = 20_000           # ASSUMED,
    ae_lr: float = 1e-3              # ASSUMED,
    ae_batch: int = 256              # ASSUMED,
    ae_dec_hidden: int = 256         # ASSUMED,
    bc_episodes: int = 50            # ASSUMED,
    bc_epochs: int = 5               # ASSUMED,
    bc_batch: int = 32               # library (imitation BC),
    bc_lr: float = 1e-3              # library (imitation BC Adam),
    bc_ent_weight: float = 1e-3      # library (imitation BC ent_weight),
    bc_train_extractor: bool = False  # (INFERRED): False = track extractor frozen during BC (small/full-scale); True = the
                                     # imitation-fork default, BC optimises every policy parameter (recheck)


@dataclass(frozen=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    pre: PretrainConfig = field(default_factory=PretrainConfig)
    seed_root: int = 15              # ASSUMED (loop iteration number), README
    n_eval_scenarios: int = 100      # paper §3,
    n_seeds: int = 5                 # loop protocol,
    lane_keys_shared_across_arms: bool = False   # review: True = lane key from the seed alone (common random numbers
                                                 # across ablation arms); False = small/full-scale key tree (arm id in the key)


# Key-tree constants (README, README)
K_EVAL_SCEN = 0xE7A1     # the 100 final-evaluation scenarios -- NEVER used before the final evaluation
K_VALID = 0x7A11         # 16 validation scenarios (learning curves, small-scale diagnostics)
K_HAND_TUNE = 0x4A4D     # 20 training-distribution scenarios for tuning the hand rule
K_SMALL = 0x6A72         # small-scale scenarios (added implementation; disjoint from the eval set)
K_BASE_ACT = 0xBA5E      # Static / Random action keys


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def with_env(cfg: Config, **kw) -> Config:
    return replace(cfg, env=replace(cfg.env, **kw))
