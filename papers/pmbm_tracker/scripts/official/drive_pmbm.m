function drive_pmbm(in_mat, out_dir, run_list)
% DRIVE_PMBM  Oracle driver (own code, GNU Octave) around the AUTHORS' PMBM functions
% (Garcia-Fernandez et al., github.com/Agarciafernandez/MTT, folder "PMBM filter").
% It replays the loop of PMBMtarget_filter.m on EXPORTED measurement sets with EXPLICIT parameters and dumps the
% filter's internal state in a canonical numeric form (.mat v6, one file per run) for differential testing.
%
%   drive_pmbm(in_mat, out_dir)            all runs in in_mat
%   drive_pmbm(in_mat, out_dir, run_list)  only the given 0-BASED run indices (so 3 processes can split the work)
%
% INPUT  in_mat (written by scripts/official/npz_to_mat.py or scipy.io.savemat):
%   z          [R,K,M,2]  measurements (x,y);  zvalid [R,K,M]  (1 = valid; valid entries keep their order)
%   F,Q [4x4], H [2x4], Rcov [2x2]            motion/measurement model   (Rcov is the measurement covariance "R")
%   p_s, p_d                                  survival / detection probabilities
%   w_b [Nb], m_b [Nb,4], P_b [Nb,4,4]        Poisson birth intensity components (weight, mean, cov)
%   lambda0 [N0], m0 [N0,4], P0 [N0,4,4]      Poisson prior at k=0 (used only if predict_first = 0, i.e. the CODE setting,
%                                             where filter_pred is initialised directly with this PPP and the first step
%                                             is an update).  For predict_first = 1 (PAPER setting) the prior is EMPTY
%                                             at k=0 and PoissonMBMtarget_pred adds the birth intensity before step 1.
%   predict_first                             1 = paper (empty prior, predict, then update); 0 = code (no predict before k=1)
%   intensity_clutter                         clutter intensity c(z) = lambda_c / area   (explicit scalar)
%   N_h, T_pruning, T_pruningPois, gating, existence_threshold, est1_thr
%   dump_runs  [optional, 0-based]            runs that get FULL tables (Poisson, tracks, globHyp, Murty); other runs get
%                                             estimates + counts only.  default: [0 1]
%   dump_pred  [optional, default 1]          also dump the predicted state (full-dump runs)
%   murty_log  [optional, default 1]          dump per-parent cost matrices and Murty outputs (full-dump runs)
%   murty_steps [optional, default 8]         ... but only for steps k <= murty_steps (the matrices are big)
%
% OUTPUT  out_dir/meta.mat and out_dir/run_RRR.mat (RRR = 0-based, 3 digits).  See scripts/official/load_dump.py and
% oracle_setup.txt for the variable layout.  Estimates are computed on the UPDATED density BEFORE pruning, exactly
% as in PMBMtarget_filter.m.

more off;
root = getenv('PMBM_ORACLE_ROOT');
if isempty(root), root = fileparts(mfilename('fullpath')); end
addpath(fullfile(root, 'PMBM filter'));   % official code (work copy), '/' separators (the .m files use '\')
addpath(fullfile(root, 'GOSPA code'));
addpath(fullfile(root, 'Assignment'));    % murty.m, assignmentoptimal.m
addpath(fullfile(root, 'shims'));         % mvnpdf, poissrnd
global PMBM_ORACLE_MURTY

P = load(in_mat);
R_runs = size(P.z, 1); K = size(P.z, 2); M = size(P.z, 3);
if nargin < 3 || isempty(run_list), run_list = 0:(R_runs - 1); end
if ~isfield(P, 'dump_runs'), P.dump_runs = [0 1]; end
if ~isfield(P, 'dump_pred'), P.dump_pred = 1; end
if ~isfield(P, 'murty_log'), P.murty_log = 1; end
if ~isfield(P, 'murty_steps'), P.murty_steps = 8; end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ---- model (explicit) -------------------------------------------------------------
F = P.F; Q = P.Q; H = P.H; Rc = P.Rcov; p_s = P.p_s; p_d = P.p_d;
Nb = numel(P.w_b);
weights_b = reshape(P.w_b, 1, Nb);
means_b = reshape(P.m_b, Nb, 4)';
covs_b = zeros(4, 4, Nb);
for i = 1:Nb, covs_b(:, :, i) = reshape(P.P_b(i, :, :), 4, 4); end
N0 = numel(P.lambda0);
lambda0 = reshape(P.lambda0, 1, N0);
m0 = reshape(P.m0, N0, 4)';
covs0 = zeros(4, 4, N0);
for i = 1:N0, covs0(:, :, i) = reshape(P.P0(i, :, :), 4, 4); end
intensity_clutter = P.intensity_clutter;
Nhyp_max = P.N_h; T_pruning = P.T_pruning; T_pruningPois = P.T_pruningPois;
gating_threshold = P.gating; existence_threshold = P.existence_threshold;
est1_thr = P.est1_thr; predict_first = P.predict_first;

meta = struct('octave_version', version(), 'predict_first', predict_first, 'p_d', p_d, 'p_s', p_s, ...
  'N_h', Nhyp_max, 'T_pruning', T_pruning, 'T_pruningPois', T_pruningPois, 'gating', gating_threshold, ...
  'existence_threshold', existence_threshold, 'est1_thr', est1_thr, 'intensity_clutter', intensity_clutter, ...
  'R_runs', R_runs, 'K', K, 'M', M, 'in_mat', in_mat, 'dump_runs', P.dump_runs);
try, save('-v6', fullfile(out_dir, 'meta.mat'), '-struct', 'meta'); catch, end   % parallel processes race on this file; harmless

for rr = run_list(:)'
  t_run = tic;
  full = any(P.dump_runs(:) == rr);
  S = struct();
  est_cells = cell(3, K);
  n_glob_upd = zeros(K, 1); n_glob_pr = zeros(K, 1); n_trk_upd = zeros(K, 1); n_trk_pr = zeros(K, 1);
  t_step = zeros(K, 1);

  % ---- initial state -------------------------------------------------------------
  if predict_first
    % paper: no targets at time 0 (empty PPP, no tracks); the first predict adds the birth PPP.
    filter_upd = struct('weightPois', zeros(1, 0), 'meanPois', zeros(4, 0), 'covPois', zeros(4, 4, 0), ...
                        'tracks', {cell(0, 1)}, 'globHyp', [], 'globHypWeight', []);
    filter_pred = PoissonMBMtarget_pred(filter_upd, F, Q, p_s, weights_b, means_b, covs_b);
  else
    % code (PMBMtarget_filter.m): filter_pred initialised with the Poisson prior, first step is an update
    filter_pred.weightPois = lambda0; filter_pred.meanPois = m0; filter_pred.covPois = covs0;
    filter_pred.tracks = cell(0, 1); filter_pred.globHyp = []; filter_pred.globHypWeight = [];
  end

  for k = 1:K
    t_k = tic;
    idx = find(squeeze(P.zvalid(rr + 1, k, :)) ~= 0);
    n = numel(idx);
    if n > 0
      z = reshape(P.z(rr + 1, k, idx, :), n, 2)';
    else
      z = zeros(2, 0);
    end
    pre = sprintf('k%03d', k);

    do_log = full && P.murty_log && k <= P.murty_steps;
    if do_log, PMBM_ORACLE_MURTY = {}; else PMBM_ORACLE_MURTY = 0; end
    if full && P.dump_pred
      S = dump_state(S, ['r' pre], filter_pred, false);   % predicted state going into the update
    end
    filter_upd = PoissonMBMtarget_update(filter_pred, z, H, Rc, p_d, k, gating_threshold, intensity_clutter, Nhyp_max);
    if full
      S.(['z' pre]) = z;
      S = dump_state(S, ['u' pre], filter_upd, true);
      if do_log, S = dump_murty(S, ['u' pre], PMBM_ORACLE_MURTY); end
    end

    X1 = PoissonMBMtarget_estimate1(filter_upd, est1_thr);
    X2 = PoissonMBMtarget_estimate2(filter_upd);
    X3 = PoissonMBMtarget_estimate3(filter_upd);
    est_cells{1, k} = reshape(X1, 4, []); est_cells{2, k} = reshape(X2, 4, []); est_cells{3, k} = reshape(X3, 4, []);
    n_glob_upd(k) = numel(filter_upd.globHypWeight); n_trk_upd(k) = numel(filter_upd.tracks);

    filter_upd = PoissonMBMtarget_pruning(filter_upd, T_pruning, T_pruningPois, Nhyp_max, existence_threshold);
    n_glob_pr(k) = numel(filter_upd.globHypWeight); n_trk_pr(k) = numel(filter_upd.tracks);
    if full, S = dump_state(S, ['p' pre], filter_upd, false); end

    filter_pred = PoissonMBMtarget_pred(filter_upd, F, Q, p_s, weights_b, means_b, covs_b);
    t_step(k) = toc(t_k);
  end

  % ---- estimates (padded) + counts, for every run -------------------------------------
  for e = 1:3
    Emax = max(1, max(cellfun(@(c) size(c, 2), est_cells(e, :))));
    A = nan(4, Emax, K); cnt = zeros(K, 1);
    for k = 1:K
      c = est_cells{e, k}; cnt(k) = size(c, 2);
      if cnt(k) > 0, A(:, 1:cnt(k), k) = c; end
    end
    S.(sprintf('est%d_x', e)) = A; S.(sprintf('est%d_n', e)) = cnt;
  end
  S.n_glob_upd = n_glob_upd; S.n_glob_pruned = n_glob_pr; S.n_trk_upd = n_trk_upd; S.n_trk_pruned = n_trk_pr;
  S.t_step = t_step; S.full_dump = double(full); S.run_index = rr; S.K = K;
  save('-v6', fullfile(out_dir, sprintf('run_%03d.mat', rr)), '-struct', 'S');
  printf('[INFO] run %d done in %.2f s (full=%d); mean glob hyps after pruning %.1f\n', rr, toc(t_run), full, mean(n_glob_pr));
  fflush(stdout);
end
end

% =====================================================================================
function S = dump_state(S, pre, f, with_wlog)
% Canonical flat dump of a PMBM density.  Hypothesis table: one row per single-target hypothesis entry
% (track i, index j inside the track) in the order of the official arrays.  Entries that the official update leaves
% empty by auto-extension (gated-out hypotheses; eB = 0, empty mean/cov) are kept with valid = 0 so that
% the indices stored in globHyp stay meaningful.
Np = numel(f.weightPois);
S.([pre '_poisW']) = reshape(f.weightPois, Np, 1);
if Np > 0
  S.([pre '_poisM']) = f.meanPois;                         % 4 x Np
  S.([pre '_poisP']) = f.covPois;                          % 4 x 4 x Np
else
  S.([pre '_poisM']) = zeros(4, 0); S.([pre '_poisP']) = zeros(4, 4, 0);
end
Nt = numel(f.tracks);
trk_tini = zeros(Nt, 1); trk_nh = zeros(Nt, 1);
for i = 1:Nt
  trk_tini(i) = f.tracks{i}.t_ini; trk_nh(i) = numel(f.tracks{i}.eB);
end
Nh = sum(trk_nh);
h_trk = zeros(Nh, 1); h_j = zeros(Nh, 1); h_e = zeros(Nh, 1); h_w = nan(Nh, 1); h_wk = zeros(Nh, 1); h_valid = zeros(Nh, 1);
h_mean = nan(4, Nh); h_cov = nan(4, 4, Nh); h_len = zeros(Nh, 1);
ahc = cell(1, Nh);
q = 0;
for i = 1:Nt
  tr = f.tracks{i};
  nh = trk_nh(i);
  hasWk = with_wlog && isfield(tr, 'weightBLog_k'); hasW1 = with_wlog && ~hasWk && isfield(tr, 'weightBLog');
  for j = 1:nh
    q = q + 1;
    h_trk(q) = i; h_j(q) = j; h_e(q) = tr.eB(j);
    if hasWk
      h_wk(q) = 2; if j <= numel(tr.weightBLog_k), h_w(q) = tr.weightBLog_k(j); end
    elseif hasW1
      h_wk(q) = 1; if j == 1, h_w(q) = tr.weightBLog(1); end
    end
    mj = tr.meanB{j}; cj = tr.covB{j}; aj = tr.aHis{j};
    if ~(isempty(mj) || isempty(aj))
      h_valid(q) = 1; h_mean(:, q) = mj(:); h_cov(:, :, q) = cj; ahc{q} = aj(:)'; h_len(q) = numel(aj);
    end
  end
end
Lmax = 1; if Nh > 0, Lmax = max(1, max(h_len)); end
AH = -ones(Lmax, Nh);
for q = 1:Nh, AH(1:h_len(q), q) = ahc{q}(:); end
S.([pre '_trk_tini']) = trk_tini; S.([pre '_trk_nhyp']) = trk_nh;
S.([pre '_h_trk']) = h_trk; S.([pre '_h_j']) = h_j; S.([pre '_h_eB']) = h_e;
S.([pre '_h_wlog']) = h_w; S.([pre '_h_wkind']) = h_wk; S.([pre '_h_valid']) = h_valid;
S.([pre '_h_mean']) = h_mean; S.([pre '_h_cov']) = h_cov; S.([pre '_h_ahis']) = AH; S.([pre '_h_ahis_len']) = h_len;
S.([pre '_globHyp']) = f.globHyp;
S.([pre '_globW']) = reshape(f.globHypWeight, numel(f.globHypWeight), 1);
end

function S = dump_murty(S, pre, L)
S.([pre '_mur_n']) = numel(L);
for i = 1:numel(L)
  fn = fieldnames(L{i});
  for q = 1:numel(fn)
    S.(sprintf('%s_mur%02d_%s', pre, i, fn{q})) = L{i}.(fn{q});
  end
end
end
