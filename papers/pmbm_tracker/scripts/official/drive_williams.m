function drive_williams(in_mat, out_dir, alg, run_list)
% DRIVE_WILLIAMS  Oracle driver (own code, GNU Octave) around Williams' TOMB/MOMB code (arXiv 1203.2995 ancillary files
% predict.m, update.m, lbp.m, tomb.m, momb.m), same exported measurement sets and input .mat as drive_pmbm.m.
% It replays the loop of the ancillary main.m (predict -> update -> LBP -> TOMB|MOMB -> estimate) with the model built from
% the EXPLICIT parameters in the input .mat (so the paper's section VII model can be run, not gentruth.m's):
%   F,Q,H,Rcov,p_s,p_d ; birth PPP: w_b,m_b,P_b (-> model.lambdab/xb/Pb) ; clutter intensity -> model.lambda_fa
%   wil_lambdau0 [optional, default 0 = EMPTY prior] intensity of an initial unknown-target PPP component with mean/cov
%                 wil_m0 [1,4], wil_P0 [1,4,4]  (the ancillary main.m uses lambdau=10, xu=model.xb, Pu=model.Pb)
%   wil_exist_thresh [optional, default 0.7 = paper section VII]  existence threshold of the r > thr estimate
%   dump_runs [optional, 0-based, default [0 1]] full per-step tables for these runs, estimates+counts for the others
% alg = 'TOMB' or 'MOMB'.  run_list = 0-based run indices (default all).
% Output: out_dir/wil_<alg>_run_RRR.mat.  Estimates are formed AFTER update+LBP+TOMB/MOMB at each step:
%   est_thr  : x(:, r > thr)                         (the paper's rule; ancillary main.m for TOMB with thr = 0.8)
%   est_map  : MAP cardinality of the MB, top-r means (ancillary main.m rule for MOMB)
more off;
root = getenv('PMBM_ORACLE_ROOT');
if isempty(root), root = fileparts(mfilename('fullpath')); end
addpath(fullfile(root, 'williams2015_anc'));   % official ancillary code (work copy)
addpath(fullfile(root, 'shims'));
P = load(in_mat);
R_runs = size(P.z, 1); K = size(P.z, 2);
if nargin < 3 || isempty(alg), alg = 'TOMB'; end
if nargin < 4 || isempty(run_list), run_list = 0:(R_runs - 1); end
if ~isfield(P, 'dump_runs'), P.dump_runs = [0 1]; end
if ~isfield(P, 'wil_exist_thresh'), P.wil_exist_thresh = 0.7; end
if ~isfield(P, 'wil_lambdau0'), P.wil_lambdau0 = 0; end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

model.F = P.F; model.Q = P.Q; model.H = P.H; model.R = P.Rcov; model.Pd = P.p_d; model.Ps = P.p_s;
model.existThresh = P.wil_exist_thresh;
Nb = numel(P.w_b);
model.lambdab = reshape(P.w_b, 1, Nb);
model.xb = reshape(P.m_b, Nb, 4)';
model.Pb = zeros(4, 4, Nb);
for i = 1:Nb, model.Pb(:, :, i) = reshape(P.P_b(i, :, :), 4, 4); end
model.lambda_fa = P.intensity_clutter;

for rr = run_list(:)'
  t_run = tic;
  full = any(P.dump_runs(:) == rr);
  S = struct();
  % initial state: no tracks; unknown-target PPP empty (paper) or one component
  r = zeros(0, 1); x = zeros(4, 0); Pm = zeros(4, 4, 0);
  if P.wil_lambdau0 > 0
    lambdau = P.wil_lambdau0; xu = reshape(P.wil_m0, 1, 4)'; Pu = reshape(P.wil_P0, 4, 4);
  else
    lambdau = zeros(1, 0); xu = zeros(4, 0); Pu = zeros(4, 4, 0);
  end
  est_thr = cell(1, K); est_map = cell(1, K); nmb = zeros(K, 1); nppp = zeros(K, 1); sum_r = zeros(K, 1); t_step = zeros(K, 1);
  for t = 1:K
    t_k = tic;
    idx = find(squeeze(P.zvalid(rr + 1, t, :)) ~= 0); n = numel(idx);
    if n > 0, z = reshape(P.z(rr + 1, t, idx, :), n, 2)'; else z = zeros(2, 0); end
    pre = sprintf('k%03d', t);
    [r, x, Pm, lambdau, xu, Pu] = predict(r, x, Pm, lambdau, xu, Pu, model);
    if full
      S.(['r' pre '_r']) = r; S.(['r' pre '_x']) = x; S.(['r' pre '_P']) = Pm;
      S.(['r' pre '_lam']) = lambdau(:); S.(['r' pre '_xu']) = xu; S.(['r' pre '_Pu']) = Pu;
    end
    [lambdau, xu, Pu, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew] = update(lambdau, xu, Pu, r, x, Pm, z, model);
    [pupd, pnew] = lbp(wupd, wnew);
    if full
      S.(['u' pre '_z']) = z; S.(['u' pre '_wupd']) = wupd; S.(['u' pre '_rupd']) = rupd; S.(['u' pre '_xupd']) = xupd;
      S.(['u' pre '_Pupd']) = Pupd; S.(['u' pre '_wnew']) = wnew; S.(['u' pre '_rnew']) = rnew; S.(['u' pre '_xnew']) = xnew;
      S.(['u' pre '_Pnew']) = Pnew; S.(['u' pre '_pupd']) = pupd; S.(['u' pre '_pnew']) = pnew;
      S.(['u' pre '_lam']) = lambdau(:); S.(['u' pre '_xu']) = xu; S.(['u' pre '_Pu']) = Pu;
    end
    if strcmp(alg, 'TOMB')
      [r, x, Pm] = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew);
    elseif strcmp(alg, 'MOMB')
      [r, x, Pm] = momb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew);
    else
      error(['unknown algorithm ' alg]);
    end
    % estimates
    ss = r > model.existThresh;
    est_thr{t} = x(:, ss);
    ssm = false(size(r)); nn = 0;
    if ~isempty(r)
      pcard = prod(1 - r) * poly(-r ./ (1 - r));      % same rule as ancillary main.m (MOMB branch)
      [~, nn] = max(pcard); [~, o] = sort(-r); nn = nn - 1; ssm(o(1:nn)) = true;
    end
    est_map{t} = x(:, ssm);
    nmb(t) = numel(r); nppp(t) = numel(lambdau); sum_r(t) = sum(r);
    if full
      S.(['m' pre '_r']) = r; S.(['m' pre '_x']) = x; S.(['m' pre '_P']) = Pm;
    end
    t_step(t) = toc(t_k);
  end
  for e = 1:2
    if e == 1, cc = est_thr; nm = 'est_thr'; else cc = est_map; nm = 'est_map'; end
    Emax = max(1, max(cellfun(@(c) size(c, 2), cc)));
    A = nan(4, Emax, K); cnt = zeros(K, 1);
    for t = 1:K
      cnt(t) = size(cc{t}, 2); if cnt(t) > 0, A(:, 1:cnt(t), t) = cc{t}; end
    end
    S.([nm '_x']) = A; S.([nm '_n']) = cnt;
  end
  S.n_mb = nmb; S.n_ppp = nppp; S.sum_r = sum_r; S.t_step = t_step; S.full_dump = double(full); S.run_index = rr; S.K = K;
  save('-v6', fullfile(out_dir, sprintf('wil_%s_run_%03d.mat', alg, rr)), '-struct', 'S');
  printf('[INFO] williams %s run %d done in %.2f s (full=%d)\n', alg, rr, toc(t_run), full); fflush(stdout);
end
end
