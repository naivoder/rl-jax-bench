function gospa_ref(in_mat, out_mat)
% Oracle for the GOSPA metric (authors' GOSPA.m + ComputeGOSPAerror.m, Octave).  Own driver code.
%   in_mat : A (random pairs) = {xs [Np,2,Nmax] (NaN padded), ys likewise, nx, ny}, plus run data: est [4,E,K], n_est [K],
%            X_truth [16,K] (official layout), t_birth, t_death, c, p, alpha
more off;
root = getenv('PMBM_ORACLE_ROOT');
addpath(fullfile(root, 'PMBM filter')); addpath(fullfile(root, 'GOSPA code')); addpath(fullfile(root, 'Assignment')); addpath(fullfile(root, 'shims'));
P = load(in_mat);
c = P.c; p = P.p; alpha = P.alpha;
Np = size(P.xs, 1);
d = zeros(Np, 1); loc = d; mis = d; fal = d;
for i = 1:Np
  X = squeeze(P.xs(i, :, 1:P.nx(i))); Y = squeeze(P.ys(i, :, 1:P.ny(i)));
  X = reshape(X, 2, P.nx(i)); Y = reshape(Y, 2, P.ny(i));
  [dd, ~, dc] = GOSPA(X, Y, p, c, alpha);
  d(i) = dd; loc(i) = dc.localisation; mis(i) = dc.missed; fal(i) = dc.false;
end
% real run: ComputeGOSPAerror on the dumped estimates (k = 1..K), c_gospa = c
K = size(P.est, 3);
sq = zeros(K, 1); sl = sq; sm = sq; sf = sq;
for k = 1:K
  n = P.n_est(k);
  Xe = zeros(0, 1);
  for j = 1:n, Xe = [Xe; P.est(:, j, k)]; end
  [a, b, m, f] = ComputeGOSPAerror(Xe, P.X_truth, P.t_birth, P.t_death, c, k);
  sq(k) = a; sl(k) = b; sm(k) = m; sf(k) = f;
end
save('-v6', out_mat, 'd', 'loc', 'mis', 'fal', 'sq', 'sl', 'sm', 'sf');
printf('[INFO] gospa_ref done: %d pairs, %d steps\n', Np, K);
end
