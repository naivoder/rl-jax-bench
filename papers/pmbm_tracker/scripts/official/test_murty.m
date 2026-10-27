function test_murty(ntrials, seed)
% Verify the repo's murty.m (Option 1) against BRUTE FORCE k-best enumeration on random cost matrices
% (the thing kBest2DAssign.m would compute).  Semantics checked: rows are the "measurements" (all assigned), costs may be
% negative / +Inf / tied.  Own test code; murty.m itself is the authors' (work copy).
more off;
root = getenv('PMBM_ORACLE_ROOT'); if isempty(root), root = fileparts(mfilename('fullpath')); end
addpath(fullfile(root, 'Assignment'));
if nargin < 1, ntrials = 2000; end
if nargin < 2, seed = 1; end
rand('seed', seed); randn('seed', seed);
nbad = 0; ncheck = 0; nerr = zeros(1,5); nwrong = zeros(1,5); nzero = 0; kinds = {'dense', 'sparseInf', 'ties', 'filterlike', 'zeromin'};
cnt_bad = zeros(1, numel(kinds)); cnt_tot = zeros(1, numel(kinds));
for t = 1:ntrials
  kind = mod(t, numel(kinds)) + 1;
  switch kinds{kind}
    case 'dense'
      n = randi(4); m = n + randi(3) - 1; C = 5 * randn(n, m);
    case 'sparseInf'
      n = randi(4); m = n + randi(3); C = 5 * randn(n, m); C(rand(n, m) < 0.4) = Inf;
    case 'ties'
      n = randi(4); m = n + randi(3) - 1; C = randi(4, n, m) - 2;
    case 'filterlike'   % rows = measurements, cols = tracks: private "new track" column per row + gated old tracks
      n = randi(5); nold = randi(4); m = nold + n; C = Inf(n, m);
      for i = 1:n
        C(i, nold + i) = -log(rand() * 3 + 1e-3);              % private column (new track / clutter option)
        old = find(rand(1, nold) < 0.5); C(i, old) = 4 * randn(1, numel(old)) - 2;
      end
    case 'zeromin'
      n = randi(3); m = n + randi(2); C = zeros(n, m); C(rand(n, m) < 0.3) = 1;
  end
  kb = randi(12);
  [A_b, c_b] = brute_kbest(C, kb);                                % reference (sorted, up to kb)
  errd = false;
  try
    [A_m, c_m] = murty(C, kb);
  catch err
    A_m = []; c_m = []; errd = true;
  end
  ok = true;
  nm = size(A_m, 1);
  if nm ~= numel(c_b), ok = false; end
  if ok
    c_m = c_m(:)';
    if max(abs(sort(c_m) - c_b)) > 1e-9, ok = false; end
    for q = 1:nm                                                   % each returned assignment must be valid and match its cost
      a = A_m(q, :);
      if any(a == 0) || numel(unique(a)) < numel(a), ok = false; break; end
      cc = sum(C(sub2ind(size(C), 1:size(C, 1), a)));
      if ~isfinite(cc) || abs(cc - c_m(q)) > 1e-9, ok = false; break; end
    end
    if size(unique(A_m, 'rows'), 1) ~= nm, ok = false; end
  end
  if numel(c_b) == 0 && errd, ok = true; nzero += 1; end          % no feasible assignment at all: an error is acceptable
  if ~ok && errd, nerr(kind) += 1; elseif ~ok, nwrong(kind) += 1; end
  cnt_tot(kind) += 1; ncheck += 1;
  if ~ok, nbad += 1; cnt_bad(kind) += 1; if nbad <= 4 && ~errd, disp(C); printf('kbest=%d brute costs: %s\nmurty costs: %s\n', kb, mat2str(c_b, 6), mat2str(c_m(:)', 6)); end, end
end
for q = 1:numel(kinds), printf('kind %-10s: %4d trials, %4d mismatches vs brute force (%d errors thrown, %d wrong results)\n', kinds{q}, cnt_tot(q), cnt_bad(q), nerr(q), nwrong(q)); end
printf('(%d trials had NO feasible assignment; murty throwing an error there is not counted as a mismatch)\n', nzero);
printf('TOTAL %d trials, %d mismatches\n', ncheck, nbad);
end

function [A, c] = brute_kbest(C, kb)
% all injective row->column assignments with finite cost, k cheapest (costs sorted ascending)
[n, m] = size(C); P = perms_inj(n, m); A = []; c = [];
for q = 1:size(P, 1)
  cc = sum(C(sub2ind(size(C), 1:n, P(q, :))));
  if isfinite(cc), A(end+1, :) = P(q, :); c(end+1) = cc; end
end
[c, o] = sort(c); A = A(o, :);
kb = min(kb, numel(c)); A = A(1:kb, :); c = c(1:kb);
end

function P = perms_inj(n, m)
if n == 0, P = zeros(1, 0); return; end
P = zeros(0, n);
sub = perms_inj(n - 1, m);
for q = 1:size(sub, 1)
  for col = 1:m
    if ~any(sub(q, :) == col), P(end+1, :) = [sub(q, :), col]; end
  end
end
end
