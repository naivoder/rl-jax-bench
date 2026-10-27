function y = mvnpdf(x, mu, Sigma)
% Octave shim (own code): multivariate normal pdf, SINGLE or MULTIPLE observations.
% The official PMBM code calls mvnpdf(z_m, z_pred, S) with 2x1 COLUMN vectors (one observation of dim d);
% MATLAB treats this as one observation.  Rule: a vector argument whose length equals size(Sigma,1) is ONE observation.
% Otherwise x is n-by-d (rows = observations) and mu is 1-by-d or n-by-d.
d = size(Sigma, 1);
if isvector(x) && numel(x) == d
  x = x(:)';
end
if isvector(mu) && numel(mu) == d
  mu = mu(:)';
end
x = x - mu;                       % broadcasting (mu 1-by-d or n-by-d)
L = chol(Sigma, 'lower');
q = L \ x';                       % d-by-n
maha = sum(q .^ 2, 1)';
logdet = 2 * sum(log(diag(L)));
y = exp(-0.5 * maha - 0.5 * logdet - 0.5 * d * log(2 * pi));
end
