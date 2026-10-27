function r = poissrnd(lambda, varargin)
% Octave shim (own code): Poisson random numbers via core randp (not used by the exported-measurement driver).
if isempty(varargin)
  r = randp(lambda);
else
  r = randp(lambda, varargin{:});
end
end
