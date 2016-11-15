function [b, db] = betai(y, hyp, j)
%BETAI Derived parameter beta(y).
%   b       = BETAI(y, hyp) compute beta(y) for = [alpha, beta1, delta],
%   where beta(0) = -Inf, beta(r) = Inf and beta(y) = beta1 +
%   sum(delta(1:(y-1))) for 0 < y < r.
%   [b, db] = BETAI(y, hyp) compute also the gradient in all
%   hyperparameters.
%   [b, db] = BETAI(y, hyp, j) compute partial derivations according to all
%   hyperparameters indexed by j.

  beta1 = hyp(2);
  delta = hyp(3:end);

  r = length(delta) + 2;
  assert(y >= 0 && y <= r, 'ordinal class out of range');

  if y == 0
    b = -Inf;
  elseif y == r
    b = Inf;
  else
    b = beta1 + sum(delta(1:(y-1)));
  end

  if nargout >= 2
    if nargin < 3
      j = 1:length(hyp);
    end

    assert(all(j >= 1));

    if y == 0 || y == r
      db = zeros(1, length(j));
    else
      db = double(j == 2 | ...
        (j >= 3 & j <= r & j - 1 <= y));
    end
  end
end

