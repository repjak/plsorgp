function [b, db] = betai(y, hyp, j)
%BETAI Derived parameter beta(y) for a vector of ordinal classes y.
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

  if any(y < 0 & y > r)
    error('Ordinal class out of range.');
  end

  if ~isvector(y)
    error('The input is not a vector.');
  end

  b = zeros(size(y, 1), size(y, 2));

  for i = 1:length(y)
    switch y(i)
      case 0
        b(i) = -Inf;
      case r
        b(i) = Inf;
      otherwise
        b(i) = beta1 + sum(delta(1:(y(i)-1)));
    end
  end

  if nargout >= 2
    if nargin < 3
      j = 1:length(hyp);
    end

    if ~all(j >= 1 & j <= length(hyp))
      error('Parameter index out of range.');
    end

    % the derivative is calculated as follows:
    % * one for beta1 and all deltas with indices less or
    %   equal y + 1, for all y strictly between zero and r
    % * zero for all other ys and all other hyperparameters
    drv = @(z, k) ...
      double(z > 0 & z < r & ...
        (k == 2 | (k >= 3 & k <= r & k - 1 <= z)) ...
      );

    if size(y, 2) == 1
      db = bsxfun(drv, y, reshape(j, 1, length(j)));
    else
      db = bsxfun(drv, y, reshape(j, length(j), 1));
    end
  end
end

