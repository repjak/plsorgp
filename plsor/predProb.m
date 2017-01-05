function [p, dp] = predProb(y, hyp, mu, s2, dmu, ds2, j)
% PREDPROB predictive probability function.

  n = length(y);
  y = reshape(y, n, 1);

  assert(length(hyp) >= 2);
  alpha = hyp(1);
  delta = hyp(3:end);

  % number of hyperparameters
  r = length(delta) + 2;

  if ~all(0 <= y & y <= r)
    error('Ordinal class out of range.');
  end

  if any([length(mu), length(s2)] ~= n)
    error('Lengths of targets and predictions don''t match.');
  end

  if nargout >= 2
    % partial derivatives for all input points

    if nargin < 6
      error('Derivatives of predicted mean and variance are required.');
    end

    if ~all(size(dmu) == size(ds2))
      error('Dimensions of predictive means and variances don''t match.');
    end

    if size(dmu, 1) ~= n
      error(['The derivative matrix must have N rows, ' ...
        'where N is the number of targets.']);
    end

    if nargin < 7
      j = 1:(length(hyp) + size(dmu, 2));
    elseif ~all(j >= 1 & j <= length(hyp) + size(dmu, 2))
      error('Parameter index out of range.');
    end
  end

  g = sqrt(1 + alpha^2 .* s2);
  g2 = 1 + alpha^2 .* s2;

  if nargout < 2
    b1 = betai(y, hyp);
    b2 = betai(y-1, hyp);
  else
    [b1, db1] = betai(y, hyp, j(j <= length(hyp)));
    [b2, db2] = betai(y-1, hyp, j(j <= length(hyp)));
  end

  f = alpha .* mu + b1;
  h = alpha .* mu + b2;

  % predictive distribution (Eq. 9)
  p = normcdf(f./g) - normcdf(h./g);

  if nargout >= 2
    dp = zeros(n, length(j));
    df = zeros(n, length(j));
    dh = zeros(n, length(j));
    dg = zeros(n, length(j));

    for k = 1:length(j)
      if j(k) == 1
        % dalpha
        df(:, k) = mu;
        dh(:, k) = mu;
        dg(:, k) = alpha * s2 ./ g;
      elseif j(k) == 2 || j(k) <= length(hyp)
        % dbeta1 or ddelta(j(k) - 1)
        df(:, k) = db1(:, j(k));
        dh(:, k) = db2(:, j(k));
        % by initialization: dg(:, k) == 0;
      else
        % dtheta(j(k) - length(hyp)) or dsigma2
        m = j(k) - length(hyp);
        df(:, k) = alpha .* dmu(:, m);
        dh(:, k) = alpha .* dmu(:, m);
        dg(:, k) = alpha^2 .* ds2(:, m) ./ (2 * g);
      end
    end

    d1 = bsxfun(@times, df, g) - bsxfun(@times, dg, f);
    d2 = bsxfun(@times, dh, g) - bsxfun(@times, dg, h);
    d1(y == r, :) = 0;
    d2(y == 1, :) = 0;
    dp = (bsxfun(@times, d1, normpdf(f./g)) - ...
      bsxfun(@times, d2, normpdf(h./g)));
    dp = bsxfun(@rdivide, dp, g2);
  end
end

