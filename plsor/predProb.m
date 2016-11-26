function [p, dp] = predProb(y, hyp, mu, s2, dmu, ds2, j)
  n = length(y);

  assert(length(hyp) >= 2);
  alpha = hyp(1);
  delta = hyp(3:end);

  r = length(delta) + 2;

  if ~all(0 <= y & y <= r)
    error('Ordinal class out of range.');
  end

  if any([length(mu), length(s2)] ~= n)
    error('Lengths of targets and predictions don''t match.');
  end

  w = sqrt(1 + alpha^2 .* s2);
  w2 = 1 + alpha^2 .* s2;

  q1 = arrayfun(@(i) alpha * mu(i) + betai(y(i), hyp), 1:n)';
  q2 = arrayfun(@(i) alpha * mu(i) + betai(y(i) - 1, hyp), 1:n)';

  p = normcdf(q1./w) - normcdf(q2./w);

  if nargout > 1
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

    dp = zeros(n, length(hyp) + size(dmu, 2));

    if nargin < 7
      j = 1:(length(hyp) + size(dmu, 2));
    elseif ~all(j <= length(hyp) + size(dmu, 2))
      error('Parameter index out of range.');
    end

    for i = 1:n
      if y(i) - 1 == 0 && y(i) == r
        % if r == 1 then prob == 1
        continue;
      elseif y(i) == 1
        % prob == normcdf(...) - 0
        [fi, dfi] = f(y(i), hyp, mu(i), dmu(i, :), j);
        [gi, dgi] = g(y(i), hyp, s2(i), ds2(i, :), j);
        dp(i, :) = normpdf(fi/gi) * (dfi * gi - dgi * fi) / w2(i);
      elseif y(i) == r
        % prob == 1 - normcdf(...)
        [hi, dhi] = f(y(i)-1, hyp, mu(i), dmu(i, :), j);
        [gi, dgi] = g(y(i), hyp, s2(i), ds2(i, :), j);
        dp(i, :) = normpdf(hi/gi) * (dgi * hi - dhi * gi) / w2(i);
      else
        [fi, dfi] = f(y(i), hyp, mu(i), dmu(i, :), j);
        [hi, dhi] = f(y(i)-1, hyp, mu(i), dmu(i, :), j);
        [gi, dgi] = g(y(i), hyp, s2(i), ds2(i, :), j);
        d1 = dfi * gi - dgi * fi;
        d2 = dhi * gi - dgi * hi;
        dp(i, :) = (d1 * normpdf(fi/gi) - d2 * normpdf(hi/gi)) / w2(i);
      end
    end
  end

  function [z, df] = f(y, hyp, mu, dmu, j)
    z = alpha * mu + betai(y, hyp);
    assert(~isinf(betai(y, hyp)));

    if nargout == 1
      return;
    end

    df = zeros(1, length(j));

    for k = 1:length(j)
      if j(k) == 1
        % dalpha
        df(k) = mu;
      elseif j(k) == 2 || j(k) <= length(hyp)
        % dbeta1 or ddelta(j(k) - 1)
        [~, db] = betai(y, hyp, j(k));
        df(k) = db;
      else
        % dtheta(j(k) - length(hyp)) or dsigma2
        m = j(k) - length(hyp);
        df(k) = alpha * dmu(m);
      end
    end
  end

  function [z, dg] = g(~, hyp, s2, ds2, j)
    z = sqrt(1 + alpha^2 * s2);

    if nargout == 1
      return;
    end

    dg = zeros(1, length(j));

    for k = 1:length(j)
      if j(k) == 1
        % dalpha
        dg(k) = alpha * s2 / z;
      elseif j(k) == 2 || j(k) <= length(hyp)
        % dbeta1 or ddelta(j(k) - 1)
        dg(k) = 0;
      else
        % dtheta(j(k) - length(hyp)) or dsigma2
        m = j(k) - length(hyp);
        dg(k) = alpha^2 * ds2(m) / (2 * z);
      end
    end
  end
end

