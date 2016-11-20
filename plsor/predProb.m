function [p, dp] = predProb(y, hyp, mu, s2, dmu, ds2, j)
  n = length(y);

  assert(length(hyp) >= 2);
  alpha = hyp(1);
  delta = hyp(3:end);

  r = length(delta) + 2;

  if ~all(0 <= y & y <= r)
    error('Ordinal class out of range.');
  end

  w = sqrt(1 + alpha^2 * s2);
  q1 = arrayfun(@(z) alpha * mu + betai(z, hyp), y) ./ w;
  q2 = arrayfun(@(z) alpha * mu + betai(z - 1, hyp), y) ./ w;

  p = bsxfun(@(x1, x2) normcdf(x1) - normcdf(x2), q1, q2);

  if nargout > 1
    % partial derivatives for all input points

    if nargin < 6
      error('Derivatives of predicted mean and variance are required.');
    end

    dp = zeros(n, length(hyp) + length(dmu));

    if nargin < 7
      j = 1:(length(hyp) + length(dmu));
    end

    assert(length(dmu) == length(ds2));
    assert(all(j <= length(hyp) + length(dmu)));

    for i = 1:length(y)
      if y(i) - 1 == 0 && y(i) == r
        % if r == 1 then logprob == 0
        continue;
      elseif y(i) == 1
        % logprob == log(normcdf(...))
        [fi, dfi] = f(y(i), hyp, mu, dmu, j);
        [gi, dgi] = g(y(i), hyp, s2, ds2, j);
        dp(i, :) = normpdf(fi/gi) * (dfi * gi - dgi * fi) / gi^2;
      elseif y(i) == r
        % logprob == log(1 - normcdf(...))
        [hi, dhi] = h(y(i), hyp, mu, dmu, j);
        [gi, dgi] = g(y(i), hyp, s2, ds2, j);
        dp(i, :) = normpdf(hi/gi) * (dgi * hi - dhi * gi) / gi^2;
      else
        [fi, dfi] = f(y(i), hyp, mu, dmu, j);
        [hi, dhi] = h(y(i), hyp, mu, dmu, j);
        [gi, dgi] = g(y(i), hyp, s2, ds2, j);
        d1 = dfi * gi - dgi * fi;
        d2 = dhi * gi - dgi * hi;
        dp(i, :) = (d1 * normpdf(fi/gi) - d2 * normpdf(hi/gi)) / gi^2;
      end
    end
  end

  function [z, df] = f(y, hyp, mu, dmu, j)
    z = alpha * mu + betai(y, hyp);

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

  function [z, dh] = h(y, hyp, mu, dmu, j)
    z = alpha * mu + betai(y - 1, hyp);

    if nargout == 1
      return;
    end

    dh = zeros(1, length(j));

    for k = 1:length(j)
      if j(k) == 1
        % dalpha
        dh(k) = mu;
      elseif j(k) == 2 || j(k) <= length(hyp)
        % dbeta1 or ddelta(j(k) - 1)
        [~, db] = betai(y - 1, hyp, j(k));
        dh(k) = db;
      else
        % dtheta(j(k) - length(hyp)) or dsigma2
        m = j(k) - length(hyp);
        dh(k) = alpha * dmu(m);
      end
    end
  end

  function [z, dg] = g(~, hyp, s2, ds2, j)
    z = sqrt(1 + alpha^2 * s2);

    if nargout == 1
      return;
    end

    dg = zeros(1, length(j));

    for k = reshape(j, 1, numel(j))
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

