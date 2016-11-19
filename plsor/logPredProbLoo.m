function p = logPredProbLoo(i, y, hyp, nHypCov, Kinv, Kinvy, gradOpts)
%LOGPREDLOO Log predictive probability of the ith training example using the
%remaining training examples.

  alpha = hyp(1);
  delta = hyp(3:end);

  n = length(y);          % the number of training examples
  r = length(delta) + 2;  % the number of ordinal classes

  assert(y(i) > 0 && y(i) <= r, 'The ordinal class out of range.');
  assert(i >= 1 && i <= length(y), 'The left-out example index out of range.');
  assert(all(size(Kinv) == n), 'The covariance matrix inverse must be of shape N-N.');
  assert(length(Kinvy) == n, 'Kinv * y product length must be N.');

  muloo = y(i) - Kinvy(i) / Kinv(i, i);
  s2loo = 1 / Kinv(i, i);

  p = log(predProb(y(i), hyp, muloo, s2loo));

  if nargin >= 7
    diagZKinv = gradOpts.diagZKinv;
    ZKinvy = gradOpts.ZKinvy;
    diagKinv2 = gradOpts.diagKinv2;
    Kinv2y = gradOpts.Kinv2y;

    if ~isfield(gradOpts, 'j')
      % add one for the noise hyperparameter
      j = 1:(length(hyp)+nHypCov+1);
    else
      j = gradOpts.j;
      assert(all(j >= 1 && j <= length(hyp) + nHypCov + 1));
    end

    assert(all(size(diagZKinv) == [n 1 nHypCov]));
    assert(all(size(ZKinvy) == [n 1 nHypCov]));
    assert(all(size(diagKinv2) == [1 n]));
    assert(all(size(Kinv2y) == [n 1]));

    % allocate space for partial derivatives
    p = [p zeros(1, length(j))];

    if y(i) - 1 == 0 && y(i) == r
      % if r == 1 then logprob == 0
      return;
    elseif y(i) == 1 % y(i) - 1 == 0
      % logprob == log(normcdf(...))
      [fi, dfi] = f(j);
      [gi, dgi] = g(j);
      for l = reshape(j, 1, numel(j))
        d = dfi(l) * gi - dgi(l) * fi;
        p(l+1) = (d * normpdf(fi/gi)) / (gi^2 * normcdf(fi/gi));
      end
    elseif y(i) == r
      % logprob == log(1 - normcdf(...))
      [hi, dhi] = h(j);
      [gi, dgi] = g(j);
      for l = reshape(j, 1, numel(j))
        d = dgi(l) * hi - dhi(l) * gi;
        p(l+1) = (d * normpdf(hi/gi)) / (gi^2 * (1-normcdf(hi/gi)));
      end
    else
      [fi, dfi] = f(j);
      [hi, dhi] = h(j);
      [gi, dgi] = g(j);
      for l = reshape(j, 1, numel(j))
        d1 = dfi(l) * gi - dgi(l) * fi;
        d2 = dhi(l) * gi - dgi(l) * hi;
        p(l+1) = (d1 * normpdf(fi/gi) - d2 * normpdf(hi/gi)) / ...
          (gi^2 * (normcdf(fi/gi) - normcdf(hi/gi)));
      end
    end
  end


  function [z, df] = f(j)
    z = alpha*muloo + betai(y(i), hyp);

    if nargout == 1
      return;
    end

    df = zeros(1, length(j));

    for k = 1:length(j)
      if j(k) == 1
        %dalpha
        df(k) = muloo;
      elseif j(k) == 2 || j(k) <= length(hyp)
        % dbeta1 or ddelta(j(k) - 1)
        [~, db] = betai(y(i), hyp, j(k));
        df(k) = db;
      elseif j(k) <= length(hyp) + nHypCov
        % dtheta(j(k) - length(hyp))
        m = j(k) - length(hyp);
        df(k) = dHypCov(alpha, Kinv(i, i), Kinvy(i), ...
          diagZKinv(i, 1, m), ZKinvy(i, 1, m));
      elseif j(k) == length(hyp) + nHypCov + 1
        % dsigma2
        df(k) = dHypCov(alpha, Kinv(i, i), Kinvy(i), ...
          Kinv2y(i), diagKinv2(i));
      end
    end
  end

  function [z, dh] = h(j)
    z = alpha*muloo + betai(y(i)-1, hyp);

    if nargout == 1
      return;
    end

    dh = zeros(1, length(j));

    for k = 1:length(j)
      if j(k) == 1
        % dalpha
        dh(k) = muloo;
      elseif j(k) == 2 || j(k) <= length(hyp)
        % dbeta1 or ddelta(j(k) - 1)
        [~, db] = betai(y(i), hyp, j(k));
        dh(k) = db;
      elseif j(k) <= length(hyp) + nHypCov
        % dtheta(j(k) - length(hyp))
        m = j(k) - length(hyp);
        dh(k) = dHypCov(alpha, Kinv(i, i), Kinvy(i), ...
          diagZKinv(i, 1, m), ZKinvy(i, 1, m));
      elseif j(k) == length(hyp) + nHypCov + 1
        % dsigma2
        dh(k) = dHypCov(alpha, Kinv(i, i), Kinvy(i), ...
          Kinv2y(i), diagKinv2(i));
      end
    end
  end

  function d = dHypCov(alpha, Kinvii, Kinvyi, ZKinvii, ZKinvyi)
    d = (alpha / Kinvii) * (ZKinvyi - Kinvyi * ZKinvii / Kinvii);
  end

  function [z, dg] = g(j)
    z = sqrt(1 + alpha^2 * s2loo);

    if nargout == 1
      return;
    end

    dg = zeros(1, length(j));

    for k = reshape(j, 1, numel(j))
      if j(k) == 1
        % dalpha
        dg(k) = alpha * s2loo / z;
      elseif j(k) == 2 || j(k) <= length(hyp)
        % dbeta1 or ddelta(j(k) - 1)
        dg(k) = 0;
      elseif j(k) <= length(hyp) + nHypCov
        % dtheta(j(k) - length(hyp))
        m = j(k) - length(hyp);
        dg(k) = (alpha^2 * diagZKinv(i, 1, m)) / (2 * z * Kinv(i, i)^2);
      elseif j(k) == length(hyp) + nHypCov + 1
        % dsigma2
        dg(k) = (alpha^2 * diagKinv2(i)) / (2 * z * Kinv(i, i)^2);
      end
    end
  end

end

