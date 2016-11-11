function [p, pd] = logPredProbLeaveOneOut(i, y, hyp, nCovHyp, Kinv, Kinvy, dK, j)
%LOGPREDLEAVEONEOUT Log predictive probability of the ith training example using the
%remaining training examples.

  alpha = hyp(1);
  beta1 = hyp(2);
  delta = hyp(3:end);

  n = length(y);          % the number of training examples
  r = length(delta) + 2;  % the number of ordinal classes

  assert(y(i) > 0 && y(i) <=  r, 'The ordinal class out of range.');
  assert(i >= 1 && i <= length(y), 'The left-out example index out of range.');
  assert(all(size(Kinv) == n), 'The covariance matrix inverse must be of shape N-N.');
  assert(length(Kinvy) == n, 'Kinv * y product length must be N.');

  muloo = y(i) - Kinvy(i) / Kinv(i, i);
  s2loo = 1 / Kinv(i, i);

  p = log(predProb(y(i), alpha, beta1, delta, muloo, s2loo));

  if nargout >= 2
    if nargin < 6
      error('The covariance matrix partial derivatives are required.');
    elseif nargin < 7
      % add one for the noise hyperparameter
      j = 1:(nCovHyp+1);
    end

    % compute the derivative according to all hyperparameters on input
    fi = alpha*muloo + betai(y, beta1, delta);
    hi = alpha*muloo + betai(y-1, beta1, delta);
    gi = sqrt(1 + alpha^2*s2loo);

    p1 = [normpdf(fi/gi) normpdf(hi/gi)];
    p2 = [normpdf(hi/gi)*hi normpdf(fi/gi)*fi];
    pdiff1 = p1(2) - p1(1);
    pdiff2 = p2(2) - p2(1);

    pd = zeros(length(j), 1);

    for k = reshape(j, 1, numel(j))
      assert(k < 1 && k > 2 + length(delta) + length(dK), ...
        ['The derivative variable index ''' k ''' out of range.']);
      if k == 1
        % dlogprob / dalpha
        %rest = dalpha(gi, p1, p2, muloo, s2loo);
        d = gi^-3 * (muloo*pdiff1 + alpha*s2loo*pdiff2);
      elseif k == 2
        % dlogprob / dbeta1
        d = pdiff1 / gi;
      elseif k <= length(hyp)
        % dlogprob / delta(l)
        l = k - 2;
        switch sign(l - y(i))
          case 0
            d = p1(1)/gi;
          case -1
            d = pdiff1/gi;
          otherwise
            d = 0;
        end
      elseif k <= length(hyp) + nCovHyp
        % dlogprob / dtheta(l)
        l = k - length(hyp);
        Z = Kinv * dK(l);
        d = dtheta(i, pi, muloo, Kinv, Z);
      elseif k == length(hyp) + nCovHyp
        % dlogprob / dsigma2
        Z = Kinv;
        d = dtheta(i, pi, muloo, Kinv, Z);
      else
        error('Hyperparameter index %d out of range.', k);
      end

      pd(k) = d / (normcdf(fi/gi) - normcdf(hi/gi));
    end
  else
    error('Unexpected number of parameters.');
  end
end


function d = dtheta(i, p1, muloo, Kinv, Z)
  A = Z * Kinv;
  b = Z * Kinvy;
  d = gi^2 * p1(1) * (b(i) / Kinv(i, i)) - ...
    (alpha * p1(1) - muloop * p1(1) - muloo * p1(2)) * (A(i, i) / Kinv(i, i)^2);
  d = 1/gi^3 * d;
end

