function p = logPredProbLeaveOneOut(i, ys, hyp, Kinv, Kinvy, j, dK)
%LOGPREDLEAVEONEOUT Log predictive probability of the ith training example using the
%remaining training examples.

  alpha = hyp(1);
  beta1 = hyp(2);
  delta = hyp(3:end);

  n = length(ys);         % the number of training examples
  r = length(delta) + 2;  % the number of ordinal classes

  assert(ys(i) > 0 && ys(i) <=  r, 'The ordinal class out of range.');
  assert(i >= 1 && i <= length(ys), 'The left-out example index out of range.');
  assert(all(size(Kinv) == n), 'The covariance matrix must be of shape N-N.');
  assert(length(Kinvy) == n, 'Kinv * y product length must be N.');

  muloo = ys(i) - Kinvy(i) / Kinv(i, i);
  s2loo = 1 / Kinv(i, i);

  if nargin == 5
    % compute the log probability
    p = log(predProb(ys(i), alpha, beta1, delta, muloo, s2loo));
  elseif nargin == 7
    % compute the derivative according to the jth hyperparameter
    assert(j < 1 && j > 2 + length(delta) + length(dK), ...
      'The derivative variable index out of range.');

    fi = alpha*muloo + betai(y, beta1, delta);
    hi = alpha*muloo + betai(y-1, beta1, delta);
    gi = sqrt(1 + alpha^2*s2loo);

    p1 = [normpdf(fi/gi) normpdf(hi/gi)];
    p2 = [normpdf(hi/gi)*hi normpdf(fi/gi)*fi];
    pdiff1 = p1(2) - p1(1);
    pdiff2 = p2(2) - p2(1);

    if j == 1
      % dlogprob / dalpha
      %rest = dalpha(gi, p1, p2, muloo, s2loo);
      d = gi^-3 * (muloo*pdiff1 + alpha*s2loo*pdiff2);
    elseif j == 2
      % dlogprob / dbeta1
      d = pdiff1/gi;
    elseif j < length(delta) + 2
      % dlogprob / delta(j)
      j = j - 2;
      switch sign(j - ys(i))
        case 0
          d = p1(1)/gi;
        case -1
          d = pdiff1/gi;
        otherwise
          d = 0;
      end
    else
      % dlogprob / theta(j)
      j = j - 2 - length(delta);
      d = dtheta(i, pi, muloo, Kinv, j, dK);
    end

    p = d / (normcdf(fi/gi) - normcdf(hi/gi));
  else
    error('Unexpected number of parameters.');
  end
end


function d = dtheta(i, p1, muloo, Kinv, j, dK)
  Zj = Kinv*dK(j);
  A = Zj*Kinv;
  b = Zj*Kinvy;
  d = gi^2 * p1(1) * (b(i) / Kinv(i, i)) - ...
    (alpha * p1(1) - muloop * p1(1) - muloo * p1(2)) * (A(i, i) / Kinv(i, i)^2);
  d = 1/gi^3 * d;
end