function p = logPredProbLoo(i, y, hyp, nHypCov, Kinv, Kinvy, gradOpts)
%LOGPREDLOO Log predictive probability of the ith training example using the
%remaining training examples.

  assert(length(hyp) >= 2);
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

  if nargin < 7
    p = log(predProb(y(i), hyp, muloo, s2loo));
  else
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

    thetaj = j(j > length(hyp) & j <= length(hyp) + nHypCov + 1);
    dmuloo = zeros(length(thetaj), 1);
    ds2loo = zeros(length(thetaj), 1);

    % partial derivatives of dmuloo and ds2loo
    for l = 1:length(thetaj)
      % the offset of covariance hyperparameters
      m = thetaj(l) - length(hyp);

      if m < nHypCov + 1
        % dtheta
        dmuloo(l) = (ZKinvy(i, 1, m) - ...
          Kinvy(i) * diagZKinv(i, 1, m) / Kinv(i, i)) / Kinv(i, i);
        ds2loo(l) = diagZKinv(i, 1, m) / Kinv(i, i)^2;
      else
        % dsigma2
        dmuloo(l) = (Kinv2y(i) - ...
          Kinvy(i) * diagKinv2(i) / Kinv(i, i)) / Kinv(i, i);
        ds2loo(l) = diagKinv2(i) / Kinv(i, i)^2;
      end
    end

    [pval, dp] = predProb(y(i), hyp, muloo, s2loo, dmuloo, ds2loo, j);

    p = [log(pval) dp ./ pval];
  end
end

