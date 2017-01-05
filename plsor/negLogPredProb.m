function [logp, dlogp] = negLogPredProb(hyp, nHypCov, covFcn, X, y, n, dbg)
% NEGLOGPREDPROB Negative logarithm of predictive probability. 
% The objective function that is to be optimized. (Eq. 11)

  if nargin < 8
    dbg = false;
  end

  errtol = 1e-4;

  hypPlsor = hyp(1:end-1-nHypCov);
  hypCov = hyp(end-nHypCov:end-1);
  sigman = hyp(end);
  sigman2 = sigman^2;

  % the covariance matrix and its gradient if required
  if nargout >= 2
    [K, dK] = covFcn(X, X, hypCov);
    grad = true;
  else
    K = covFcn(X, X, hypCov);
    grad = false;
  end

  % (K + sigman2 * eye(n)) / sigman2 == R' * R
  R = chol(K + sigman2 * eye(n)) / sigman;

  V = R' \ (1./sigman * eye(n));
  diagKinv = dot(V, V)'; % diagKinv = diag(inv(K + sigman2 * eye(n)))
  assert(~dbg || ...
    sum(abs(diagKinv - diag(cholinv(R) ./ sigman2))) < n*errtol);

  Kinvy = cholsolve(R, y) / sigman2; % (K + sigman2 * eye(n)) * Kinvy == y
  assert(~dbg || sum(abs(Kinvy - (cholinv(R) ./ sigman2) * y)) < n*errtol);

  % leave-one-out predictive mean (Eq. 12)
  muloo = y - Kinvy ./ diagKinv;
  % leave-one-out variance (Eq. 13)
  s2loo = 1 ./ diagKinv - sigman2;

  if dbg
    i0 = randi(n);
    idx = [1:(i0 - 1) (i0 + 1):n];
    [muloo0, s2loo0] = gpPred(X(idx, :), y(idx), X(i0, :), covFcn, ...
      hypCov, sigman2);
    assert(sum(abs([muloo0 s2loo0] - [muloo(i0) s2loo(i0)])) <= n*errtol);
  end

  if grad
    Kinv = cholinv(R) / sigman2;

    ds2loo = zeros(n, nHypCov + 1);
    dmuloo = zeros(n, nHypCov + 1);

    dK(:, :, end+1) = 2 * sigman * eye(n);

    for l = 1:size(dK, 3)
      Z = cholsolve(R, dK(:, :, l)) ./ sigman2; % K * Z == dK(:, :, l)
      ds2loo1 = dot(Z', Kinv)' ./ (diagKinv.^2);
	    dmuloo(:, l) = (Z * Kinvy) ./ diagKinv - Kinvy .* ds2loo1;
      ds2loo(:, l) = ds2loo1;
    end

    ds2loo(:, end) = ds2loo(:, end) - 2 * sigman;

    [p, dp] = predProb(y, hypPlsor, muloo, s2loo, dmuloo, ds2loo);

    logp = -sum(reallog(p));
    dlogp = -sum(bsxfun(@rdivide, dp, p), 1);
  else
    p = predProb(y, hypPlsor, muloo, s2loo);

    logp = -sum(reallog(p));
  end
end

