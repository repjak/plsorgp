function [p, dp] = negLogPredProb(hyp, nCovHyp, covFcn, cvPredProbfcn, X, y, n)
%NEGLOGPREDPROB The objective function that is to be optimized.

  hypPlsor = hyp(1:end-1-nCovHyp);
  hypCov = hyp(end-nCovHyp:end-1);
  sigma2 = hyp(end);

  % the covariance matrix and its gradiend if required
  if nargout > 1
    [K, dK] = covFcn(X, X, hypCov);
  else
    K = covFcn(X, X, hypCov);
    dK = [];
  end

  R = chol(K + sigma2 * eye(n));
  Kinv = R \ (R' \ eye(n));
  Kinvy = R \ (R' \ y);

  if nargout < 2
    nlp1 = @(itr, ite) sum(arrayfun(@(i) cvPredProbfcn(i, y, hypPlsor, nCovHyp, Kinv, Kinvy), ite));
    p = -sum(crval.crossval(nlp1, (1:n)'));
  else
    % the gradient
    nlp1 = @(itr, ite) sum(arrayfun(@(i) cvPredProbfcn(i, y, hypPlsor, nCovHyp, Kinv, Kinvy, dK), ite));
    r = -sum(crval.crossval(nlp1, (1:n)'), 1);
    p = r(1);
    dp = r(2);
  end
end

