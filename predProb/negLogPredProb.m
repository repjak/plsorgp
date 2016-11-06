function p = negLogPredProb(hyp, covfcn, nCovHyp, cvPredProbfcn, X, ys, n)
%NEGLOGPREDPROB The objective function that is to be optimized.

  hypPlsor = hyp(1:end-nCovHyp);
  hypCov = hyp(end-nCovHyp+1:end);

  % compute the covariance matrix and all its derivatives
  K = covfcn(X, X, hypCov, (1:length(hyp.cov))');
  Kinv = inv(K);
  Kinvy = K \ y;

  nlp1 = @(itr, ite) sum(arrayfun(@(i) cvPredProbfcn(i, ys, hypPlsor, Kinv, Kinvy), ite));
  p = -sum(crval.crossval(nlp1, (1:n)'));

end

