function [p, dp] = negLogPredProb(hyp, nHypCov, covFcn, cvPredProbFcn, X, y, n)
%NEGLOGPREDPROB The objective function that is to be optimized.

  hypPlsor = hyp(1:end-1-nHypCov);
  hypCov = hyp(end-nHypCov:end-1);
  sigma2 = hyp(end);

  % the covariance matrix and its gradiend if required
  if nargout >= 2
    [K, dK] = covFcn(X, X, hypCov);
    grad = true;
  else
    K = covFcn(X, X, hypCov);
    dK = [];
    grad = false;
  end

  R = chol(K + sigma2 * eye(n));
  Kinv = cholinv(R); % R' * R * Kinv == eye(n)
  Kinvy = cholsolve(R, y); % K * Kinvy == y

  if grad
    ZKinv = zeros(size(dK)); % for Kinv * dK(l) * Kinv
    ZKinvy = zeros(size(Kinv, 1), 1, size(dK, 3)); % for Kinv * dK(l) * Kinvy
    for l = 1:size(dK, 3)
      Z = cholsolve(R, dK(:, :, l)); % K * Z == dK
      ZKinv(:, :, l) = cholsolve(R, Z, 'right'); % ZKinv * K = Z
      ZKinvy(:, :, l) = Z * Kinvy;
    end
  end

  if ~grad
    % one iteration of the crossvalidation
    nlp1 = @(itr, ite) ...
      cvPredProbFcn(ite, y, hypPlsor, nHypCov, Kinv, Kinvy);

    % crossvalidate
    res = -sum(crval.crossval(nlp1, (1:n)'));
    p = res(1); % the objective value
  else
    % gather all precomputed values for CV's gradient
    gradOpts = struct( ...
      'dK', dK, ...
      'ZKinv', ZKinv, ...
      'ZKinvy', ZKinvy, ...
      'diagKinv2', dot(Kinv, Kinv'), ...
      'Kinv2y', Kinv * Kinvy ...
    );

    % one iteration of crossvalidation
    nlp1 = @(itr, ite) ...
      cvPredProbFcn(ite, y, hypPlsor, nHypCov, Kinv, Kinvy, gradOpts);

    % crossvalidate
    res = -sum(crval.crossval(nlp1, (1:n)'));

    assert(all(size(res) == [1 length(hyp) + 1]));

    p = res(1); % the objective value
    dp = res(2:end); % the gradient
  end
end

