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
    grad = false;
  end

  R = chol(K + sigma2 * eye(n));

  Kinv = cholinv(R); % R' * R * Kinv == eye(n)
  Kinvy = cholsolve(R, y); % K * Kinvy == y

  if grad
    Kinv2y = cholsolve(R, Kinvy); % K * Kinv * Kinvy = Kinvy
    diagKinv2 = dot(Kinv, Kinv); % diag(Kinv * Kinv)

    diagZKinv = zeros(size(Kinv, 1), 1, size(dK, 3));
    ZKinvy = zeros(size(Kinv, 1), 1, size(dK, 3)); % for Kinv * dK(l) * Kinvy

    for l = 1:size(dK, 3)
      Z = cholsolve(R, dK(:, :, l)); % K * Z == dK(:, :, l)
      diagZKinv(:, 1, l) = dot(Z, Kinv)'; % ZKinv * K = Z
      ZKinvy(:, 1, l) = Z * Kinvy;
    end
  end

  if ~grad
    % one iteration of the crossvalidation
    nlp1 = @(itr, ite) ...
      cvPredProbFcn(ite, y, hypPlsor, nHypCov, Kinv, Kinvy);
  else
    % gather all precomputed values for CV's gradient
    gradOpts = struct( ...
      'diagZKinv', diagZKinv, ...
      'ZKinvy', ZKinvy, ...
      'diagKinv2', diagKinv2, ...
      'Kinv2y', Kinv2y ...
    );

    % one iteration of crossvalidation
    nlp1 = @(itr, ite) ...
      cvPredProbFcn(ite, y, hypPlsor, nHypCov, Kinv, Kinvy, gradOpts);
  end

  % crossvalidate
  vals = crval.crossval(nlp1, (1:n)');
  res = -sum(vals);

  p = res(1); % the objective value

  if grad
    dp = res(2:end); % the gradient
  end
end

