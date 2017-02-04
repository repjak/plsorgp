function [logp, dlogp, muloo, s2loo] = negLogPredProb(hyp, nHypCov, covFcn, X, y, dbg)
% NEGLOGPREDPROB Negative logarithm of predictive probability. 
% The objective function that is to be optimized. (Eq. 11)
%
% Input:
%   hyp     - hyperparameter vector [alpha beta1 delta cov sigman]
%   nHypCov - number of covariance function hyperparameters
%   covFcn  - covariance function handle 
%   X       - input data | N x dim double
%   y       - function values for input X | N x 1 double
%   dbg     - debugging mode | boolean
%
% Output:
%   logp  - negative logarithm of probability | double
%   dlogp - derivative of logp | double
%   muloo - leave-one-out mean prediction for X | N x 1 double
%   s2loo - leave-one-out variance prediction for X | N x 1 double

  if nargin < 6
    dbg = false;
  end

  errtol = 1e-4;
  dlogp = [];

  hypPlsor = hyp(1:end-1-nHypCov);
  hypCov = hyp(end-nHypCov:end-1);
  sigman = hyp(end);
  n = size(X, 1);

  %%%%%%%%%%%%%%%%
  % Jakubova verze
  %%%%%%%%%%%%%%%%

  %{
  % the covariance matrix and its gradient if required
  if nargout >= 2
    [K, dK] = covFcn(X, X, hypCov);
    grad = true;
  else
    K = covFcn(X, X, hypCov);
    grad = false;
  end

  sigman2 = exp(2*sigman);
  sigman = exp(sigman);

  % (K + sigman2 * eye(n)) / sigman2 == R' * R
  R = chol(K + sigman2 * eye(n)) / sigman;

  V = R' \ (1./sigman * eye(n));
  diagKinv = dot(V, V)'; % diagKinv = diag(inv(K + sigman2 * eye(n)))
  assert(~dbg || ...
    sum(abs(diagKinv - diag(cholinv(R) ./ sigman2))) < n*errtol);

  Kinvy = cholsolve(R, y) / sigman2; % (K + sigman2 * eye(n)) * Kinvy == y
  assert(~dbg || sum(abs(Kinvy - (cholinv(R) ./ sigman2) * y)) < n*errtol);
  %}

  %%%%%%%%%%%%%%%%%%%%
  % Lukasova verze
  %%%%%%%%%%%%%%%%%%%%

  % Calculate covariances
  %
  if iscell(covFcn)
    K__X_N__X_N = feval(covFcn{:}, hypCov, X);    % train covariance (posterior?)

    if nargout >= 2
      dK = zeros(size(K__X_N__X_N, 1), size(K__X_N__X_N, 2), nHypCov);
      for j = 1:nHypCov
        dK(:, :, j) = feval(covFcn{:}, hypCov, X, [], j);
      end
      grad = true;
    else
      grad = false;
    end
  end

  % Posterior
  %
  % evaluate mean vector for X_N
  % m_N = feval(model.meanFcn, model.hyp.mean, X_N');
  % noise variance of likGauss
  sn2 = exp(2*sigman);
  % Cholesky factor of covariance with noise
  L = chol(K__X_N__X_N/sn2 + eye(n) + 0.0001*eye(n));
  R = L;
  % inv(K+noise) * (y_N - mean)
  Kinv = solve_chol(L, eye(n)) / sn2;

  KinvyL    = Kinv*y;
  diagKinvL = diag(Kinv);

  Kinvy = KinvyL;
  diagKinv = diagKinvL;
  sigman2  = sn2;

  % leave-one-out predictive mean (Eq. 12)
  muloo = y - Kinvy ./ diagKinv;
  % leave-one-out variance (Eq. 13)
  s2loo = 1 ./ diagKinv;

  if dbg
    i0 = randi(n);
    idx = [1:(i0 - 1) (i0 + 1):n];
    [muloo0, s2loo0] = gpPred(X(idx, :), y(idx), X(i0, :), covFcn, ...
      hypCov, sigman2);
    assert(sum(abs([muloo0 s2loo0] - [muloo(i0) s2loo(i0)])) <= n*errtol);
  end

  if grad
    Kinv = cholsolve(R, eye(n)) / sigman2;

    ds2loo = zeros(n, nHypCov + 1);
    dmuloo = zeros(n, nHypCov + 1);

    dK(:, :, end+1) = 2 * sigman2 * eye(n);

    for l = 1:size(dK, 3)
      Z = cholsolve(R, dK(:, :, l)) ./ sigman2; % K * Z == dK(:, :, l)
      ds2loo1 = dot(Z', Kinv)' ./ (diagKinv.^2);
	    dmuloo(:, l) = (Z * Kinvy) ./ diagKinv - Kinvy .* ds2loo1;
      ds2loo(:, l) = ds2loo1;
    end

    ds2loo(:, end) = ds2loo(:, end);% - 2 * sigman2;

    [p, dp] = predProb(y, hypPlsor, muloo, s2loo, dmuloo, ds2loo);

    logp = -sum(reallog(p));
    dlogp = -sum(bsxfun(@rdivide, dp, p), 1);
  else
    p = predProb(y, hypPlsor, muloo, s2loo);

    logp = -sum(reallog(p));
  end
end

