function [mu, s2] = gpPred(X, y, Xnew, covFcn, hyp, sigma2, varargin)
%GPPRED A prediction for a Gaussian process with Gaussian likelihood.
%   mu       = gpPred(X, y, Xnew, covFcn, hyp, sigma2) mean predictions at
%   M-R test data Xnew given N-R training data X, N-1 vector of response
%   values y, the covariance function covFcn, its hyperparameters hyp and
%   the noise parameter sigma2.
%   mu       = gpPred(X, [], Xnew, covFcn, hyp, [], R, Kinvy) supply
%   a precomputed upper triangular Cholesky factor R of the covariance
%   matrix (with sigma2 added), so that R'R = cov(X, X) + sigma2*eye(N)
%   and a solution to K * Kinvy = y for Kinvy.
%   [mu, s2] = gpPred(__) compute also the predictive variances
%   at the new points.

  n = size(X, 1);
  m = size(Xnew, 1);

  if nargin <= 6
    if any(size(y) ~= [n 1])
      error('The training values must be of shape %d-1', n);
    end

    K = covFcn(X, X, hyp) + sigma2 * eye(n);
    R = chol(K);
    Kinvy = cholsolve(R, y);
  elseif nargin >= 8
    R = varargin{1};
    Kinvy = varargin{2};
  else
    error('Not enough input parameters.');
  end

  if any(size(Xnew, 2) ~= size(X, 2))
    error('Training and test data dimensions don''t match.');
  end

  Ks = covFcn(X, Xnew, hyp);
  assert(all(size(Ks) == [n m]));

  mu = Ks' * Kinvy;
  assert(all(size(mu) == [m 1]));

  if nargout >= 2
    kss = covFcn(Xnew, 'diag', hyp);
    assert(all(size(kss) == [m 1]));

    v = R' \ Ks;
    assert(all(size(v) == [n m]));

    s2 = kss - dot(v, v)';
    assert(all(s2 >= 0));
    assert(all(size(s2) == [m 1]));
  end
end

