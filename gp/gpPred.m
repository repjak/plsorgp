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

    if iscell(covFcn)
      % GPML covariances
      K = feval(covFcn{:}, hyp, X);    % train covariance (posterior?)
    else
      % Jakub's covariance
      K = covFcn(X, X, hyp);
    end

    K = K/sigma2 + 1.0001* eye(n);
    R = chol(K);
    Kinvy = cholsolve(R, y) / sigma2;
  elseif nargin >= 8
    % FIX THIS / DO NOT USE THIS
    % This seems to not work correctly!!!
    R = varargin{1};
    Kinvy = varargin{2};
  else
    error('Not enough input parameters.');
  end

  if iscell(covFcn)
    hyp_gpml.cov = hyp;
    hyp_gpml.lik = log(sigma2)/2;

    post.sW = ones(n,1)/sqrt(sigma2);   % sqrt of noise precision vector
    post.L  = R;
    post.alpha = Kinvy;

    [mu_GPML, s2_GPML] = gp(hyp_gpml, @infExact, @meanZero, covFcn, @likGauss, ...
      X, post, Xnew);

    mu = mu_GPML;
    s2 = s2_GPML;
    return
  end

  if any(size(Xnew, 2) ~= size(X, 2))
    error('Training and test data dimensions don''t match.');
  end

  if iscell(covFcn)
    % GPML covariances
    Ks = feval(covFcn{:}, hyp, X, Xnew);    % train covariance (posterior?)
  else
    % Jakub's covariance
    Ks = covFcn(X, Xnew, hyp);
  end

  assert(all(size(Ks) == [n m]));

  mu = Ks' * Kinvy;
  assert(all(size(mu) == [m 1]));

  if nargout >= 2
    if iscell(covFcn)
      kss = feval(covFcn{:}, hyp, Xnew, 'diag');
    else
      kss = covFcn(Xnew, 'diag', hyp);
    end
    assert(all(size(kss) == [m 1]));

    % Jakub's predictive variance
    % ... seems wrong :(
    %{
    v = R' \ Ks;
    assert(all(size(v) == [n m]));

    s2_J = max(kss - dot(v, v)', zeros(size(kss)));
    assert(all(s2_J >= 0));
    assert(all(size(s2_J) == [m 1]));
    %}


    %
    % Lukas' predictive variances
    %
    % {
    L = R;
    Ltril = all(all(tril(L,-1)==0));   % is L an upper triangular matrix?
    if (Ltril)   % L is triangular => use Cholesky parameters (alpha,sW,L)
      sW = ones(n,1) / sqrt(sigma2);      % sqrt of noise precision vector
      V  = L' \ (repmat(sW,1,m) .* Ks);
      fs2 = kss - sum(V.*V,1)';        % predictive variances
    else         % L is not triangular => use alternative parametrisation
      fs2 = kss + sum(Ks .* (L*Ks), 1)';  % predictive variances
    end
    % remove numerical noise i.e. negative variances
    Fs2 = max(fs2, 0);
    % apply likelihood function
    [~, ~, Ys2] = feval(@likGauss, log(sigma2)/2, [], mu(:), Fs2(:));
    % correct GPML's bug for the case when Fs2 == zeros(...)
    if (size(Ys2,1) == 1)
      Ys2 = repmat(Ys2, size(Fs2,1), 1);
    end
    s2_L = Ys2;
    % }

    % Use Lukas's prediction
    s2 = s2_L;
  end
end

