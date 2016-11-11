function [Kmn, dKmn] = sqexpard(Xm, z, theta, j)
%SQEXPARD The squared exponential covariance function with automatic
%relevance detection (ARD).
%   Kmn        = SQEXPARD(Xm, Xn, theta) m x n cross-covariances
%   Km         = SQEXPARD(Xm, 'diag', theta) n self-covariances
%   [Kmn dKmn] = SQEXPARD(__) compute also the gradient
%   [Kmn dKmn] = SQEXPARD(__, j) partial derivations according to all
%   hyperparameters from j
%
%   The squared exponential with ARD is defined as:
%
%      k(x, y) = sigma^2 * exp(-1/2 * (x - y)' * diag(1/l.^2) * (x - y)).
%
%   where sigma is signal variance and l contains characteristic
%   length scales for each dimension.
%
%   Input arguments:
%
%      Xm    -- an m-d matrix of m examples with dimensionality d
%      Xn    -- an n-d matrix
%      theta -- a vector of d+1 hyperparameters, theta = [sigma; l]
%      j     -- a vector of hyperparameter indices
%
%   See also SQEXP.

  if nargin < 3
    error('Not enough input arguments.');
  end

  d = size(Xm, 2);
  m = size(Xm, 1);

  if ischar(z) && ~strcmp(z, 'diag')
    error(['Unrecognized argument ''' z '''.']);
  elseif strcmp(z, 'diag')
    diag = true;
  else
    diag = false;
    Xn = z;
    n = size(Xn, 1);

    if d ~= size(Xn, 2)
      error('Dimensions must agree.');
    end
  end

  if length(theta) ~= d+1 || any(theta) <= 0
    error(['The squared exponential with ARD expects a column vector', ...
      ' of dim+1 positive hyperparameters.']);
  end

  sigma = theta(1);
  l = reshape(theta(2:end), length(theta)-1, 1);

  if diag
    % self-covariances
    Kmn = zeros(size(Xm, 1), 1);
  else
    % substract mean for numerical stabilization (inspiration from gpml package)
    mu = (m/(m+n))*mean(Xm, 1) + (n/(m+n))*mean(Xn, 1);
    Xm = bsxfun(@minus, Xm, mu);
    Xn = bsxfun(@minus, Xn, mu);

    Xm = bsxfun(@rdivide, Xm, (1./l)');
    Xn = bsxfun(@rdivide, Xn, (1./l)');

    % the pairwise squared distances
    Kmn = bsxfun(@plus, dot(Xm, Xm, 2), dot(Xn, Xn, 2)') - 2 * (Xm * Xn');
  end

  if nargout < 2

  else
    % for the partial derivatives
    dKmn = repmat(exp(-Kmn / 2), 1, 1, length(j));

    Kmn = sigma^2 * exp(-Kmn / 2);

    if nargin < 4
      j = (1:(d+1));
    end

    for k = reshape(j, 1, numel(j))
      if k == 1
        % the signal variance parameter
        dKmn(:, :, k) = 2 * sigma * dKmn(:, :, k);
      elseif k <= d + 1
        % a length-scale parameter
        d = k - 1;

        if diag
          dKmn(:, :, k) = zeros(size(Kmn, 1), size(Kmn, 2));
        else
          % pairwise squared distances in kth dimension
          sqdistk = bsxfun(@(xk, yk) (xk - yk).^2, Xm(:, d), Xn(:, d)');
          dKmn(:, :, k) = sigma^2 * dKmn(:, :, k) .* (sqdistk / l(d)^3);
        end
      else
        error(['Hyperparameter index ''' k ''' out of range.']);
      end
    end
  end
end

