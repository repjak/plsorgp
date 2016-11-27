function [Kmn, dKmn] = sqexp(Xm, z, theta, j)
%SQEXP The squared exponential covariance function.
%   Kmn         = SQEXP(Xm, Xn, theta) m x n cross-covariances
%   Kmn         = SQEXP(Xm, 'diag', theta) n self-covariances
%   [Kmn, dKmn] = SQEXP(__) the covariance matrix and its gradient
%   [Kmn, dKmn] = SQEXP(__, j) partial derivations according to all
%   hyperparameters from j
%
%   The squared exponential is defined as:
%
%      k(x, y) = sigma^2 * exp((-1/2)* (1/l^2) * (x - y)'*(x - y)).
%
%   where sigma is signal variance and l is the characteristic length
%   scale.
%
%   Input arguments:
%
%      Xm    -- an m-d matrix of m examples with dimensionality d
%      Xn    -- an n-d matrix
%      theta -- a vector of 2 hyperparameters,
%               theta = [log(sigma^2); log(l)]
%      j     -- a vector of hyperparameter indices
%
%   The output is a m-n matrix of cross-covariances or a m-1 vector of
%   self-covariances, if the flag 'diag' is specified. If a vector of
%   hyperparameter indices j is given as the fourth argument, the output
%   contains also partial derivatives according to given hyperparameters in
%   3rd dimension.
%
%   See also SQEXPARD.

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

  if length(theta) ~= 2
    error('The squared exponential expects a vector of 2 hyperparameters.');
  end

  sigma2 = exp(2 * theta(1));
  l = exp(theta(2));

  if any([sigma2; l] <= 0)
    error('Hyperparameters must be positive.');
  end

  if diag
    % self-covariances
    Kmn = zeros(size(Xm, 1), 1);
  else
    % substract mean for numerical stabilization (inspiration from the gpml package)
    mu = (m/(m+n))*mean(Xm, 1) + (n/(m+n))*mean(Xn, 1);
    Xm = bsxfun(@minus, Xm, mu);
    Xn = bsxfun(@minus, Xn, mu);

    Xm = Xm ./ l;
    Xn = Xn ./ l;

    % squared distances
    Kmn = bsxfun(@plus, dot(Xm, Xm, 2), dot(Xn, Xn, 2)') - 2 * (Xm * Xn');
  end

  if nargout < 2
    Kmn = sigma2 * exp(-Kmn / 2);
  else
    if nargin < 4
      j = 1:2;
    end

    % for the partial derivatives
    dKmn = repmat(Kmn, 1, 1, length(j));

    Kmn = sigma2 * exp(-Kmn / 2);

    for k = 1:length(j)
      switch j(k)
        case 1
          dKmn(:, :, k) = 2 * sigma2 * exp(-dKmn(:, :, k)/2);
        case 2
          dKmn(:, :, k) = sigma2 * exp(-dKmn(:, :, k)/2) .* ...
            dKmn(:, :, k);
        otherwise
          error('Hyperparameter index %d out of range.', j(k));
      end
    end
  end
end

