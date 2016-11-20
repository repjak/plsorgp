function err = misclassErr(ypred, y, type)
%CLASSERR Compute either misclassification rate, absolute error or
%mean-squared error on predicted data.
%   mcr    = MISCLASSERR(ypred, y) return misclassification rate. Vectors
%   ypred and y are of the same length n.
%   The misclassification rate (or zero-one error) is defined as
%   1 - (1/n) * sum(ind(ypred(i), y(i))) for i = 1,...,n, where
%   ind(x, y) = 1 if x == y and ind(x, y) = 0 otherwise.
%   abserr = MISCLASSERR(ypred, y, 'abs') return (1/n) *
%   sum(abs(ypred(i) - y(i))) for i in range of ypred.
%   mse    = MISCLASSERR(ypred, y, 'mse') return the mean squared error.

  if length(ypred) ~= length(y)
    error('Input vectors must be of the same length.');
  end

  n = length(ypred);
  if n == 0
    err = 0;
    return;
  end

  if nargin < 3 || strcmp(type, 'zeroone') || strcmp(type, 'mcr')
    err = 100 * sum(ypred ~= y) / n;
  elseif strcmp(type, 'abs')
    err = sum(abs(ypred - y)) / n;
  elseif strcmp(type, 'mse')
    err = sum(abs(ypred - y)^2) / n;
  else
    error('Unknown error type ''%s''.', type);
  end
end

