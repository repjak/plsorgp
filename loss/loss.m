function [minLoss, idx, predProbs, losses] = loss(P, L)
%RISK A general loss-minimization function.
%   [expLoss, idx, predProbs, losses] = LOSS(P, L) minimizes a loss
%   function given by a R-R square matrix C, where C(i,j) expresses the
%   risk (loss) that the ith ordinal class is misclassfied for the jth
%   ordinal class.
%
%   P is a N-R matrix of N predicted probability distributions for each of
%   R ordinal classes (ordered).
%
%   Returns the following vectors:
%
%      minLoss   -- a row vector of minimal losses for each prediction
%      idx       -- the indices of the predicted classes
%      predProbs -- the predicted probability values
%      losses    -- an N-R matrix of column vector of losses per each
%                prediction
%
%   See also ABSERR, ZEROONE.

  assert(isempty(P) || all(abs(sum(P, 2) - 1) <= 1e-4));

  n = size(P, 1);  % size of data
  r = size(P, 2);  % number of groups

  % assert(isempty(P) || all(abs(sum(P, 2) - 1) <= 1e-4));

  if nargin < 2
    error('Loss matrix not specified.');
  end

  if any(size(L) ~= [r r])
    error(['The loss matrix must be a square matrix with each ' ...
      'dimension equal to the number of classes.']);
  end

  losses = zeros(n, r);
  for i = 1:n
    losses(i, :) = P(i, :) * L';
  end

  [minLoss, idx] = min(losses, [], 2);

  idxLogical = bsxfun(@eq, (1:r)', idx');
  Q = P';
  predProbs = Q(idxLogical);
end

