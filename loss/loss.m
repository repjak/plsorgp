function [minLoss, idx, predProbs, losses] = loss(P, L)
%RISK A general loss-minimization function.
%   [expLoss, idx, predProbs, losses] = LOSS(P, L) minimizes a loss
%   function given by a square matrix C, where C(i,j) expresses the risk
%   (loss) that the ith ordinal class is misclassfied for the jth ordinal
%   class.
%
%   P is a R-N matrix of N predicted probability distributions for each of
%   R ordinal classes (ordered).
%
%   Returns the following vectors:
%   minLoss   -- a row vector of minimal losses for each prediction
%   idx       -- the indices of the predicted classes
%   predProbs -- the predicted probability values
%   losses    -- an N-R matrix of column vector of losses per each
%                prediction
%
%   See also HINGE, ZEROONE.

  r = size(P, 1);  % number of groups
  n = size(P, 2);  % size of data

  if n < 1
    error('The number of predictions must be positive.');
  end

  if r < 2
    error('At least 2 classes are expected.');
  end

  if any(size(L) ~= [r r])
    error(['The loss matrix must be a square matrix with each ' ...
      'dimension equal to the number of classes.']);
  end

  losses = zeros(r,  n);
  for i = 1:n
    losses(:, i) = L'*P(:, i);
  end

  [minLoss, idx] = min(losses);

  idxLogical = bsxfun(@eq, idx, (1:r)');
  predProbs = P(idxLogical)';
end

