function [minLoss, idx, predProbs, losses] = hinge(P)
%HINGE A variation of the hinge loss function for probabilistic ordinal
%regression.
%   The loss of misclassification is a function of the ordinal classes
%   distance, more precisely, L(i, j) = abs(i-j)/sum([1, ..., r-1]), where
%   r is the number of ordinal classes.
%
%   See also LOSS, ZEROONE.

  r = size(P, 2);  % number of groups

  if r < 3
    w = 1;
  else
    w = 0.5*(r)*(r-1);  % 1 + 2 + ... + r - 1
  end

  penalty = @(i, j) abs(i-j)/w;
  L = bsxfun(penalty, (1:r), (1:r)');

  [minLoss, idx, predProbs, losses] = loss(P, L);
end

