function [minLoss, idx, predProbs, losses] = zeroone(P)
%ONEZERO Find ordinal class with the highest probability.
%   The zero-one loss for a class is defined as the sum of probabilities
%   over all other classes. The loss is effectively minimized by finding
%   the mode of the predicted distribution.
%
%   See also LOSS, HINGE.

  assert(isempty(P) || all(abs(sum(P, 2) - 1) <= 1e-4));

  r = size(P, 2);

  losses = 1 - P;
  [minLoss, idx] = min(losses, [], 2);

  idxLogical = bsxfun(@eq, (1:r)', idx');
  Q = P';
  predProbs = Q(idxLogical);
end

