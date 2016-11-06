function [minLoss, idx, predProbs, losses] = zeroone(P)
%ONEZERO Find ordinal class with the highest probability.
%   The zero-one loss for a class is defined as the sum of probabilities
%   over all other classes. The loss is effectively minimized by finding
%   the mode of the predicted distribution.
%
%   See also LOSS, HINGE.

  r = size(P, 1);
  n = size(P, 2);

  if n < 1
    error('The number of predictions must be positive.');
  end

  if r < 2
    error('At least 2 classes are expected.');
  end

  losses = 1 - P;
  [minLoss, idx] = min(losses);

  idxLogical = bsxfun(@eq, idx, (1:r)');
  predProbs = P(idxLogical)';
end

