function [b, edges] = binning(y, k)
%BINNING Discretize data into uniformely sized intervals.
%   [b, edges] = BINNING(y, k) divide range of y into k intervals
%   determined by k + 1 edges such that edge(j+1) - edge(j) == range(y) / k
%   for all 1 <= j <= k-1 and edge(k+1) == Inf. The vector b satisfies
%   the condition that b(i) equals j iff edges(j) <= y(i) < edges(j+1).

  if k <= 0
    error('The number of groups must be a positive number.');
  end

  edges = [cumsum([min(y) repmat(range(y) / k, 1, k-1)]) Inf];
  b = arrayfun(@(x) find(x < edges, 1) - 1, y);
end

