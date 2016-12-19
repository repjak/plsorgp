function [b, edges] = binning(y, k, type)
% BINNING Discretize data into uniformely sized intervals.
%   [b, edges] = BINNING(y, k, type) divide range of y into k intervals
%   determined by k + 1 edges according to binning type. The vector b 
%   satisfies the condition that b(i) equals j iff edges(j) <= y(i) < 
%   edges(j+1). Edges always satisfy the condition: edge(1) = -Inf and 
%   edge(k+1) == Inf.
%
% Input:
%   y - values to bin
%   k - number of bins
%   type - type of binning:
%            'uniform' - edges satisfy: edge(j+1) - edge(j) == range(y) / k
%                        for all 2 <= j <= k-1
%            'none'    - number of bins equals to number of datapoints,
%                        edges satisfy: edge(j) = 

  if k <= 0
    error('The number of groups must be a positive number.');
  end

  switch type
    case {'unif', 'uniform'}
      diff = range(y) / (k-1);
      edges = [-Inf, min(y) + diff/2 : diff : max(y) - diff/2, Inf];
      % edges = [cumsum([min(y) repmat(range(y) / k, 1, k-1)]) Inf];
      b = arrayfun(@(x) find(x < edges, 1) - 1, y);
    case 'none'
      [~, b] = unique(y);
      y_sort = y(b)';
      edges = [-Inf, (y_sort(1:end-1) + y_sort(2:end))/2, Inf];
    otherwise
  end
end

