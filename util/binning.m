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
%            'best'      - for all 2 <= j <= k-1 edges satisfy: 
%                            edge(j) = (y_sort(j) + y_sort(j-1))/2, 
%                          where y_sort is sorted y in ascend order, 
%            'none'      - number of bins equals to number of datapoints 
%                          (k = N), edges satisfy: 
%                            edge(j) = (y(j+1) + y(j))/2
%            'uniform'   - uniformly distributed bins,
%                          for all 2 <= j <= k-1 edges satisfy: 
%                            edge(j+1) - edge(j) == range(y) / k
%            'unipoints' - uniformly distributed points
%
% Output:
%   b     - binning of 'y' values, vector of the same length as 'y'
%   edges - edges computed according to binning 'type', 1x(k+1) vector

  if k <= 0
    error('The number of groups must be a positive integer.');
  end
  
  [~, b] = unique(y);
  y_sort = y(b)';

  switch type
    % k-1 best values has its own bin
    case 'best'
      edges = [-Inf, (y_sort(1:k-1) + y_sort(2:k))/2,  Inf]; 
      
    % uniformly distributed bins
    case {'unif', 'uniform'}
      diff = range(y) / (k-1);
      edges = [-Inf, min(y) + diff/2 : diff : max(y) - diff/2, Inf];
      
    % uniformly distributed points
    case {'unip', 'unipoints'}
      n = length(y);
      rem = mod(n, k);
      binSize = [floor(n/k)*ones(1, k - rem), ceil(n/k)*ones(1, rem)];
      pid = cumsum(binSize(1:end-1));
      edges = [-Inf, (y_sort(pid) + y_sort(pid + 1))/2, Inf];
      
    % one point one bin
    case 'none'
      edges = [-Inf, (y_sort(1:end-1) + y_sort(2:end))/2, Inf];
      
    otherwise
      error('Undefined binning type: ''%s''', type)
  end
  
  
  b = arrayfun(@(x) find(x < edges, 1) - 1, y);
end

