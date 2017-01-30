function [b, edges] = binning(y, k, type)
% BINNING Discretize data into intervals according to chosen strategy.
%
%   [b, edges] = BINNING(y, k, type) divide range of y into k intervals
%   determined by k + 1 edges according to binning type. The vector b 
%   satisfies the condition that b(i) equals j iff edges(j) <= y(i) < 
%   edges(j+1). Edges always satisfy the condition: edge(1) = -Inf and 
%   edge(k+1) == Inf.
%
%   bintype = BINNING('list') returns cell-array of implemented binning
%   types.
%
% Input:
%   y - values to bin
%   k - number of bins
%   type - type of binning:
%            'best'       - for all 2 <= j <= k-1 edges satisfy: 
%                             edge(j) = (y_sort(j) + y_sort(j-1))/2, 
%                           where y_sort is sorted y in ascend order, 
%            'cluster'    - agglomerative hierarchical clustering
%            'none'       - number of bins equals to number of datapoints 
%                           (k = N), edges satisfy: 
%                             edge(j) = (y(j+1) + y(j))/2
%            'logcluster' - agglomerative hierarchical clustering of log(y)
%            'loguniform' - log-uniformly distributed bins,
%                           for all 2 <= j <= k-1 edges satisfy: 
%                             edge(j+1) - edge(j) == range(log(y)) / k
%                         - y is shifted to be positive
%            'quantile'   - evenly spaced cumulative probabilities of 
%                           points 
%            'uniform'    - uniformly distributed bins,
%                           for all 2 <= j <= k-1 edges satisfy: 
%                             edge(j+1) - edge(j) == range(y) / k
%            'unipoints'  - uniformly distributed points
%
% Output:
%   b     - binning of 'y' values, vector of the same length as 'y'
%   edges - edges computed according to binning 'type', 1x(k+1) vector
%
% See Also:
%   binningComparison

  % default values
  edges = [];
  bintype = {'none', 'best', 'cluster', 'logcluster', 'uniform', 'loguniform', 'quantile', 'unipoints'};
  
  % list all implemented binning types
  if nargin == 1 && strcmp(y, 'list')
    b = bintype;
    return
  end

  % checkout input values
  assert(k > 0 || strcmp(type, 'none'), ...
    'The number of groups must be a positive integer (if not using binning type ''none'').');
  assert(any(strcmp(type, bintype)), 'Undefined binning type ''%s''', type)
  logType = strcmp(type(1:3), 'log');
  
  % normalize input
  n = length(y);
  y = reshape(y, n, 1);
  % compute logarigthm of the input if necessary
  if logType
    if any(y <= 0)
      y_shift = y - min(y, [], 1) + eps;
      isShifted = true;
    else
      y_shift = y;
      isShifted = false;
    end
    y_shift = log(y_shift);
    type = type(4:end);
  else
    y_shift = y;
  end
  % sort input
  [~, b] = unique(y_shift);
  y_sort = y_shift(b)';
  % recompute n due to using unique
  n = length(y_sort);
  % extreme cases
  if n == 1
    edges = [-Inf, Inf];
    b = ones(size(y));
    return
  % if the number of bins is greater or equal to the number of different
  % values, then no binning is needed
  elseif n < k + 1
    % only the uniform binning can return the number of bins lower than
    % required
    if any(strcmp(type, {'unif', 'uniform'}))
      k = n;
    else
      type = 'none';
    end
  end

  % find edges according to binning type
  switch type
    % one point one bin
    case 'none'
      edges = [-Inf, (y_sort(1:end-1) + y_sort(2:end))/2, Inf];
      
    % k-1 best values has its own bin
    case 'best'
      edges = [-Inf, (y_sort(1:k-1) + y_sort(2:k))/2,  Inf]; 
      
    % agglomerative hierarchical clustering
    case 'cluster'
      y_lin = linkage(pdist(y_sort'));
      b = cluster(y_lin, 'maxclust', k);
      edgeId = find(diff(b) ~= 0);
      
      edges = [-Inf, (y_sort(edgeId) + y_sort(edgeId + 1))/2,  Inf];
      
    % evenly spaced cumulative probabilities of points
    case {'quant', 'quantile'}
      edges = [-Inf, quantile(y_sort, k-1),  Inf]; 
      
    % uniformly distributed range
    case {'unif', 'uniform'}
      difference = range(y) / (k-1);
      edges = [-Inf, min(y) + difference/2 : difference : max(y) - difference/2, Inf];
      
    % uniformly distributed points
    case {'unip', 'unipoints'}
      rem = mod(n, k);
      binSize = [floor(n/k)*ones(1, k - rem), ceil(n/k)*ones(1, rem)];
      pid = cumsum(binSize(1:end-1));
      edges = [-Inf, (y_sort(pid) + y_sort(pid + 1))/2, Inf];
      
    otherwise
      error('Undefined binning type: ''%s''', type)
  end
  
  % shift back to exponencial if necessary
  if logType
    if isShifted
      edges(2:end-1) = exp(edges(2:end-1)) + min(y, [], 1) - eps;
    else
      edges(2:end-1) = exp(edges(2:end-1));
    end
  end
  
  % return bins according to data
  b = arrayfun(@(x) find(x < edges, 1) - 1, y);
end

