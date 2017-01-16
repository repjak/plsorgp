function binningComparison(y, nBins)
% BINNINGCOMPARISON(y, nBins) compares all implemented binning types by
% splitting data 'y' to 'nBins' bins.
%
% Input:
%   y     - values to bin or number of randomly-generated values
%   nBins - number of bins
%
% See Also:
%   binning

  if nargin < 2
    nBins = 5;
    if nargin < 1
      y = 30;
    end
  end
  
  % adjust input
  if length(y) < 2
    y = sort(randn(y, 1));
  else
    n = length(y);
    y = reshape(y, n, 1);
  end
  
  % find all implemented binnings
  bintype = binning('list');
  % perform binning
  bins = cellfun(@(type) binning(y, nBins, type), bintype, 'UniformOutput', false);
  % list results in table
  fprintf('\n')
  disp(table(y, bins{:}, 'VariableNames', ['y', bintype]))
end