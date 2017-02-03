function [logp, dlogp, muloo, s2loo] = negLogPredProbGpml(hyp, nHypCov, covFcn, X, y, dbg)
% NEGLOGPREDPROB Negative logarithm of predictive probability. 
% The objective function that is to be optimized. (Eq. 11)
%
% Input:
%   hyp     - hyperparameter vector [alpha beta1 delta cov sigman]
%   nHypCov - number of covariance function hyperparameters
%   covFcn  - covariance function handle 
%   X       - input data | N x dim double
%   y       - function values for input X | N x 1 double
%   dbg     - debugging mode | boolean
%
% Output:
%   logp  - negative logarithm of probability | double
%   dlogp - derivative of logp | double
%   muloo - leave-one-out mean prediction for X | N x 1 double
%   s2loo - leave-one-out variance prediction for X | N x 1 double

  if nargin < 6
    dbg = false;
  end
  
  % default settings
  
  meanFcn = @meanZero;
  likFcn  = @likGauss;
  infFcn  = @infExact;
      
  % TODO: solve this
  % dlogp is not computed
  dlogp = [];

  % parse input
  hypPlsor = hyp(1:end-1-nHypCov);
  % hyperparameters are supposed to be positive
  hyp_gpml.cov = hyp(end-nHypCov:end-1);
  hyp_gpml.lik = hyp(end);
  % set the mean hyperparameter if is needed
  if (~isequal(meanFcn, @meanZero))
    hyp_gpml.mean = median(y);
  end
  n = size(X, 1);
  
  % default values
  muloo = zeros(n, 1);
  s2loo = ones(n, 1);
  
  % leave-one-out using gpml
  for i = 1:n
    id = [(1:i-1), (i+1:n)];
      % leave-one-out predictive mean and variance (Eq. 12 and 13)
    [muloo(i), s2loo(i)] = gp(hyp_gpml, infFcn, meanFcn, covFcn, likFcn, ...
      X(id, :), y(id), X(i, :));
  end

  % predictive probabilities
  p = predProb(y, hypPlsor, muloo, s2loo);
  % negative logarithm of predictive probability
  logp = -sum(reallog(p));
  
end

