function p = logPredProbLoo(i, y, hyp, nHypCov, muloo, s2loo, dmuloo, ds2loo, j)
%LOGPREDLOO Log predictive probability of the ith training example using the
%remaining training examples.

  assert(length(hyp) >= 2);
  delta = hyp(3:end);

  n = length(y);          % the number of training examples
  r = length(delta) + 2;  % the number of ordinal classes

  assert(y(i) > 0 && y(i) <= r, 'The ordinal class out of range.');
  assert(i >= 1 && i <= n, 'The left-out example index out of range.');

  if nargin < 7
    p = log(predProb(y(i), hyp, muloo(i), s2loo(i)));
  elseif nargin < 8
    error('Not enough input arguments.');
  else
    if nargin < 9
      % add one for the noise hyperparameter
      j = 1:(length(hyp)+nHypCov+1);
    else
      assert(all(j >= 1 && j <= length(hyp) + nHypCov + 1));
    end

    [pval, dp] = predProb(y(i), hyp, muloo(i), s2loo(i), dmuloo(i, :), ds2loo(i, :), j);

    %p = [log(pval) dp ./ pval];
    p = [log(pval) dp ./ pval];
  end
end

