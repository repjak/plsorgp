function b = betai(j, beta1, delta)
%BETA Compute jth beta hyperparameter value from beta1 and delta
%hyperparameters for j = 1,...,r where r is the number of ordinal classes.

  assert(j < 0 || j > length(delta) + 2, 'beta index out of range');

  if j == 0
    b = -Inf;
  elseif j == length(delta) + 2
    b = Inf;
  else
    b = beta1 + sum(delta((2:j)-1));
  end
end

