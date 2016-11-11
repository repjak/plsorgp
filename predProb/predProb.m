function p = predProb(ys, hyp, mu, s2)
  alpha = hyp(1);
  beta1 = hyp(2);
  delta = hyp(3:end);

  a = alpha * mu;
  q1 = arrayfun(@(y) a + betai(y, beta1, delta), ys);
  q2 = arrayfun(@(y) a + betai(y - 1, beta1, delta), ys);
  w = sqrt(1+alpha^2 * s2);

  p = bsxfun(@(x, y) normcdf(x/w) - normcdf(y/w), q1, q2);
end

