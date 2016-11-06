function p = predProb(ys, hyp, mu, s2)
  alpha = hyp(1);
  beta1 = hyp(2);
  delta = hyp(3:end);

  a = alpha * mu;
  q1 = a + beta(ys, beta1, delta);
  q2 = a + beta(ys - 1, beta1, delta);
  w = sqrt(1+alpha^2 * sqr(s2));

  p = normcdf(q1/w) - normcdf(q2/w);
end

