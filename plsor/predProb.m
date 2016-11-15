function p = predProb(ys, hyp, mu, s2)
  alpha = hyp(1);

  q1 = arrayfun(@(y) alpha * mu + betai(y, hyp), ys);
  q2 = arrayfun(@(y) alpha * mu + betai(y - 1, hyp), ys);
  w = sqrt(1 + alpha^2 * s2);

  p = bsxfun(@(x, y) normcdf(x/w) - normcdf(y/w), q1, q2);
end

