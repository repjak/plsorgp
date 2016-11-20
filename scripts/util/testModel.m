function stats = testModel(Xtr, Ytr, Xte, Yte, settings, statFcns)
  tic;
  ordgp = OrdRegressionGP(Xtr, Ytr, settings);
  t1 = toc;

  Ypred = ordgp.predict(Xte);

  nargouts = arrayfun(@(i) statFcns(i).nargout, 1:length(statFcns));
  stats = zeros(1, sum(nargouts));

  for i = 1:length(statFcns)
    statFcn = statFcns(i);
    stats(i) = statFcn.fcn(ordgp, Ypred, Yte, t1);
  end
end

