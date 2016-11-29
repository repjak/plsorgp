classdef OrdRegressionGP < handle
  %ORDREGRESSIONGP An Ordinal Regression Gaussian Process model.

  properties (GetAccess = protected, SetAccess = private, Hidden = true)
    muX = 0             % predictors mean for standardization
    stdX = 1            % predictors std for standardization
    n                   % the number of observations
    r                   % the number of ordinal classes
    d                   % the dimensionality of the input space
    yUnq                % uniquely sorted response values
    yUnqIdx             % indices of the sorted response values
    ys                  % response values transformed into classes 1:r
    covFcn              % the covariance function
    hyp = struct()      % a structure of model's hyperparameters
    lb = struct()       % hyperparameters lower bounds
    ub = struct()       % hyperparameters upper bounds
    nHyp                % hyperparameters total
    nHypCov             % the number of the covariance fcn's hyperparameters
    nlpFcn              % the negative log probability (a pseudo likelihood)
    optRes              % a structure of optimization results
    lossFcn             % a loss function for predictions
    fitMethod = 'exact' % the fitting method, either 'exact' or 'none'
    nRandomPoints = 1   % the number of random initial points for fitting
    standardize = true  % standardize the training data
    optimopts           % optimizer options
    minNlp = Inf        % the optimal negative log probability
    K                   % the covariance matrix on the training data
    R                   % the upper-triangular cholesky factor of the covariance matrix
    Kinvy               % (K + sigma2 * eye(n)) \ y
  end

  properties(GetAccess = public, SetAccess = private)
    X                        % the training predictors
    y                        % the training response values
    NumObservations          % the number of observations
    NumClasses               % the number of ordinal classes
    Dimensionality           % the dimensionality of the input space
    Standardize = true       % standardize the training data
    FitMethod                % the fitting method, either 'exact' or 'none'
    CrossVal                 % the crossvalidation partitiong, 'leave1out'
    LossFunction             % the loss function for deciding prediction
    KernelFunction           % the GP kernel function
    KernelParameters         % the hyperparameters of the GP kernel
    PlsorParameters          % the hyperparameters of the PLSOR method
    OptimizerOptions         % optimizer options
    Sigma2                   % the GP's Gaussian noise variance
    OptimInfo                % optimizer's result info
    OptimExitFlag            % optimizer's exit flag
    OptimTrial               % the best starting point if multistart is used
    MinimumNLP               % optimized value of negative log probability
  end

  methods
    function obj = OrdRegressionGP(data, resp, varargin)
      %ORDREGRESSIONGP Fit a Probabilistic Least Squares Ordinal Regression
      %Gaussian Process model.
      %   model = ORDREGRESSIONGP(tbl, ResponseVarName)
      %   model = ORDREGRESSIONGP(tbl, formula)
      %   model = ORDREGRESSIONGP(tbl, y)
      %   model = ORDREGRESSIONGP(X, y)
      %   model = ORDREGRESSIONGP(__, Name, Value) specify Name-Value pairs
      %   using any of the previous syntaxes.
      %   model = ORDREGRESSIONGP(__, args) specify Name-Value pairs in
      %   a cell array.

      if nargin == 0
        return;
      end

      % input arguments
      p = inputParser;
      p.addRequired('data', @(x) ~isempty(x) && (istable(x) || isnumeric(x)));
      p.addRequired('resp', ...
        @(x) ~isempty(x) && (ischar(x) || (isnumeric(x) && size(x,2) == 1)));

      p.addParameter('CrossVal', 'leave1out', ...
        @(x) validateattributes(x, {'char'}, {'leave1out'}));
      p.addParameter('Standardize', true, ...
        @(x) validateattributes(x, {'logical'}, {'size', [1, 1]}));
      covFcns = {'squaredexponential', 'ardsquaredexponential'};
      p.addParameter('KernelFunction', 'ardsquaredexponential', ...
        @(x) (ischar(x) && ismember(x, covFcns)) || ...
        isa(x, 'function_handle'));
      p.addParameter('KernelParameters', [], ...
        @(x) validateattributes(x, {'numeric'}, {'nonempty'}));
      p.addParameter('PlsorParameters', [], ...
        @(x) validateattributes(x, {'numeric'}, {'nonempty'}));
      p.addParameter('FitMethod', 'exact', ...
        @(x) validateattributes(x, {'char'}, {'none', 'exact'}));
      p.addParameter('Sigma2', 0, ...
        @(x) validateattributes(x, {'double'}, {'positive'}));
      p.addParameter('OptimizerOptions', [], ...
        @(x) isa(x, 'optim.options.Fmincon'));
      p.addParameter('LossFunction', 'zeroone', ...
        @(x) (ischar(x) && ismember(x, {'zeroone', 'abserr'})) || ...
        isa(x, 'function_handle'));

      if nargin >= 3 && iscell(varargin{1})
        arglist = varargin{1};
      else
        arglist = varargin;
      end

      p.parse(data, resp, arglist{:});

      % extract data
      if istable(p.Results.data)
        [obj.X, obj.y] = extractTableData(p.Results.data, p.Results.resp);
      else
        obj.X = p.Results.data;
        obj.y = p.Results.resp;

        if ~isnumeric(obj.y)
          error(['If predictor data are given in a matrix than the response' ...
            'must be a column vector.']);
        end

        if size(obj.X, 1) ~= length(obj.y)
          error('Dimensions don''t match.');
        end
      end

      obj.d = size(obj.X, 2);

      % remove NaNs
      notnans = ~logical(sum([isnan(obj.X) isnan(obj.y)], 2));
      obj.X = obj.X(notnans, :);
      obj.y = obj.y(notnans, :);

      if (size(obj.X, 1) < 2)
        error('Data not large enough for training after NaNs removal.');
      end

      % standardize
      if p.Results.Standardize
        obj.muX = mean(obj.X);
        obj.stdX = std(obj.X);
        obj.X = bsxfun(@minus, obj.X, obj.muX);

        if obj.stdX > 0
          obj.X = bsxfun(@rdivide, obj.X, obj.stdX);
        end
      end
      obj.standardize = p.Results.Standardize;

      % map targets into 1:r
      [obj.yUnq, obj.yUnqIdx, obj.ys] = unique(obj.y, 'sorted');
      obj.r = length(unique(obj.ys, 'sorted'));

      obj.n = size(obj.X, 1);
      obj.d = size(obj.X, 2);

      % initialize hyperparameters
      obj.hyp = struct();
      obj.hyp.cov = p.Results.KernelParameters;
      obj.hyp.plsor = p.Results.PlsorParameters;

      % set default kernel hyperparameter values
      if ischar(p.Results.KernelFunction)
        switch p.Results.KernelFunction
          case 'squaredexponential'
            obj.covFcn = @sqexp;
            if ~isempty(obj.hyp.cov) && length(obj.hyp.cov) ~= 2
              error('Kernel function ''%s'' takes 2 hyperparameters.', ...
                p.Results.KernelFunction);
            elseif isempty(obj.hyp.cov)
              obj.hyp.cov = log([1 sqrt(obj.d)]);
            else
              obj.hyp.cov = reshape(obj.hyp.cov, 1, numel(obj.hyp.cov));
            end

            obj.ub.cov = max(obj.hyp.cov, log([sqrt(1e1) 1e1]));
          case 'ardsquaredexponential'
            obj.covFcn = @sqexpard;
            if ~isempty(obj.hyp.cov) && length(obj.hyp.cov) ~= d + 1
              error('Kernel function ''%s'' takes dim + 2 hyperparameters.', ...
                p.Results.KernelFunction);
            elseif isempty(obj.hyp.cov)
              obj.hyp.cov = log([1 ones(1, obj.d)]);
            else
              obj.hyp.cov = reshape(obj.hyp.cov, 1, numel(obj.hyp.cov));
            end

            obj.ub.cov = max(obj.hyp.cov, ...
              log([sqrt(1e1) 1e1 * ones(1, obj.d) / sqrt(obj.d)]));
          otherwise
            error('Unknown kernel function ''%s''.', p.Results.KernelFunction);
        end
      else
        obj.covFcn = p.Results.KernelFunction;
        covFcnInfo = functions(obj.covFcn);
        if isempty(obj.hyp.cov)
          error('No hyperparameters for a user supplied kernel function ''%s''.', ...
            covFcnInfo.function);
        end

        obj.ub.cov = max(obj.hyp.cov, ...
          1e1 + zeros(1, length(obj.hyp.cov)));
      end

      obj.lb.cov = min(obj.hyp.cov, ...
        log([sqrt(1e-3) 1e-3 * ones(1, length(obj.hyp.cov) - 1)]));

      obj.nHypCov = length(obj.hyp.cov);

      % set the noise hyperparameter
      if p.Results.Sigma2 == 0
        obj.hyp.sigma2 = var(obj.ys) / 2;
      else
        obj.hyp.sigma2 = p.Results.Sigma2;
      end

      obj.lb.sigma2 = min(obj.hyp.sigma2, 1e-9);
      obj.ub.sigma2 = max(obj.hyp.sigma2, 2 * var(obj.ys));

      % set default plsor values
      if isempty(obj.hyp.plsor)
        alpha = 1;
        delta = repmat(2 / obj.r, 1, obj.r - 2);
        beta1 = -1;
        obj.hyp.plsor = [alpha beta1 delta];
      end

      obj.lb.plsor = min(obj.hyp.plsor, ...
        [-Inf -Inf 1e-9 + zeros(1, length(obj.hyp.plsor)-2)]);
      obj.ub.plsor = Inf(1, length(obj.hyp.plsor));

      obj.nHyp = length(obj.hyp.cov) + length(obj.hyp.plsor) + 1;

      if ischar(p.Results.LossFunction)
        switch p.Results.LossFunction
          case 'zeroone'
            obj.lossFcn = @zeroone;
          case 'abserr'
            obj.lossFcn = @abserr;
          otherwise
            error('Unknown loss function ''%s''.', p.Results.LossFunction);
        end
      else
        obj.lossFcn = p.Results.LossFunction;
      end

      obj.fitMethod = p.Results.FitMethod;

      switch p.Results.CrossVal
        case 'leave1out'
          covFcn = obj.covFcn;
          nHypCov = obj.nHypCov;
          X = obj.X;
          y = obj.ys;
          n = obj.n;

          obj.nlpFcn = @(hyp) ...
            negLogPredProb(hyp, nHypCov, covFcn, X, y, n);
        otherwise
          error('Cross-validation ''%s'' not supported.', p.Results.CrossVal);
      end

      if isempty(p.Results.OptimizerOptions)
        obj.optimopts = optimoptions( ...
          @fmincon, ...
          'GradObj', 'on', ...
          'Display', 'off', ...
          'MaxIter', 3e3, ...
          'Algorithm', 'interior-point' ...
        );
      else
        obj.optimopts = p.Results.OptimizerOptions;
      end

      obj.fit();

      % precompute the covariance matrix for prediction calls
      obj.K = obj.covFcn(obj.X, obj.X, obj.hyp.cov);
      obj.R = chol(obj.K + obj.hyp.sigma2 * eye(n));
      obj.Kinvy = cholsolve(obj.R, obj.ys);
    end

    function [y, p, mu, s2] = predict(obj, Xnew)
      %ORDREGRESSIONGP.PREDICT An ordinal probabilistic prediction.
      %   [y, p]         = ORDREGRESSIONGP.PREDICT(Xnew) return a column
      %   vector of predicted classes and a column vector of predicted
      %   probabilities for N-R data Xnew.
      %   [y, p, mu, s2] = ORDREGRESSIONGP.PREDICT(Xnew) return also the
      %   predictive mean and variance of the latent variable.

      if size(Xnew, 2) ~= obj.d
        error('Input dimensionality %d does not agree.', size(Xnew, 2));
      end

      % normalize the data
      Xnew = bsxfun(@minus, Xnew, obj.muX);

      if obj.stdX > 0
        Xnew = bsxfun(@rdivide, Xnew, obj.stdX);
      end

      m = size(Xnew, 1);

      % GP prediction assuming Gaussian likelihood
      [mu, s2] = gpPred(obj.X, [], Xnew, obj.covFcn, obj.hyp.cov, ...
        [], obj.R, obj.Kinvy);

      % probabilistic predictions for all classes and all test data
      P = zeros(m, obj.r);
      for i = 1:m
        P(i, :) = predProb(1:obj.r, obj.hyp.plsor, ...
          repmat(mu(i), obj.r, 1), repmat(s2(i), obj.r, 1));
      end

      [~, idx, predProbs] = obj.lossFcn(P);

      % map the predicted ordinal class index back to the input range
      y = obj.yUnq(idx);
      p = predProbs;
    end

    function hyp = get.KernelParameters(obj)
      s = functions(obj.covFcn);
      if strcmp(s, 'sqexp') || strcmp(s, 'sqexpard')
        hyp = [exp(2 * obj.hyp.cov(1)) exp(obj.hyp.cov(2:end))];
      else
        hyp = obj.hyp.cov;
      end
    end

    function hyp = get.PlsorParameters(obj)
      hyp = obj.hyp.plsor;
    end

    function sigma2 = get.Sigma2(obj)
      sigma2 = obj.hyp.sigma2;
    end

    function n = get.NumObservations(obj)
      n = obj.n;
    end

    function r = get.NumClasses(obj)
      r = obj.r;
    end

    function d = get.Dimensionality(obj)
      d = obj.d;
    end

    function cv = get.CrossVal(~)
      cv = 'leave1out';
    end

    function lossFcn = get.LossFunction(obj)
      s = functions(obj.lossFcn);
      lossFcn = s.function;
    end

    function covFcn = get.KernelFunction(obj)
      s = functions(obj.covFcn);
      switch s
        case 'sqexp'
          covFcn = 'squaredexponential';
        case 'sqexpard'
          covFcn = 'ardsquaredexponential';
        otherwise
          covFcn = s.function;
      end
    end

    function fitMethod = get.FitMethod(obj)
      fitMethod = obj.fitMethod;
    end

    function optimopts = get.OptimizerOptions(obj)
      optimopts = obj.optimopts;
    end

    function minNlp = get.MinimumNLP(obj)
      minNlp = obj.minNlp;
    end
  end

  methods (Access = private)
    function fit(obj)
      switch obj.fitMethod
        case 'exact'
          startPoints = zeros(2, obj.nHyp);
          startPoints(1, :) = [obj.hyp.plsor obj.hyp.cov sqrt(obj.hyp.sigma2)];

          for i = 2:(obj.nRandomPoints + 1)
            undef = true;

            while undef
              b = sort(2 * (obj.r - 1) * rand(1, obj.r - 1));
              alpha = 4 * rand() - 2;
              beta1 = 2 * rand() - 1;
              delta = arrayfun(@(i) b(i) - b(i - 1), 2:(obj.r - 1));
              hyp0 = [min(obj.ub.plsor, max(obj.lb.plsor, [alpha beta1 delta])) ...
                obj.hyp.cov sqrt(obj.hyp.sigma2)];
              y0 = obj.nlpFcn(hyp0);
              undef = isinf(y0) || isnan(y0);
            end

            startPoints(i, :) = hyp0;
          end

          optproblem = struct( ...
            'solver', 'fmincon', ...
            'objective', obj.nlpFcn, ...
            'lb', [obj.lb.plsor obj.lb.cov obj.lb.sigma2], ...
            'ub', [obj.ub.plsor obj.ub.cov obj.ub.sigma2], ...
            'options', obj.optimopts ...
          );

          for i = 1:size(startPoints, 1)
            optproblem.x0 = startPoints(i, :);

            warning('off', 'MATLAB:nearlySingularMatrix');

            try
              [minx, miny, exitflag, optinfo] = fmincon(optproblem);
            catch err
              report = getReport(err);
              warning('fmincon in trial %d failed with error:\n%s', i, ...
                report);
              continue;
            end

            warning('on', 'MATLAB:nearlySingularMatrix');

            if miny < obj.minNlp;
              obj.hyp.plsor = minx(1:length(obj.hyp.plsor));
              obj.hyp.cov = minx(length(obj.hyp.plsor)+1:end-1);
              obj.hyp.sigma2 = minx(end)^2;

              obj.minNlp = miny;
              obj.OptimInfo = optinfo;
              obj.OptimExitFlag = exitflag;
              obj.OptimTrial = i;
            end
          end
        case 'none'
          return;
        otherwise
          error('Fit method ''%s'' not supported', obj.fitMethod);
      end
    end

  end

end

