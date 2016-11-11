classdef OrdRegressionGP
  %ORDREGRESSIONGP An Ordinal Regression Gaussian Process model.

  properties (GetAccess = protected, SetAccess = private)
    X                   % the predictors
    y                   % user-supplied response values
    muX = 0             % predictors mean for standardization
    stdX = 1            % predictors std for standardization
    n                   % the number of observations
    r                   % the number of ordinal classes
    d                   % the dimension of the input space
    yUnq                % uniquely sorted response values
    yUnqIdx             % indices of the sorted response values
    ys                  % response values transformed into classes 1:r
    covFcn              % the covariance function
    hyp = struct()      % a structure of model's hyperparameters
    nHyp                % the number of the covariance fcn's hyperparameters
    nlp = Inf           % the optimized negative log probability
    optRes = struct()   % a structure of optimization results
    lossFcn = @zeroone  % a loss function for predictions
    K                   % the covariance matrix on the training data
    R                   % the upper-triangular cholesky factor of the covariance matrix
    Kinvy               % (K + sigma2 * eye(n)) \ y
  end

  methods
    function model = OrdRegressionGP(data, resp, varargin)
      %ORDREGRESSIONGP Fit a Probabilistic Least Squares Ordinal Regression Gaussian
      %Process model.
      %   model = ORDREGRESSIONGP(tbl, ResponseVarName)
      %   model = ORDREGRESSIONGP(tbl, formula)
      %   model = ORDREGRESSIONGP(tbl, y)
      %   model = ORDREGRESSIONGP(X, y)
      %   model = ORDREGRESSIONGP(__, Name, Value)

      if nargin == 0
        return;
      end

      % input arguments
      p = inputParser;
      p.addRequired('data', @(x) ~isempty(x) && (istable(x) || isnumeric(x)));
      p.addRequired('resp', ...
        @(x) ~isempty(x) && (ischar(x) || (isint(x) && size(x,2) == 1)));

      p.addParameter('CrossVal', 'leave1out', ...
        @(x) validateattributes({'char'}, {'leave1out'}));
      p.addParameter('Standardize',      true, ...
        @(x) validateattributes({'logical'}, {'size', [1, 1]}));
      covFcns = {'squaredexponential', 'ardsquaredexponential'};
      p.addParameter('KernelFunction', 'ardsquaredexponential', ...
        @(x) (ischar(x) && ismember(x, covFcns)) || isa(x, 'function_handle'));
      p.addParameter('KernelParameters', [], ...
        @(x) validateattributes({'numeric'}, {'nonempty'}));
      p.addParameter('PlsorParameters', [], ...
        @(x) validateattributes({'numeric'}, {'nonempty'}));
      p.addParameter('FitMethod', ...
        @(x) validateattributes({'char'}, {'none', 'exact'}));
      p.addParameter('Sigma2', 0, ...
        @(x) validateattributes({'double'}, {'positive'}));
      p.addParameter('OptimizerOptions', [], ...
        @(x) isa(x, 'optim.options.Fmincon'));
      p.addParameter('LossFunction', 'zeroone', ...
        @(x) validateattributes({'char'}, {'zeroone', 'hinge'}));

      p.parse(data, resp, varargin{:});

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
        obj.X = bsxfun(@divide, obj.X, obj.stdX);
      end

      % determine ordinal classes
      [obj.yUnq, obj.yUnqIdx, obj.ys] = unique(y, 'sorted');
      obj.r = length(unique(ys, 'sorted'));

      obj.n = size(X, 1);
      obj.d = size(X, 2);

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
              obj.hyp.cov = [1 1/d];
            else
              obj.hyp.cov = reshape(obj.hyp.cov, 1, numel(obj.hyp.cov));
            end
          case 'ardsquaredexponential'
            obj.covFcn = @sqexpard;
            if ~isempty(obj.hyp.cov) && length(obj.hyp.cov) ~= d + 1
              error('Kernel function ''%s'' takes dim + 2 hyperparameters.', ...
                p.Results.KernelFunction);
            elseif isempty(obj.hyp.cov)
              hyp.cov = [1 ones(d, 1)/d];
            else
              hyp.cov = reshape(obj.hyp.cov, 1, numel(obj.hyp.cov));
            end
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
      end

      switch p.Results.LossFunction
        case 'zeroone'
          obj.lossFcn = @zeroone;
        case 'hinge'
          obj.lossFcn = @hinge;
        otherwise
          error('Unknown loss function ''%s''.', p.Results.LossFunction);
      end

      % set the noise hyperparameter
      if p.Results.Sigma2 == 0
        obj.hyp.sigma2 = std(ys) / sqrt(2);
      else
        obj.hyp.sigma2 = p.Results.SigmaNoise;
      end

      % set default plsor values
      if isempty(hyp.plsor)
        alfa = -1;
        deltas = repmat(ceil(range(ys) / r), 1, r - 2);
        betas = deltas(1) .* (1:r-2);
        obj.hyp.plsor = [alfa beta1 betas];
      end

      obj.nHyp = length(hyp.cov) + length(hyp.plsor) + 1;

      switch p.Results.CrossVal
        case 'leave1out'
          nlp = @(hyp) negLogPredProb(hyp, length(hyp.cov), covFcn, ...
            @logPredProbLeaveOneOut, X, ys, n);
        otherwise
          error('Cross-validation ''%s'' not supported.', p.Results.CrossVal);
      end

      switch p.Results.FitMethod
        case 'exact'
          if isempty(p.Results.OptimizerOptions)
            optimopts = optimoptions( ...
              @fmincon, ...
              'GradObj', 'on' ...
            );
          else
            optimopts = p.Results.OptimizerOptions;
          end

          optproblem = struct( ...
            'objective', nlp, ...
            'x0', [hyp.plsor hyp.cov hyp.sigma], ...
            'lb', [-Inf -Inf ...
              eps+[zeros(1, length(hyp.plsor)-2) zeros(1, length(hyp.cov)+1)]], ...
            'options', optimopts ...
          );

          [hyp0, p, exitflag, optinfo] = fmincon(optproblem);

           obj.optRes = struct( ...
            'min', hyp0, ...
            'minVal', p, ...
            'exitflag', exitflag, ...
            'optinfo', optinfo ...
          );

          obj.hyp.plsor = hyp0(1:length(obj.hyp.plsor));
          obj.hyp.cov = hyp0(length(obj.hyp.plsor)+1:end-1);
          obj.hyp.sigma2 = hyp0(end);

          obj.nlp = p;
        case 'none'
        otherwise
          error('Fit method ''%s'' not supported', p.Results.FitMethod);
      end

      obj.K = obj.covFcn(obj.X, obj.X, obj.hyp.cov);
      obj.R = chol(obj.K + obj.hyp.sigma2 * eye(n));
      obj.Kinvy = cholsolve(R, obj.ys);
    end

    function [y, p] = predict(obj, Xnew)
      %ORDREGRESSIONGP.PREDICT An ordinal probabilistic prediction.
      %   [y, p] = ORDREGRESSIONGP.PREDICT(Xnew) return a column vector of
      %   predicted classes and a column vector of predicted probabilities
      %   for N-R data Xnew.

      if size(Xnew, 2) ~= obj.d
        error('Input dimensionality %d does not agree.', size(Xnew, 2));
      end

      % normalize the data
      Xnew = bsxfun(@minus, Xnew, obj.muX);
      Xnew = bsxfun(@rdivide, Xnew, obj.stdX);

      % GP prediction assuming Gaussian likelihood
      [mu, s2] = gpPred(obj.X, [], Xnew, obj.covFcn, obj.hyp.cov, ...
        [], obj.R, obj.Kinvy);

      % probabilistic predictions for all classes and all test data
      P = zeros(m, obj.r);
      for i = 1:m
        P(i, :) = predProb(1:obj.r, obj.hyp.plosr, mu(i), s2(i));
      end

      [~, idx, predProbs] = obj.lossFcn(P);

      % map the predicted ordinal class index back to the input range
      y = obj.yUnq(idx);
      p = predProbs;
    end

  end

end

