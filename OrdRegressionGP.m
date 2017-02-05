classdef OrdRegressionGP < handle
  % ORDREGRESSIONGP An Ordinal Regression Gaussian Process model.
  % Based on P.K.Srijith (2012): 'A Probability Least Squares Approach to 
  % Ordinal Regression'.

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
    KernelBounds             % the bounds of kernel hyperparameters
    KernelParameters         % the hyperparameters of the GP kernel
    PlsorParameters          % the hyperparameters of the PLSOR method
    OptimizerOptions         % optimizer options
    Sigma2                   % the GP's Gaussian noise variance
    Sigma2Bounds             % the bounds of GP's Gaussian noise variance
    OptimInfo                % optimizer's result info
    OptimExitFlag            % optimizer's exit flag
    OptimTrial               % the best starting point if multistart is used
    MinimumNLP               % optimized value of negative log probability
    NumStartPoints           % the number of initial points for fitting
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
        @(x) ischar(x) && ismember(x, {'leave1out'}));
      p.addParameter('Standardize', true, ...
        @(x) validateattributes(x, {'logical'}, {'size', [1, 1]}));
      covFcns = {'squaredexponential', 'ardsquaredexponential'};
      p.addParameter('KernelFunction', 'squaredexponential', ...
        @(x) ((ischar(x) && ismember(x, covFcns))) || ...
        isa(x, 'function_handle') || iscell(x));
      p.addParameter('KernelParameters', [], ...
        @(x) validateattributes(x, {'numeric'}, {'nonempty'}));
      p.addParameter('KernelBounds', [], ...
        @(x)isnumeric(x))
      p.addParameter('PlsorParameters', [], ...
        @(x) validateattributes(x, {'numeric'}, {'nonempty'}));
      p.addParameter('FitMethod', 'exact', ...
        @(x) ischar(x) && ismember(x, {'none', 'exact'}));
      p.addParameter('Sigma2', [], ...
        @(x) isnumeric(x));
      p.addParameter('Sigma2Bounds', [], ...
        @(x) isnumeric(x));
      p.addParameter('OptimizerOptions', [], ...
        @(x) isa(x, 'optim.options.Fmincon'));
      p.addParameter('LossFunction', 'zeroone', ...
        @(x) (ischar(x) && ismember(x, {'zeroone', 'abserr'})) || ...
        isa(x, 'function_handle'));
      p.addParameter('NumStartPoints', 5, ...
        @(x) validateattributes(x, {'numeric'}, {'nonempty', 'integer', ...
        'positive'}));

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

            % kernel bounds should be [cov1lb, cov1ub; cov2lb, cov2ub; ...]
            if isempty(p.Results.KernelBounds)
              obj.KernelBounds = [-2*ones(length(obj.hyp.cov), 1), ...
                                   2*ones(length(obj.hyp.cov), 1)];
            else
              obj.KernelBounds = p.Results.KernelBounds;
            end

            obj.ub.cov = max([obj.hyp.cov + eps; ...
                              log([sqrt(1e1), 1e1]); ...
                              obj.KernelBounds(:, 2)']);
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

            % kernel bounds should be [cov1lb, cov1ub; cov2lb, cov2ub; ...]
            if isempty(p.Results.KernelBounds)
              obj.KernelBounds = [-2*ones(length(obj.hyp.cov), 1), ...
                                   2*ones(length(obj.hyp.cov), 1)];
            else
              obj.KernelBounds = p.Results.KernelBounds;
            end

            obj.ub.cov = max([obj.hyp.cov + eps; ...
              log([sqrt(1e1) 1e1 * ones(1, obj.d) / sqrt(obj.d)]); ...
              obj.KernelBounds(:, 2)']);
          otherwise
            error('Unknown kernel function ''%s''.', p.Results.KernelFunction);
        end
      else
        obj.covFcn = p.Results.KernelFunction;
        if isa(obj.covFcn, 'function_handle')
          % function handle name
          covFcnName = [' ''', func2str(obj.covFcn), ''''];
        else
          covFcnName = '';
        end
        if isempty(obj.hyp.cov)
          error('No hyperparameters for a user supplied kernel function%s.', ...
            covFcnName);
        end

        % kernel bounds should be [cov1lb, cov1ub; cov2lb, cov2ub; ...]
        if isempty(p.Results.KernelBounds)
          obj.KernelBounds = [-2*ones(length(obj.hyp.cov), 1), ...
                               2*ones(length(obj.hyp.cov), 1)];
        else
          obj.KernelBounds = p.Results.KernelBounds;
        end

        % standardize hyperparameters input
        obj.hyp.cov = reshape(obj.hyp.cov, 1, numel(obj.hyp.cov));
        obj.ub.cov = max(obj.hyp.cov + eps, ...
                         obj.KernelBounds(:, 2)');
      end

      obj.lb.cov = min(obj.hyp.cov - eps, ...
                       obj.KernelBounds(:, 1)');

      obj.nHypCov = length(obj.hyp.cov);
      
      % set the number of starting points
      obj.NumStartPoints = p.Results.NumStartPoints;

      % set the noise hyperparameter
      if isempty(p.Results.Sigma2)
        obj.hyp.sigma2 = log(var(obj.ys) / 2);
      else
        obj.hyp.sigma2 = p.Results.Sigma2;
      end

      if isempty(p.Results.Sigma2Bounds)
        obj.Sigma2Bounds = [log(1e-6), log(1e1)];
      else
        obj.Sigma2Bounds = p.Results.Sigma2Bounds;
      end

      obj.lb.sigma2 = min(obj.hyp.sigma2 - eps, obj.Sigma2Bounds(1));
      obj.ub.sigma2 = max(obj.hyp.sigma2 + eps, obj.Sigma2Bounds(2));

      % initialize probability distribution
      pd = makedist('Normal', 0, 1);
      % compute cdf bound value
      cdfb = abs(icdf(pd, eps)) / 2;

      % set default plsor values
      if isempty(obj.hyp.plsor)
        alpha = 1;
        delta = repmat(2 / obj.r, 1, obj.r - 2);
        beta1 = -1;
        obj.hyp.plsor = [alpha beta1 delta];
      end

      alphaBnd = 1e3;
      obj.lb.plsor = min(obj.hyp.plsor, ...
        [-alphaBnd, -Inf, zeros(1, length(obj.hyp.plsor)-2)]);
      obj.ub.plsor = [alphaBnd, repmat(1000, 1, length(obj.hyp.plsor)-1)];

      obj.nHyp = length(obj.hyp.cov) + length(obj.hyp.plsor) + 1;

      % set loss function
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

      % set fitting method
      obj.fitMethod = p.Results.FitMethod;

      % set cross-validation method
      switch p.Results.CrossVal
        case 'leave1out'
          covFcn = obj.covFcn;
          nHypCov = obj.nHypCov;
          X = obj.X;
          y = obj.ys;
          n = obj.n;

          obj.nlpFcn = @(hyp) ...
            negLogPredProb(hyp, nHypCov, covFcn, X, y);
        otherwise
          error('Cross-validation ''%s'' not supported.', p.Results.CrossVal);
      end
      
      % set optimizer options
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
      
      % fit model
      obj.fit();

      % non-gpml
      % precompute the covariance matrix for prediction calls
      if iscell(obj.covFcn)
        obj.K = feval(obj.covFcn{:}, obj.hyp.cov, obj.X);
      else
        obj.K = obj.covFcn(obj.X, obj.X, obj.hyp.cov);
      end

%       obj.R = chol(obj.K/exp(obj.hyp.sigma2) + eye(n) + 0.0001*eye(n));
      obj.R = chol(obj.K + exp(obj.hyp.sigma2) * eye(n));
      obj.Kinvy = cholsolve(obj.R, obj.ys);
    end

    function [y, p, mu, s2, e] = predict(obj, Xnew)
      %ORDREGRESSIONGP.PREDICT An ordinal probabilistic prediction.
      %   [y, p]         = ORDREGRESSIONGP.PREDICT(Xnew) return a column
      %   vector of predicted classes and a column vector of predicted
      %   probabilities for N-R data Xnew.
      %   [y, p, mu, s2] = ORDREGRESSIONGP.PREDICT(Xnew) return also the
      %   predictive mean and variance of the latent variable.
      %   [y, p, mu, s2, e] = ORDREGRESSIONGP.PREDICT(Xnew) return also a
      %   column vector of predicted classes weighted using the probability
      %   of individual classes

      if size(Xnew, 2) ~= obj.d
        error('Input dimensionality %d does not agree.', size(Xnew, 2));
      end

      % normalize the data
      if obj.standardize
        Xnew = bsxfun(@minus, Xnew, obj.muX);

        if obj.stdX > 0
          Xnew = bsxfun(@rdivide, Xnew, obj.stdX);
        end
      end

      m = size(Xnew, 1);

      % Do not to use the pre-computed matrices, as it does not work :(
      [mu, s2] = gpPred(obj.X, obj.y, Xnew, obj.covFcn, obj.hyp.cov, exp(obj.hyp.sigma2));
      % [mu, s2] = gpPred(obj.X, [], Xnew, obj.covFcn, obj.hyp.cov, ...
      %   exp(obj.hyp.sigma2), obj.R, obj.Kinvy);

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
      e = P*(1:obj.r)';
    end

    function hyp = get.KernelParameters(obj)
      if iscell(obj.covFcn)
        hyp = obj.hyp.cov;
      else
        s = functions(obj.covFcn);
        if strcmp(s.function, 'sqexp') || strcmp(s.function, 'sqexpard')
          hyp = [exp(2 * obj.hyp.cov(1)) exp(obj.hyp.cov(2:end))];
        else
          hyp = obj.hyp.cov;
        end
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
      if iscell(obj.covFcn)
        covFcn = obj.covFcn;
      else
        s = functions(obj.covFcn);
        switch s.function
          case 'sqexp'
            covFcn = 'squaredexponential';
          case 'sqexpard'
            covFcn = 'ardsquaredexponential';
          otherwise
            covFcn = s.function;
        end
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
          startPoints = zeros(obj.NumStartPoints, obj.nHyp);
          undef = true(1, obj.NumStartPoints);
          
          % user defined starting point
          startPoints(1, :) = [obj.hyp.plsor, obj.hyp.cov, obj.hyp.sigma2/2];
          
          % compute mu and s2
          [~, ~, muloo, s2loo] = obj.nlpFcn(startPoints(1, :));
          
          % initialize probability distribution
          pd = makedist('Normal', 0, 1);
          % compute cdf precision bound value
          cdfb = abs(icdf(pd, eps)) / 2;
          
          % adjust plsor hyperparameter values          
          plsor_hyp = obj.adjustPlsorParams(obj.hyp.plsor, muloo, s2loo, cdfb);
          
          % compute likelihood for user-defined point (adjusted to 
          % precision boundaries)
          startPoints(1, :) = [plsor_hyp, obj.hyp.cov, obj.hyp.sigma2/2];
          y0 = obj.nlpFcn(startPoints(1, :));
          undef(1) = isinf(y0) || isnan(y0);
          
          % random gp hyperparameters
          hyp_rand.cov = (obj.ub.cov - obj.lb.cov).*rand(1, length(obj.lb.cov)) + obj.lb.cov;
          hyp_rand.sigma2 = (obj.ub.sigma2 - obj.lb.sigma2).*rand() + obj.lb.sigma2;
          
          % compute mu and s2
          [~, ~, muloo, s2loo] = obj.nlpFcn([obj.hyp.plsor, hyp_rand.cov, hyp_rand.sigma2/2]);
          
          i = 2;
          % find the rest of feasible starting points by random
          while sum(~undef) < obj.NumStartPoints
            % generate random plsor params
            plsor_hyp = obj.adjustPlsorParams(obj.r, muloo, s2loo, cdfb);
            hyp0 = [min(obj.ub.plsor, max(obj.lb.plsor, plsor_hyp)) ...
                  hyp_rand.cov hyp_rand.sigma2/2];
            y0 = obj.nlpFcn(hyp0);

%             y0 = NaN;        
%             while abs(alpha) < 2 && (isinf(y0) || isnan(y0))
%               alpha = 2 * alpha;
%               for s = [1 -1]
%                 alpha = s * alpha;
%                 hyp0 = [min(obj.ub.plsor, max(obj.lb.plsor, [alpha beta1 delta])) ...
%                   hyp_rand.cov hyp_rand.sigma2/2];
%                 y0 = obj.nlpFcn(hyp0);
%                 if ~isinf(y0) && ~isnan(y0)
%                   break;
%                 end
%               end
%             end
            undef(i) = isinf(y0) || isnan(y0);
            startPoints(i, :) = hyp0;
            i = i+1;
          end

          if any(undef) && ~all(undef)
            warning('%d/%d starting points had undefined value.', sum(undef), length(undef));
            startPoints = startPoints(~undef, :);
          end

          if all(undef)
            warning('The model is untrained: no starting point with defined value found.');
            return;
          end

          optproblem = struct( ...
            'solver', 'fmincon', ...
            'objective', obj.nlpFcn, ...
            'lb', [obj.lb.plsor obj.lb.cov obj.lb.sigma2/2], ...
            'ub', [obj.ub.plsor obj.ub.cov obj.ub.sigma2/2], ...
            'options', obj.optimopts ...
          );

          % find hyperparameters with minimal negative log probability
          % using startPoints
          for i = 1:size(startPoints, 1)
            optproblem.x0 = startPoints(i, :);
            % for hyperparameter debugging and optimization:
            % y0 = obj.nlpFcn(startPoints(i, :));
            % fprintf('[%d]  Alpha: %0.4f  Lik: %0.4f  | ', i,  startPoints(i, 1), y0)

            warning('off', 'MATLAB:nearlySingularMatrix');

            % minimize negative log probability
            try
              [minx, miny, exitflag, optinfo] = fmincon(optproblem);
              % fprintf('[fmincon]  Alpha: %0.4f  Lik: %0.4f\n', minx(1), miny)
            catch err
              fprintf('Error\n')
              report = getReport(err);
              warning('fmincon in trial %d failed with error:\n%s', i, ...
                report);
              continue;
            end

            warning('on', 'MATLAB:nearlySingularMatrix');

            if miny < obj.minNlp;
              obj.hyp.plsor = minx(1:length(obj.hyp.plsor));
              obj.hyp.cov = minx(length(obj.hyp.plsor)+1:end-1);
              obj.hyp.sigma2 = minx(end)*2;

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
  
  methods (Static)
    
    function hyp = adjustPlsorParams(hyp, mu, s2, prb)
    % hyp = adjustPlsorParams(hyp, mu, s2, prec) adjusts plsor
    % hyperparameters to necessary bounds given by cumulative normal
    % distribution function. Parameters alpha, beta1 and betar has to 
    % satisfy the following condition for all mu and s2:
    % -prb < (alpha*mu + beta1)/sqrt(1+s2*alpha^2) <
    % < (alpha*mu + betar)/sqrt(1+s2*alpha^2) < prb.
    % Therefore, if (mu_max-mu_min)^2 > 4*prb^2*s2_min then
    % abs(alpha) < 2*prb/sqrt((mu_max-mu_min)^2 - 4*prb^2*s2_min).
    %
    % hyp = adjustPlsorParams([alpha, beta1, delta], ...) adjusts 
    % plsor hyperparameters to necessary bounds given by cumulative normal
    % distribution function.
    %
    % hyp = adjustPlsorParams(r, ...) generates r-level plsor 
    % hyperparameters within necessary bounds given by cndf.
    %
    % Input:
    %   hyp - plsor hyperparameters [alpha, beta1, delta] or number of
    %         delta parameters
    %   mu  - predicted GP means
    %   s2  - predicted GP variances
    %   prb - chosen precision bound
    
      % compute min and max of mu and s2
      mu_min = min(mu);
      mu_max = max(mu);
      s2_min = min(s2);
      
      % true if alpha has to be bounded
      isAlphaBounded = (mu_max-mu_min)^2 > 4*prb^2*s2_min;
      % bound for alpha
      if isAlphaBounded
        alphaBnd = 2*prb/sqrt((mu_max-mu_min)^2 - 4*prb^2*s2_min);
      else
        % TODO: find appropriate value - now the same as lower and upper
        % bound
        alphaBnd = 1e3;
      end

      % starting point (if empty, generate at random)
      if length(hyp) == 1
        % alpha = sign(alpha)*abs(alpha)
        alpha = (1-2*(randi(2)-1)) * rand()*alphaBnd;
        beta1 = -prb*sqrt(1 + s2_min*alpha^2) + max(-alpha*mu);
        betar =  prb*sqrt(1 + s2_min*alpha^2) + min(-alpha*mu);
        b = sort((betar - beta1) * rand(1, hyp - 1) + beta1);
        delta = arrayfun(@(i) b(i) - b(i - 1), 2:(hyp - 1));
      else
        alpha = hyp(1);
        beta1 = hyp(2);
        delta = hyp(3:end);
      end
      
      % alpha, beta1 and betar has to satisfy the following condition 
      % for all mu and s2:
      % -prb < (alpha*mu + beta1)/sqrt(1+s2*alpha^2) <
      % (alpha*mu + betar)/sqrt(1+s2*alpha^2) < prb
      % Therefore, if (mu_max-mu_min)^2 > 4*prb^2*s2_min then
      % abs(alpha) < 2*prb/sqrt((mu_max-mu_min)^2 - 4*prb^2*s2_min).
      if isAlphaBounded && ( abs(alpha) >= alphaBnd )
        % alpha will be the half of the bound
        alpha = sign(alpha)*alphaBnd/2;
      end
      % compute bounds for beta
      beta_lb = -prb*sqrt(1 + s2_min*alpha^2) + max(-alpha*mu);
      beta_ub =  prb*sqrt(1 + s2_min*alpha^2) + min(-alpha*mu);
      % check beta1
      if ( beta1 < beta_lb ) || ( beta1 > beta_ub )
        beta1 = beta_lb;
      end
      % compute betar
      betar = beta1 + sum(delta);
      % check betar
      if ( betar < beta1 ) || (betar > beta_ub)
        delta = delta*(beta_ub - beta1)/sum(delta);
      end
      
      % return adjusted values
      hyp = [alpha, beta1, delta];
      
    end

  end

end

