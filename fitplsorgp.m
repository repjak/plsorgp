function model = fitplsorgp(data, resp, varargin)
%FITPSLORGP Fit a Probabilistic Least Squares Ordinal Regression Gaussian
%Process model.
%   model = FITPSLORGP(tbl, ResponseVarName)
%   model = FITPSLORGP(tbl, formula)
%   model = FITPSLORGP(tbl, y)
%   model = FITPSLORGP(X, y)
%   model = FITPSLORGP(__, Name, Value)
%
%   See also PLSORGPMODEL.

  p = inputParser;
  p.addRequired('data', @(x) ~isempty(x) && (istable(x) || isnumeric(x)));
  p.addRequired('resp', ...
    @(x) ~isempty(x) && (ischar(x) || (isint(x) && size(x,2) == 1)));

  p.addParameter('CrossVal', 'leave1out', ...
    @(x) validateattributes({'char'}, {'leave1out'}));
  p.addParameter('Standardize',      true, ...
    @(x) validateattributes({'logical'}, {'size', [1, 1]}));

  covfcns = {'squaredexponential', 'ardsquaredexponential'};
  p.addParameter('KernelFunction', 'ardsquaredexponential', ...
    @(x) (ischar(x) && ismember(x, covfcns)) || isa(x, 'function_handle'));

  p.addParameter('KernelParameters', [], ...
    @(x) validateattributes({'numeric'}, {'nonempty'}));
  p.addParameter('PlsorParameters', [], ...
    @(x) validateattributes({'numeric'}, {'nonempty'}));
  p.addParameter('Optimization', ...
    @(x) validateattributes({'char'}, {'fmincon'}));
  p.addParameter('FitMethod', ...
    @(x) validateattributes({'char'}, {'none', 'exact'}));

  p.parse(data, resp, varargin{:});

  % extract data
  if istable(p.Results.data)
    [X, y] = extractTableData(p.Results.data, p.Results.resp);
  else
    X = p.Results.data;
    y = p.Results.resp;

    if ~isnumeric(y)
      error(['If predictor data are given in a matrix than the response' ...
        'must be a column vector.']);
    end

    if size(X, 1) ~= length(y)
      error('Dimensions don''t match.');
    end
  end

  % remove NaNs
  notnans = ~logical(sum([isnan(X) isnan(y)], 2));
  X = X(notnans, :);
  y = y(notnans, :);

  if (size(X, 1) < 2)
    error('Data not large enough for training after NaNs removal.');
  end

  % standardize
  if p.Results.Standardize
    muX = mean(X);
    sigmaX = std(X);
    X = bsxfun(@minus, X, muX);
    X = bsxfun(@divide, X, sigmaX);
  else
    muX = 0;
    sigmaX = 1;
  end

  % determine ordinal classes
  [yUnq, yUnqIdx, ys] = unique(y, 'sorted');
  C = unique(ys, 'sorted');
  r = length(C);

  n = size(X, 1);
  d = size(X, 2);

  hyp = struct();
  hyp.cov = p.Results.KernelParameters;
  hyp.plsor = p.Results.PlsorParameters;

  % set default kernel hyperparameter values
  if ischar(p.Results.KernelFunction)
    switch p.Results.KernelFunction
      case 'squaredexponential'
        covfcn = @sqexp;
        if ~isempty(hyp.cov) && length(hyp.cov) ~= 2
          error('Kernel function ''%s'' takes 2 hyperparameters.', ...
            p.Results.KernelFunction);
        elseif empty(hyp.cov)
          hyp.cov = [1; 1 / d];
        end
      case 'ardsquaredexponential'
        covfcn = @sqexpard;
        if ~isempty(hyp.cov) && length(hyp.cov) ~= d + 1
          error('Kernel function ''%s'' takes dim + 2 hyperparameters.', ...
            p.Results.KernelFunction);
        elseif empty(hyp.cov)
          hyp.cov = [1; ones(d, 1) / d];
        end
      otherwise
        error('Unknown kernel function ''%s''.', p.Results.KernelFunction);
    end
  else
    covfcn = p.Results.KernelFunction;
    covfcninfo = functions(covfcn);
    if isempty(hyp.cov)
      error('No hyperparameters for a user supplied kernel function ''%s''.', ...
        covfcninfo.function);
    end
  end

  % set default plsor values
  if isempty(hyp.plsor)
    alfa = -1;
    deltas = repmat(ceil(range(ys) / r), r - 2, 1);
    betas = deltas(1) .* (1:r-2)';
    hyp.plsor = [alfa; beta1; betas];
  end

  switch p.Results.FitMethod
    case 'none'
      return;
    case 'exact'
      Xtr = X;
      ytr = ys;
  end

  switch p.Results.CrossVal
    case 'leave1out'
      nlp = @(hyp) negLogPredProb(hyp, covfcn, length(hyp.cov), @negLogPredProb, Xtr, ytr, n);
    otherwise
      error('Cross-validation ''%s'' not supported.', p.Results.CrossVal);
  end

  switch p.Results.Optimization
    case 'fmincon'
      [hyp0, p, exitflag, optinfo] = fmincon(...
        nlp, ...
        [hyp.plsor; hyp.cov]', ...
        [], [], [], [], ...
        [-Inf -Inf zeros(1, length(hyp.plsor)-2)] ...
      );
    otherwise
      error('Optimization ''%s'' not supported.', p.Results.Optimization);
  end

end

