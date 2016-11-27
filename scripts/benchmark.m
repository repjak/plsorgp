% initialize rng for reproducibility
seed = 2;
rng(seed);

% don't actually run the benchmarks
dry = true;

mpath = mfilename('fullpath');
addpath(fullfile(fileparts(mpath), 'util/'));
datapath = fullfile(fileparts(mpath), 'data/');
outpath = fullfile(fileparts(mpath), 'results/');

if ~isdir(outpath)
  mkdir(outpath)
end

%% Data
% Datasets for benchmarking are downloaded from Chu Wei's website
% <http://www.gatsby.ucl.ac.uk/~chuwei/ordinalregression.html
% Benchmarking of Ordinal Regression>.

datasets = struct(...
  'url', { ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_diabetes.zip', ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_pyrimidines.zip', ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_triazines.zip', ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_breast.zip', ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_machine.zip', ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_autompg.zip', ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_bostonhousing.zip', ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_stocksdomain.zip', ...
    'http://www.gatsby.ucl.ac.uk/~chuwei/data/ordinal_abalone.zip' ...
  }, ...
  'name', { ...
    'Diabetes', ...
    'Pyrimidine', ...
    'Triazines', ...
    'Wisconsin', ...
    'Machine', ...
    'AutoMPG', ...
    'Boston', ...
    'Stocks', ...
    'Abalone' ...
  }, ...
  'filename', { ...
    'Diabetes/diabetes.data', ...
    'pyrimidines/pyrim', ...
    'triazines/triazines', ...
    'wisconsin/wpbc', ...
    'machinecpu/machine', ...
    'Auto-Mpg/auto.data', ...
    'bostonhousing/housing', ...
    'stocksdomain/stock', ...
    'abalone/abalone' ...
  }, ...
  'holdout', {13, 24, 86, 64, 59, 192, 206, 350, 3177}, ...
  'tbl', cell(1, 9) ...
);

datasets = datasets(1:2);

nDatasets = length(datasets);
downloadData(datapath, datasets);
[datasets, dataInfo] = preprocData(datapath, datasets);

disp(dataInfo);

%% Tested models
% Tested model settings:

% tested algorithm settings
models = struct(...
  'name', ...
    { ...
      %'PLSOR_SE_ZEROONE' ...
      %'PLSOR_SE_ABSERR' ...
      'PLSOR_SEARD' ...
    }, ...
  'settings', ...
    { ...
      %{'KernelFunction', 'squaredexponential', 'LossFunction', 'zeroone'} ...
      %{'KernelFunction', 'squaredexponential', 'LossFunction', 'abserr'} ...
      {'KernelFunction', 'ardsquaredexponential'} ...
    } ...
);
nModels = length(models);

modelsInfo = struct2table(models);

disp(modelsInfo);

%% Statistics
% Tests are performed for two discretizations of the output space:
%
% * into 5 bins
% * into 10 bins
%
% Each model is tested on each dataset in a holdout crossvalidation with 20
% random repetitions (or folds). The following performance measures are
% recoreded in each repetition:
%
% * the missclassification rate
% * the absolute error
% * the time taken by fitting the model
% * the optimized value of the negative log probability (NLP)
% * the best of all starting points
%
% The mean and standard deviations of all of these measures are calculated
% over all repetitions for each model and each dataset.

% a table with results will be generated for each bin size
bins = [5 10];
nBins = length(bins);

% the number of Monte-Carlo repetitions in crossvalidation
nReps = 20;

% statistics of crossvalidation trials
statFcns = struct( ...
  'name', {'MCR', 'AbsErr', 'FitTime', 'MinNLP', 'FitTrial'}, ...
  'nargout', {1, 1, 1, 1, 1}, ...
  'fcn', { ...
    @(~, ypred, yte, ~) misclassErr(ypred, yte, 'mcr'), ...
    @(~, ypred, yte, ~) misclassErr(ypred, yte, 'abs'), ...
    @(~, ~, ~, t) t, ...
    @(mdl, ~, ~, ~) mdl.MinimumNLP, ...
    @(mdl, ~, ~, ~) mdl.OptimTrial ...
  } ...
);
nStatFcns = length(statFcns);

%% Running benchmarks

% a multidimensional result cell array
% the last dimension is for statistics mean vs. std
results = cell(nDatasets, nBins, nModels, nStatFcns, 2);

% data for tables with datasets in rows and all formatted combinations of
% model settings vs. statistics in columns
resTbls = repmat({cell2table(repmat({''}, nDatasets, nModels * nStatFcns + 1))}, ...
  1, nBins);

% resulting tables variable names
varNames = cell(1, nModels * nStatFcns + 1);
varNames{1} = 'Dataset';

for binIdx = 1:length(bins)
  fprintf('Bins: %d\n', bins(binIdx));

  for datasetIdx = 1:length(datasets)
    dataset = datasets(datasetIdx);
    fprintf('\tDataset: %s\n', dataset.name);

    holdout = dataset.holdout;

    X = dataset.tbl{:, 1:end-1};
    Y = binning(dataset.tbl{:, end}, bins(binIdx));

    for modelIdx = 1:length(models)
      model = models(modelIdx);
      fprintf('\t\tModel: %s\n', model.name);

      testFun = @(Xtr, Ytr, Xte, Yte) ...
        testModel(Xtr, Ytr, Xte, Yte, model.settings, statFcns);

      if ~dry
        stats = crossval(testFun, X, Y, 'holdout', holdout, ...
          'mcreps', nReps);
      else
        stats = zeros(nReps, nStatFcns);
      end

      fmt = repmat({'%.2f'}, 1, length(mean(stats)));
      fmt = strjoin(fmt, ', ');
      fprintf(['\t\tMean stats: ' fmt ' \n'], mean(stats));

      % raw results
      results(datasetIdx, binIdx, modelIdx, :, 1) = num2cell(mean(stats));
      results(datasetIdx, binIdx, modelIdx, :, 2) = num2cell(std(stats));

      % formatted results
      if isempty(resTbls{binIdx}{datasetIdx, 1}{:})
        resTbls{binIdx}{datasetIdx, 1} = {dataset.name};
      end

      for statFcnIdx = 1:nStatFcns
        statFcn = statFcns(statFcnIdx);

        col = (modelIdx - 1) * nStatFcns + statFcnIdx + 1;

        if isempty(varNames{col})
          varNames{col} = sprintf('%s__%s', model.name, statFcn.name);
        end

        resTbls{binIdx}{datasetIdx, col} = {sprintf('%5.2f+/-%4.2f', ...
          mean(stats(:, statFcnIdx)), std(stats(:, statFcnIdx)))};
      end
    end
  end
end

for i = 1:length(resTbls)
  resTbls{i}.Properties.VariableNames = varNames;
end

% save the results into a timestamped file
ts = datestr(now, 'yyyy-mm-ddTHH:MM');
filename = fullfile(outpath, ['benchmark_results_' ts '.mat']);
save(filename, 'results', 'resTbls');

%% 5 bins results

disp(resTbls{1});

%% 10 bins results

disp(resTbls{2});

