% initialize rng for reproducibility
seed = 2;
rng(seed);

mpath = mfilename('fullpath');
addpath(fullfile(fileparts(mpath), 'util/'));
datapath = fullfile(fileparts(mpath), 'data/');
outpath = fullfile(fileparts(mpath), 'results/');

if ~isdir(outpath)
  mkdir(outpath)
end

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
    'Pyrimidines', ...
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

% a table with results will be generated for each bin size
% bins = [5, 10];
bins = [5];
nBins = length(bins);

% a number of Monte-Carlo repetitions in crossvalidation
nReps = 20;

% tested algorithm settings
models = struct(...
  'name', ...
    { ...
      'PLSOR_SE' ...
      %'PLSOR_SEARD' ...
    }, ...
  'settings', ...
    { ...
      {'KernelFunction', 'squaredexponential'} ...
      %{'KernelFunction', 'ardsquaredexponential'} ...
    } ...
);
nModels = length(models);

modelsInfo = struct2table(models);
disp(modelsInfo);

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

      stats = crossval(testFun, X, Y, 'holdout', holdout, ...
        'mcreps', nReps);

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

        col = (modelIdx - 1) * nModels + statFcnIdx + 1;

        if isempty(varNames{col})
          varNames{col} = sprintf('%s__%s', model.name, statFcn.name);
        end

        resTbls{binIdx}{datasetIdx, col} = {sprintf('%5.2f+/-%4.2f', ...
          mean(stats(:, statFcnIdx)), std(stats(:, statFcnIdx)))};
      end
    end
  end
end

resTbls{1}.Properties.VariableNames = varNames;

ts = datestr(now, 'yyyy-mm-ddTHH:MM');
filename = fullfile(outpath, ['benchmark_results_' ts '.mat']);
save(filename, 'results', 'resTbls');

disp(resTbls{1});

% resTbls{2}.Properties.VariableNames = varNames;
% disp(resTbls{2});
