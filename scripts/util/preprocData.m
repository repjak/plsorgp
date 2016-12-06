function [datasets, dataInfo] = preprocData(datapath, datasets, bins, nReps)
%PREPROCDATA Convert data to csv and read them into tables.

  dataInfo = cell(length(datasets), 4);

  nTest = nan(length(datasets), 1);
  nTrain = nan(length(datasets), 1);
  nAttrs = nan(length(datasets), 1);

  for i = 1:length(datasets)
    dataset = datasets(i);

    for bin = bins
      dirname = fullfile(dataset.dirname, [num2str(bin) 'bin']);

      if ~isdir(fullfile(datapath, dirname))
        dirname = fullfile(dataset.dirname, [num2str(bin) 'bins']);
      end

      for t = {'Tr', 'Te'}
        data = cell(1, nReps);

        for k = 1:nReps
          for s = {'', '.ord'}
            if strcmp(t, 'Tr')
              filename = [dataset.prefix s{:} '_train_' num2str(bin) '.' num2str(k)];
            else
              filename = [dataset.prefix s{:} '_test_' num2str(bin) '.' num2str(k)];
            end

            filename = fullfile(datapath, dirname, filename);

            if exist(filename, 'file')
              break;
            end
          end
          csvfilename = [filename '.csv'];

          if ~exist(csvfilename, 'file')
            copyfile(filename, csvfilename);
          end

          data{k} = readtable(csvfilename, 'Delimiter', ' ', ...
            'ReadVariableNames', false ...
          );

          nAttrs(i) = size(data{k}, 2) - 1;

          if strcmp(t, 'Te')
            assert(isnan(nTest(i)) || nTest(i) == size(data{k}, 1));
            nTest(i) = size(data{k}, 1);
          else
            nTrain(i) = size(data{k}, 1);
          end
        end

        dataset.(['data' t{:} num2str(bin)]) = data;
      end
    end

    datasets(i) = dataset;
    dataInfo(i, :) = {datasets(i).name, nAttrs(i), nTrain(i), nTest(i)};
  end

  dataInfo = cell2table(dataInfo, 'VariableNames', {'Name', ...
    'Attributes', 'TrainingInstances', 'TestInstances'});
end

