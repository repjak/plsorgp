function [datasets, dataInfo] = preprocData(datapath, datasets)
%PREPROCDATA Convert data to csv and read them into tables.

  dataInfo = cell(length(datasets), 4);

  for i = 1:length(datasets)
    filename = fullfile(datapath, datasets(i).filename);

    % the whole file is read into memory
    text = fileread(filename);

    % remove leading white spaces
    text = regexprep(text, '^\s*', '');
    text = regexprep(text, '\n\s*', '\n');

    % unify delimiters
    if isempty(strfind(text, ','))
      repl = ',';
    else
      repl = '';
    end

    % reduce redundant white spaces assuming no empty fields
    text = regexprep(text, '([\t ])+', repl);

    % create a temporary csv file for readtable
    fid = fopen([filename '.csv'], 'w');
    fprintf(fid, '%s', text);
    fclose(fid);

    datasets(i).tbl = readtable([filename '.csv'], 'Delimiter', ',', ...
      'ReadVariableNames', false ...
    );

    % delete the temporary csv file
    delete([filename '.csv']);

    dataset = datasets(i);
    te = dataset.holdout;
    tr = size(dataset.tbl, 1) - dataset.holdout;
    m = size(dataset.tbl, 2);

    dataInfo(i, :) = {datasets(i).name, m - 1, tr, te};
  end

  dataInfo = cell2table(dataInfo, 'VariableNames', {'Name', ...
    'Attributes', 'TrainingInstances', 'TestInstances'});
end

