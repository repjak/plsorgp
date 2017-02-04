function downloadData(datapath, datasets)
%DOWNLOADDATA Download datasets from remote repositories.
%   DOWNLOADDATA(datapath, datasets) download datasets specified in
%   a struct array with fields 'url' and 'filename' into location
%   'datapath'.

  if ~isdir(datapath)
    mkdir(datapath);
  end

  for i = 1:length(datasets)
    dataset = datasets(i);
    url = dataset.url;
    urlparts = strsplit(url, '/');
    assert(length(urlparts) >= 2);

    filename = fullfile(datapath, urlparts{end});

    if ~exist(filename, 'file')
      fprintf('Get %s\n', url);
      websave(filename, url);
    end

    if ~isdir(fullfile(datapath, dataset.dirname))
      fprintf('Unzip %s\n', filename');
      unzip(filename, datapath);
    end
  end
end


