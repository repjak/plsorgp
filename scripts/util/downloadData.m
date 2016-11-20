function downloadData(datapath, datasets)
%DOWNLOADDATA Download datasets from remote repositories.
%   DOWNLOADDATA(datapath, datasets) download datasets specified in
%   a struct array with fields 'url' and 'filename' into location
%   'datapath'.

  if ~isdir(datapath)
    mkdir(datapath);
  end

  for dataset = datasets
    url = dataset.url;
    urlparts = strsplit(url, '/');
    assert(length(urlparts) >= 2);

    filename = fullfile(datapath, urlparts{end});

    if ~exist(filename, 'file')
      fprintf('Get %s\n', url);
      websave(filename, url);
    end

    if ~exist(fullfile(datapath, dataset.filename), 'file')
      fprintf('Unzip %s\n', filename');
      unzip(filename, datapath);
    end
  end
end


