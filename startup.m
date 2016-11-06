disp('executing plsorgp startup script');

for folder = {'cov', 'gp', 'loss', 'predProb', 'unittest', 'util'}
  addpath(fullfile(pwd, folder{:}));
end

