disp('executing plsorgp startup script');

for folder = {'cov', 'gp', 'loss', 'plsor', 'scripts', 'unittest', 'util'}
  addpath(fullfile(pwd, folder{:}));
end

