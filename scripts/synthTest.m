%% Experiments on the PLSOR method.

% Initialize the rng.
seed = 5;
rng(seed);

%% Generate test data.
% The PLSOR model is trained on a small sample of data from interval
% [0, 20]:
% - 2 random points from interval [2, 4] in category 1
% - 3 random points from interval [4, 6] in category 2
% - 10 points from [8, 14] in category 3
% - 5 points from [14, 18] in category 4
% - 1 point from [18, 20] in category 5
%
% The categories are defined so that y belongs to category i iff
% 0 < y - i < 1.
%
% The model's response is tested on linearly spaced points between
% 0 and 20.

intrvs = [2 4; 4 6; 8 14; 14 18; 18 20];
freqs = [2 3 10 5 1];
bins = [1 2 3 4 5];
es = [0 cumsum(freqs)];
Xtr = zeros(sum(freqs), 1);
ytr = zeros(sum(freqs), 1);

for i = 1:size(intrvs, 1)
  Xtr(es(i)+1:es(i+1)) = intrvs(i, 1) + range(intrvs(i, :)) * rand(1, freqs(i));
  ytr(es(i)+1:es(i+1)) = bins(i) + rand(1, freqs(i));
end

ytrBins = binning(ytr, length(bins));

Xte = linspace(0, 20, 200)';

%% Train the model and show some of its properties.
tic;
ordgp = OrdRegressionGP(Xtr, ytrBins);
t1 = toc;

disp('Elapsed time:');
disp(t1);
disp('Minimum negative log probability:');
disp(ordgp.MinimumNLP);
disp('PLSOR hyperparameters:');
disp(ordgp.PlsorParameters);
disp('Covariance hyperparameters:');
disp(ordgp.KernelParameters);
disp('The best starting point:');
disp(ordgp.OptimTrial);

%% Make predictions.
% The first output argument contains the most probable ordinal class for
% each test point. The second argument are the predicted probabilities.
% The third and fourth are the mean and variance of the latent variable
% predicted by the underlying Gaussian process.
[y, probs, mu, s2] = ordgp.predict(Xte);

%% *Figure 1:* Gaussian process and PLSOR predictions.
% Show the predicted mean and variance of the latent variables
% and classification on some test points.

h = figure();
hold on;
plot(Xtr, ytr, '+');
xlim([0 20]);
plot(Xte, mu, 'r-');
plot(Xte, mu - sqrt(s2), 'b--');
plot(Xte, mu + sqrt(s2), 'b--');
plot(Xte(10:20:end), y(10:20:end), 'ro');
legend('Training data', 'Mean', '+ std', '- std', 'Predicted class', ...
  'Location', 'southeast');
