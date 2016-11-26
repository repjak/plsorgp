% Initialize the random number generator for reproducibility.
seed = 2;
rng(seed);

%% Training data
% The PLSOR model is trained on a small sample of data from interval
% [0, 20]:
%
% * 2 random points from interval [2, 4] in category 1
% * 3 points from [4, 6] in category 2
% * 10 points from [8, 14] in category 3
% * 5 points from [14, 18] in category 4
% * 1 point from [18, 20] in category 5
%
% The target values for each category are sampled from a uniform
% distribution with mean at the category.

intrvs = [2 4; 4 6; 8 14; 14 18; 18 20];
freqs = [2 3 10 5 1];
bins = [1 2 3 4 5];
es = [0 cumsum(freqs)];
Xtr = zeros(sum(freqs), 1);
ytr = zeros(sum(freqs), 1);

for i = 1:size(intrvs, 1)
  Xtr(es(i)+1:es(i+1)) = intrvs(i, 1) + range(intrvs(i, :)) * ...
    rand(1, freqs(i));
  ytr(es(i)+1:es(i+1)) = bins(i) + rand(1, freqs(i)) - 0.5;
end

%% Fitting the model
% The model is fitted on ordinal values. In this case the output space
% is discretized by rounded off the targets.
%
% Fitting is done by creating an instance of |OrdRegressionGP| class.
Ytr = round(ytr);

tic;
ordgp = OrdRegressionGP(Xtr, Ytr);
t1 = toc;

%% Model's properties
% The created object contains optimized hyperparameters and other state
% variables required for predictions.
%
% Here, the fitting time, the successful fitting trial, the optimal
% pseudo-likelihood (negative log predicted probability) reached,
% and the fitted hyperparameter values are shown.
disp('Fitting time:');
disp(t1);
disp('The successful trial:');
disp(ordgp.OptimTrial);
disp('Minimal negative log probability:');
disp(ordgp.MinimumNLP);
disp('PLSOR hyperparameters:');
disp(ordgp.PlsorParameters);
disp('Covariance hyperparameters:');
disp(ordgp.KernelParameters);
disp('The noise hyperparameter:');
disp(ordgp.Sigma2);

%% Making predictions
% The model's response is tested on linearly spaced points between
% 0 and 20.
%
% Prediction is implemented by the |predict| method of an |OrdRegressionGP|
% object.
%
% The first output argument contains the most probable ordinal class for
% each test point. The second output argument contains the predicted
% probabilities. The mean and variance by the underlying Gaussian process
% are returned in the third and fourth arguments, respectively.
Xte = linspace(0, 20, 200)';
[y, probs, mu, s2] = ordgp.predict(Xte);

%% *Figure 1:* Gaussian process and PLSOR predictions
% Plot the predicted mean and variance of the latent variable
% and predicted classes of a uniformely spaced selection of test points.
plot(Xtr, ytr, '+');
hold on;
grid on;
xlim([0 20]);
ylim([0 length(bins) + 1]);
plot(Xte, mu, 'r-');
plot(Xte, mu - sqrt(s2), 'b--');
plot(Xte, mu + sqrt(s2), 'b--');
plot(Xte(10:10:end), y(10:10:end), 'ro');
legend('Training data', 'Predicted mean', 'mu + std', 'mu - std', 'Test predictions', ...
  'Location', 'southeast');
snapnow;
close all;