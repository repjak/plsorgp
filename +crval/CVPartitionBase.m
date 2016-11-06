classdef CVPartitionBase < handle
  %CVPARTITIONBASE A base class for crossvalidation partitioning.
  %   Partition data into test and training sets. The interface corresponds
  %   to that of cvpartition from the Statistics and Machine Learning
  %   Toolbox.
  %
  %   See also CVPARTITIONLEAVE1OUT.

  properties (Access = protected)
    testData
    trainData
  end

  properties
    NumObservations  % Number of observations
    NumTestSets      % Number of test sets
    TestSize         % Size of each test set
    TrainSize        % Size of each training set
  end

  methods (Abstract)
    idx = test(obj, varargin)      % Test indices
    idx = training(obj, varargin)  % Training indices
    repartition(obj)               % Repartition data for cross-validation
  end

end

