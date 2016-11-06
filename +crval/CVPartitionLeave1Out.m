classdef CVPartitionLeave1Out < crval.CVPartitionBase
  %CVPARTITIONLEAVE1OUT Leave-One-Out cross-validation partitioning.

  methods
    function obj = CVPartitionLeave1Out(n)
      if n < 2
        error('The number of observations must be greater than one.');
      end

      obj.NumTestSets = n;
      obj.TrainSize = (n-1) * ones(1, obj.NumTestSets);
      obj.TestSize = ones(1, obj.NumTestSets);
      obj.NumObservations = n;

      obj.testData = logical(eye(n));
      obj.trainData = ~obj.testData;
    end

    function idx = test(obj, varargin)
      if isempty(varargin)
        idx = obj.testData;
      else
        idx = obj.testData(:, varargin{1});
      end
    end

    function idx = training(obj, varargin)
      if isempty(varargin)
        idx = obj.trainData;
      else
        idx = obj.trainData(:, varargin{1});
      end
    end

    function repartition(~)
    end
  end

end

