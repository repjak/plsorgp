classdef BinningTest < matlab.unittest.TestCase
  %BINNINGTEST Summary of this class goes here
  %   Detailed explanation goes here

  methods (Test)
    function binningTest(testCase)
      for k = 1:5
        testCase.verifyEmpty(binning([], k, 'uniform'));
      end

      n = 1000;
      l = -5;
      ran = 10;
      data = l + (ran+l) * rand(1, n);

      testCase.verifyError(@() binning(data, 0, 'uniform'), '');
      testCase.verifyEqual(binning(data, 1, 'uniform'), ones(n, 1));
    end
  end

end

