classdef BinningTest < matlab.unittest.TestCase
  %BINNINGTEST Summary of this class goes here
  %   Detailed explanation goes here

  methods (Test)
    function binningTest(testCase)
      for k = 1:5
        testCase.verifyEmpty(binning([], k));
      end

      n = 1000;
      l = -5;
      ran = 10;
      data = l + (ran+l) * rand(1, n);

      testCase.verifyError(@() binning(data, 0), '');
      testCase.verifyEqual(binning(data, 1), ones(1, n));
    end
  end

end

