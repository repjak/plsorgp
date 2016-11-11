classdef LossTest < matlab.unittest.TestCase

  methods (Test)
    function zerooneTest(testCase)
      P = [0.5 0.5; 0.4 0.6; 0.9 0.1; 1 0];
      [actMinLoss, actIdx, actPredProbs, actLosses] = zeroone(P);

      expMinLoss = [0.5 0.4 0.1 0]';
      expIdx = [1 2 1 1]';
      expPredProbs = [0.5 0.6 0.9 1]';
      expLosses = 1-P;

      testCase.verifyEqual(actMinLoss, expMinLoss, 'AbsTol', 1e-2);
      testCase.verifyEqual(actIdx, expIdx);
      testCase.verifyEqual(actPredProbs, expPredProbs, 'AbsTol', 1e-2);
      testCase.verifyEqual(actLosses, expLosses, 'AbsTol', 1e-2);

      zeroone([]);
      zeroone(1);
      testCase.verifyError(@() zeroone([1 1]), 'MATLAB:assertion:failed');
    end

    function hingeTest(testCase)
      [actMinLoss, actIdx, actPredProbs, actLosses] = hinge([0.4 0.3 0.3]);

      expMinLoss = 0.2333;
      expIdx = 2;
      expPredProbs = 0.3;
      expLosses = [0.3 0.2333 0.3667];

      testCase.verifyEqual(actMinLoss, expMinLoss, 'AbsTol', 1e-2);
      testCase.verifyEqual(actIdx, expIdx);
      testCase.verifyEqual(actPredProbs, expPredProbs, 'AbsTol', 1e-2);
      testCase.verifyEqual(actLosses, expLosses, 'AbsTol', 1e-2);

      hinge([]);
      hinge(1);
      testCase.verifyError(@() hinge([1 1]), 'MATLAB:assertion:failed');
    end

    function lossTest(testCase)
      P = [0.3 0.5 0.2; 0.9 0.05 0.05; 0.1 0.2 0.7];
      L = [0 1 2; 1 0 1; 2 1 0];
      [actMinLoss, actIdx, actPredProbs, actLosses] = loss(P, L);

      expMinLoss = [0.5 0.15 0.4]';
      expIdx = [2 1 3]';
      expPredProbs = [0.5 0.9 0.7]';
      expLosses = [0.9 0.5 1.1; 0.15 0.95 1.85; 1.6 0.8 0.4];

      testCase.verifyEqual(actMinLoss, expMinLoss, 'AbsTol', 1e-2);
      testCase.verifyEqual(actIdx, expIdx);
      testCase.verifyEqual(actPredProbs, expPredProbs, 'AbsTol', 1e-2);
      testCase.verifyEqual(actLosses, expLosses, 'AbsTol', 1e-2);

      loss([], []);
      testCase.verifyError(@() loss([], L), '');
      testCase.verifyError(@() loss(1, L), '');
      testCase.verifyError(@() loss([1 1]), 'MATLAB:assertion:failed');
      testCase.verifyError(@() loss([1; 1], L), '');
    end

  end

end

