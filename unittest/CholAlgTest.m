classdef CholAlgTest < matlab.unittest.TestCase
  %CHOLALGTEST Summary of this class goes here
  %   Detailed explanation goes here

  methods (Test)
    function cholInvTest(testCase)
      for type = {'lehmer', 'minij'};
        for n = [0 1 2 5 10 20 100 500 1000 2000]
          A = gallery('lehmer', n);
          R = chol(A);
          Ainv = cholinv(R);
          testCase.verifyEqual(A*Ainv, eye(n), 'AbsTol', 1e-8);

          X = sqrt(n)*randn(n, n);
          Y = cholsolve(R, A*X);
          Z = cholsolve(R, X*A, 'right');
          testCase.verifyEqual(Y, X, 'AbsTol', 1e-8);
          testCase.verifyEqual(Z, X, 'AbsTol', 1e-8);
        end
      end
    end
  end
  
end

