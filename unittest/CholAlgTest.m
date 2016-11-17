classdef CholAlgTest < matlab.unittest.TestCase

  methods (Test)
    function cholGalleryTest(testCase)
      for type = {'lehmer', 'minij'}
        for n = [0 1 2 5 10 20 1e2 1e3]
          A = gallery(type{:}, n);
          R = chol(A);
          Ainv = cholinv(R);
          testCase.verifyEqual(A*Ainv, eye(n), 'AbsTol', 1e-4);

          L = chol(A, 'lower');
          Ainv = cholinv(L, 'lower');
          testCase.verifyEqual(A*Ainv, eye(n), 'AbsTol', 1e-4);

          X = n*randn(n, n);

          % solve AX = B via R, where R'*R = A
          Y = cholsolve(R, A*X);

          % solve XA = B via L, where L*L' = A
          Z = cholsolve(R', (X*A)', 'lower')';

          testCase.verifyEqual(Y, X, 'AbsTol', 1e-4);
          testCase.verifyEqual(Z, X, 'AbsTol', 1e-4);
        end
      end
    end

    function cholRandTest(testCase)
      n = 1e3;
      trials = ceil(n^(1/4));
      for trial = 1:trials
        A = rand(n, n);
        A = 0.5 * (A + A');
        A = A + n * eye(n);

        R = chol(A);
        Ainv1 = cholinv(R);
        Ainv2 = inv(A);

        I1 = Ainv1 * A;
        I2 = Ainv2 * A;

        err1 = sum(sum(abs(I1 - eye(n))));
        err2 = sum(sum(abs(I2 - eye(n))));

        testCase.verifyEqual(I1, eye(n), 'AbsTol', 1e-4);
        testCase.verifyLessThan(err1, err2);

        X = n*rand(n, n);
        Y = cholsolve(R, A*X);
        Z = cholsolve(R', (X*A)', 'lower')';
        testCase.verifyEqual(Y, X, 'AbsTol', 1e-4);
        testCase.verifyEqual(Z, X, 'AbsTol', 1e-4);
      end
    end
  end

end

