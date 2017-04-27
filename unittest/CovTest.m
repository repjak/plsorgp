classdef CovTest < matlab.unittest.TestCase

  methods (Test)
    function sqexpTest(testCase)
      Xm = [1 1; 0 1; -3 5];
      Xn = [-3 -4];
      m = size(Xm, 1);
      n = size(Xn, 1);

      sigma = 0.5;
      l = 2;

      testCase.verifyError(@() sqexp(Xm, Xn), '');
      testCase.verifyError(@() sqexp(Xm, [0; 1]), '');
      testCase.verifyError(@() sqexp(Xm, Xn, sigma), '');

      [actK, actdK] = sqexp(Xm, Xn, [log(sigma); log(l)], [1; 2]);
      [expK, expdK] = CovTest.sqexpNaive(Xm, Xn, [log(sigma); log(l)], [1; 2]);

      testCase.verifyEqual(actK, expK, 'AbsTol', 1e-4);
      testCase.verifyEqual(actdK, expdK, 'AbsTol', 1e-4);

      [actK1, ~] = sqexpard(Xm, Xn, [log(sigma); log(l); log(l)], [1; 2]);

      testCase.verifyEqual(actK, actK1, 'AbsTol', 1e-2);

      [actK, actdK] = sqexp(Xm, 'diag', [log(sigma); log(l)], [1; 2]);
      expK = sigma^2 * exp(-zeros(m, n)/2);
      expdK = zeros(m, n, 2);
      expdK(:, :, 1) = 2 * sigma^2 * exp(-zeros(m, n)/2);

      testCase.verifyEqual(actK, expK, 'AbsTol', 1e-4);
      testCase.verifyEqual(actdK, expdK, 'AbsTol', 1e-4);

      [actK, actdK] = sqexp(Xm, Xm, [log(sigma); log(l)], [1; 2]);
      testCase.verifyTrue(issymmetric(actK));
      testCase.verifyEqual(diag(actK), sigma^2 * ones(m, 1), 'AbsTol', 1e-4);
      testCase.verifyEqual(diag(actdK(:, :, 1)), 2 * sigma^2 * exp(-zeros(m, 1)/2), 'AbsTol', 1e-4);
      testCase.verifyEqual(diag(actdK(:, :, 2)), -zeros(m, 1), 'AbsTol', 1e-4);
    end

    function sqexpardTest(testCase)
      Xm = [1 1; 0 1; -3 5];
      Xn = [-3 -4];
      sigma = 0.5;
      l = [0.2; 2];

      [actK, actdK] = sqexpard(Xm, Xn, [log(sigma); log(l)], [1; 2]);
      [expK, expdK] = CovTest.sqexpNaive(Xm, Xn, [log(sigma); log(l)], [1; 2], 'ard');

      testCase.verifyError(@() sqexpard(Xm, Xn), '');
      testCase.verifyError(@() sqexpard(Xm, Xn, log(sigma)), '');

      testCase.verifyEqual(actK, expK, 'AbsTol', 1e-4);
      testCase.verifyEqual(actdK, expdK, 'AbsTol', 1e-4);
    end
  end

  methods (Static)
    function [Kmn, dKmn] = sqexpNaive(Xm, Xn, theta, j, varargin)
      m = size(Xm, 1);
      n = size(Xn, 1);
      sigma2 = exp(2*theta(1));
      l = exp(theta(2:end));

      Kmn = zeros(m, n);
      dKmn = zeros(m, n, length(j));

      sqdistk = @(xp, xq) (xp - xq).^2;

      for p = 1:m
        for q = 1:n
          Kmn(p, q) = sum(bsxfun(sqdistk, Xm(p, :)./l', Xn(q, :)./l'));

          for k = 1:length(j)
            if k == 1
              dKmn(p, q, k) = 2 * sigma2 * exp(-Kmn(p, q) / 2);
            else
              if nargin > 4 && strcmpi(varargin{1}, 'ard')
                dKmn(p, q, k) = sigma2 * exp(-Kmn(p, q) / 2) * sqdistk(Xm(p, k-1), Xn(q, k-1));
              else
                dKmn(p, q, k) = sigma2 * exp(-Kmn(p, q) / 2) * Kmn(p, q);
              end
            end
          end

          Kmn(p, q) = sigma2 * exp(-Kmn(p, q)/2);
        end
      end
    end
  end

end

