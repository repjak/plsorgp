function Ainv = cholinv(R)
%CHOLINV Invert a positive-definite matrix given by its Cholesky upper
%triangular factor.
%   Ainv = cholinv(R) compute the inverse of matrix A with Cholesky
%   decomposition A=R'R by solving the following systems of linear
%   equations: RX = B, R'B = I.
%
%   See also CHOLSOLVE.

  Ainv = R \ (R' \ eye(size(R, 1)));
end

