function Ainv = cholinv(R, varargin)
%CHOLINV Invert a positive-definite matrix given by its Cholesky upper
%triangular factor.
%   Ainv = cholinv(R) compute the inverse of matrix A with Cholesky
%   decomposition A=R'R by solving the following systems of linear
%   equations: RX = B, R'B = I.
%   Ainv = cholinv(L, 'lower') comput the inverse using the lower factor L,
%   i.e. A = LL'.
%
%   Note: a MEX wrapper of a LAPACK call is also provided and will
%   overshadow this function when compiled.
%
%   See also CHOLSOLVE.

  n = size(R, 1);

  if nargin >= 2 && strcmp(varargin{1}, 'lower')
    Ainv = (eye(n) / R') / R;
  elseif nargin < 2 || strcmp(varargin{1}, 'upper')
    Ainv = R \ (R' \ eye(n));
  else
    error('Unknown input argument ''%s''.', varargin{1});
  end
end

