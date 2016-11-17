function X = cholsolve(R, B, varargin)
%CHOLSOLVE Solve a system of linear equations AX = B for X with A given by
%its Cholesky decomposition, i.e. A = R'R.
%   X = cholsolve(R, B) solve AX = B for X, where A = R'R.
%   X = cholsolve(R, B, 'lower') solve AX = B for X, where A = LL'.
%
%   Note: a MEX wrapper of a LAPACK call is also provided and will
%   overshadow this function when compiled.
%
%   See also CHOLINV.

  if nargin >= 3 && strcmp(varargin{1}, 'lower')
    X = R' \ (R \ B);
  elseif nargin < 3 || strcmp(varargin{1}, 'upper')
    X = R \ (R' \ B);
  else
    error('Unknown input argument ''%s''.', varargin{1});
  end
end

