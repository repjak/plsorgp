function X = cholsolve(R, B, varargin)
%CHOLSOLVE Solve a system of linear equations AX = B for X with A given by
% its Cholesky decomposition, i.e. A = R'R.
%   X = cholsolve(R, B) solve AX = B for X, where A = R'R
%   X = cholsolve(R, 'right') solve XA = B for X, where A = RR'

  if nargin >= 3 && strcmp(varargin{1}, 'right')
    X = (B / R) / R';
  elseif nargin < 3
    X = R \ (R' \ B);
  else
    error('Unknown input argument ''%s''.', varargin{1});
  end
end

