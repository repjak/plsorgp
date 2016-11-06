function [X, y] = extractTableData(tbl, spec)
%EXTRACTTABLEDATA Extract predictor and response data from the table.
%   EXTRACTTABLEDATA(tbl, ResponseVarName) divides the table columns into
%   predictor and response data according to ResponseVarName
%   EXTRACTTABLEDATA(tbl, formula) attempts to match response and predictor
%   variables to the formula 'respVarName~predVar1[-predVar2[...]...]'

  if ischar(spec)
    if isempty(spec)
      error('The predictor and response variables must be specified.');
    end

    [varNames1, matches] = strsplit(spec, '~');
    if ~isempty(matches)
      if length(matches) > 1
        error('The formula must contain at most one delimiter ''~''');
      end

      varNames2 = strsplit(varNames1{2}, '-');
      varNames = [varNames1(1) varNames2];

      if ~all(ismember(varNames, tbl.Properties.VariableNames))
        error('Some variable empty or not recognized.');
      end

      X = tbl{:, varNames(2:end)};  % the predictor columns
      y = tbl{:, varNames(1)};      % the response column
    else
      [member, idx] = ismember(spec, tbl.Properties.VariableNames);
      if ~member
        error(['Unrecognized variable name ''' spec '''.']);
      end

      X = tbl{:,[1:idx-1 idx+1:size(tbl,2)]};
      y = tbl{:,idx};
    end
  else
    y = spec;
    if isempty(y) || size(y, 2) > 1
      error('The response is expected to be a non-empty column vector.');
    end

    X = tbl{:, :};

    if size(X, 1) ~= length(y)
      error('Dimensions don''t match.');
    end
  end
end

