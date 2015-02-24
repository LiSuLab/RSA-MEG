function rankMat=rankTransform_equalsStayEqual(mat,scale01)

% transforms the matrix mat by replacing each element by its rank in the
% distribution of all its elements.
% NaNs are ignored in this process.

if ~exist('scale01','var'), scale01=1; end;

% A logical mask of the matrix with 1s where NaNs aren't
nonNan_LOG = ~isnan(mat);

% List of all non-NaN values in the matrix
set = mat(nonNan_LOG);

% Use the sort trick to rank the values from highest to lowest
[ignore, sortedIs] = sort(set);

% Preallocate a NaN matrix the size of the original
% We'll fill in the non-NaN entries, leaving the originally ignored NaNs
% showing through
rankMat = nan(size(mat));

% List of positions of non-nan entries.
nonNan_IND = find(nonNan_LOG);

% Using a trick, we place into the non-NaN positions in the new matrix, the
% rank value we just calculated into the position its corresponding
% original value was.
rankMat(nonNan_IND(sortedIs)) = 1:numel(set);

if scale01 == 1
    % scale into [0,1]
    rankMat = (rankMat-1)/(numel(set)-1);
elseif scale01 == 2
    % scale into ]0,1[
    % (best representation of a uniform distribution between 0 and 1)
    rankMat = (rankMat-.5)/numel(set);
end%if

% Want equal values to have tied ranks
uniqueValues=unique(mat(nonNan_LOG));

% For each possible value that was attained in the original matrix
for uniqueValueI = 1:numel(uniqueValues)
    % The positions in the matrix where this value is
    cValueEntries_LOG = (mat == uniqueValues(uniqueValueI));

    rankMat(cValueEntries_LOG) = mean(rankMat(cValueEntries_LOG));
end%for

end%function
