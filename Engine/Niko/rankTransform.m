function rankArray=rankTransform(array,scale01)

% transforms the array 'array' by replacing each element by its rank in the
% distribution of all its elements



if ~exist('scale01','var'), scale01=false; end;

nonNan_LOG=~isnan(array);
set=array(nonNan_LOG); % column vector

[sortedSet, sortedIs]=sort(set);

rankArray=nan(size(array));
nonNan_IND=find(nonNan_LOG);
rankArray(nonNan_IND(sortedIs))=1:numel(set);

if scale01
    rankArray=rankArray/numel(set);
end

