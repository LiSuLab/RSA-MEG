function p=relRankIn_includeValue_lowerBound(set,value)

% returns the relative rank of value within set.
% the relative rank is the proportion of set smaller than value.
set=[set(:);value];
p=sum(set(:)<value)/numel(set);
