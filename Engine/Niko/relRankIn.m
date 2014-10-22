function p=relRankIn(set,value)

% returns the relative rank of value within set. the relative rank is the
% proportion of set smaller than value. here this is estimated for the
% population as the average between the proportion smaller and the
% proportion larger than the value in the sample (thus accounting for the
% fact that, for small samples 'set', 'value' itself represents a
% non-negligible probability mass).

set=set(:);
set=set(~isnan(set));

if ~isnan(value)
    p1=sum(set(:)<value)/numel(set);
    p2=1-sum(set(:)>value)/numel(set);
    p=(p1+p2)/2;
else
    p=nan;
end