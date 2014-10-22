function p=relRankIn_excludeValue_lowerBound(set,value)

% returns the relative rank of value within set.
% the relative rank is the proportion of set smaller than value.
%
% this is the conservative version for p value estimation.
% it assumes that the value is best represented by it's lower neighbor in
% the set (yielding a smaller proportion of probability mass below it, i.e.
% an lower bound).

set=-set;
value=-value;

if any(set(:)==value)
    p1=1-sum(set(:)<value)/numel(set);
    p2=sum(set(:)>value)/numel(set);
    p=(p1+p2)/2;
else
    p=1-min(1,(sum(set(:)<value)+.5)/numel(set));
end

