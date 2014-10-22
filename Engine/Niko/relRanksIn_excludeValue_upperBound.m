function ps=relRanksIn_excludeValue_upperBound(set,values)

% returns the relative ranks of values within set.
% the relative rank is the proportion of set smaller than value.
%
% this is the conservative version for p value estimation.
% it assumes that the value is best represented by it's upper neighbor in
% the set (yielding a larger proportion of probability mass below it, i.e.
% an upper bound).

ps=nan(size(values));
for i=1:numel(values)
    if any(set(:)==values(i))
        p1=sum(set(:)<values(i))/numel(set);
        p2=1-sum(set(:)>values(i))/numel(set);
        ps(i)=(p1+p2)/2;
    else
        ps(i)=min(1,(sum(set(:)<values(i))+.5)/numel(set));
    end
end
