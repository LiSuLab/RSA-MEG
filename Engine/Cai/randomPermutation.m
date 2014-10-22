function indicesOut = randomPermutation(n)

% Cai Wingfield 2-2010

indicesToChoseFrom = (1:n);

for i = 1:n

	position = ceil(rand*numel(indicesToChoseFrom));
	thisIndexChoice = indicesToChoseFrom(position);
	indicesOut(i) = thisIndexChoice;
	indicesToChoseFrom = removeElement(indicesToChoseFrom, position);

end%for(i)


%% === Subfunctions =============================

function vectorOut = removeElement(vectorIn, i)

if i == 1;
	vectorOut = vectorIn(2:end);
elseif i > 1 && i < numel(vectorIn)
	vectorOut = vectorIn([1:i-1, i+1:end]);
elseif i == numel(vectorIn)
	vectorOut = vectorIn(1:end-1);
else
	error('Can''t remove an element which isn''t in the input!');
end%if
