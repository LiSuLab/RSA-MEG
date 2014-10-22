% removeSpaces will remove all spaces from a string
%
% Cai Wingfield 1-2010

function stringOut = removeSpaces(stringIn)

if ~isempty(stringIn)
	stringOut = strrep(stringIn, ' ', '');
end
