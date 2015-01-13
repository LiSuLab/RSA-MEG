% removeDoubleQuotes will remove a pair of double quotes if they appear at the
% beginning and end of a string
%
% Cai Wingfield 1-2010

function stringOut = removeDoubleQuotes(stringIn)

stringOut = stringIn;

if ~isempty(stringOut)
	if strcmpi(stringOut(1), '"') && strcmpi(stringOut(end), '"')
		stringOut = stringOut(2:end-1);
	end
end
