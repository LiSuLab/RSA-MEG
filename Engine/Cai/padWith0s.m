% padWith0s takes a string stringIn (a string representation of a number) and
% pads the beginning of it with leading and trailing 0s so that the length of
% the string before the decimal is extended to maxLengthBefore and the length of
% the string after is extended to maxLengthAfter.
%
% USAGE: stringOut = padWith0s(stringIn, maxLengthBefore[, maxLengthAfter, radix])
%
% EXAMPLES:
%
% 	str1 = num2str(123.456);
%	str2 = padWithLeading0s(str1, 6, 0);    % str2 = '000123.456'
%	str3 = padWithLeading0s(str1, 2, 6);    % str3 = '123.456000'
%	str4 = padWithLeading0s(str1, 10, 10);  % str4 = '0000000123.4560000000'
%
% Cai Wingfield 2-2010

function stringOut = padWith0s(stringIn, maxLengthBefore, maxLengthAfter, radix)

if ~exist('radix', 'var'), radix = '.';, end%if

radixIndex = strfind(stringIn, radix);

if numel(radixIndex) == 1

	preRadix = stringIn(1:radixIndex-1);
	postRadix = stringIn(radixIndex+1:end);

	preLength = length(preRadix);
	postLength = length(postRadix);

	nPreMissingChars = maxLengthBefore - preLength;
	nPostMissingChars = maxLengthAfter - postLength;

	preMissingChars = '';
	for i = 1:nPreMissingChars
		preMissingChars = [preMissingChars '0'];
	end%for:i

	postMissingChars = '';
	for i = 1:nPostMissingChars
		postMissingChars = [postMissingChars '0'];
	end%for:i

	stringOut = [preMissingChars preRadix radix postRadix postMissingChars];
	
elseif numel(radixIndex) == 0

	originalLength = length(stringIn);
	nMissingChars = maxLengthBefore - originalLength;
	missingChars = '';
	for i = 1:nMissingChars
		missingChars = [missingChars '0'];
	end%for:i
	
	stringOut = [missingChars stringIn];

end%if
