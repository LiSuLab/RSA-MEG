% Writes a cell line-by-line to a file
% CW 1-2010

function writeCellToFile(cellIn, fileAddress)

fileID = fopen(fileAddress, 'w');

if numel(cellIn) ~= max(size(cellIn)), error('Can only accept 1d cell arrays :('); end%if

for lineNumber = 1:numel(cellIn)

	fprintf(fileID, [cellIn{lineNumber} '\n']);

end%for

fclose(fileID);