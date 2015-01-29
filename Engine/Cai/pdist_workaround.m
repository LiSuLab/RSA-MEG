%  pdist_workaround is a function which tries to emulate some of the
%  functionality of MATLAB's pdist, but without it BREAKING ALL THE FREAKIN'
%  TIME WHILE USING SPEARMAN!
%  
%  Cai Wingfield 12-2009

function Y = pdist_workaround(X, distanceMeasure)

if ~strcmpi(distanceMeasure, 'Spearman')
	Y = pdist(X, distanceMeasure);
else
	%% If Spearman's being used

	% catch errors
	if numel(size(X)) > 2, error('(!) Input matrix must be 2-dimensional!'); end

	[nRows, nColumns] = size(X);

	yIndex = 1;
	
	for row1i = 1:nRows
		row1 = X(row1i, :);
		for row2i = row1i+1:nRows
			row2 = X(row2i, :);
			%Y(yIndex) = spearman(row1', row2'); % Doesn't help!
			warning off all
			Y(yIndex) = corr(row1', row2', 'type', 'spearman');
			warning on all
			yIndex = yIndex + 1;
		end%for
	end%for

end%if
