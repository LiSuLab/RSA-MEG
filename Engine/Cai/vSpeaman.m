% rho = vSpearman(a, b)
%
% Will correlate each column of a with each column of b using Spearman's
% rank correlation. (Method vectorised for speed.)
%
% Based on andy Thwaites' code and method; CW 5-2010

function rhos = vSpeaman(a, b)
	
 	[aCols aRows] = size(a);
	[bCols bRows] = size(b);
	
	if aRows ~= bRows, error('vSpeaman:nonMatchingRows', 'a and b must have the same number of rows each.'); end%if

	% Rank a and b
	
	[aSorted aIndices] =  sort(a);	clear aSorted;
	[junk aRanks] = sort(aIndices);	clear junk;
	
	[bSorted bIndices] =  sort(b);	clear bSorted;
	[junk bRanks] = sort(bIndices);	clear junk;
	
	% Now for the pairwise thing
	
	a2 = repmat(aRanks, 1, bCols);
	b2 = kron(bRanks, ones(1, aCols));

	dsqr = (a2 - b2) .^ 2;
	rhos = 1 - ((6*sum(dsqr))./ (sizeofp.*(sizeofp.^2-1)));	
	
end
