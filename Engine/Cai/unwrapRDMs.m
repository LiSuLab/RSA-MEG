function [RDMs,nRDMs] = unwrapRDMs( RDMs_struct )
% unwraps dissimiliarity matrices in a structured array with meta data by
% extracting the dissimilarity matrices (in square or upper triangle form)
% and lining them up along the third dimension. (if they are already in
% that format they are handed back unchanged.)
%
% CW 5-2010: Edit to allow structs to contain RDMs of mixed types.  If all
%            the RDMs in a struct are of the same type (utv or square),
%            they're unwrapped to be a stack of the same form; else if
%            they're of mixed type, they're all made square. Also, if RDMs
%            in a struct are of different sizes, they'll all be returned as
%            squares with nans outside. (Is this best?)

if isstruct(RDMs_struct)
    % in struct form
    nRDMs=size(RDMs_struct,2);
	
	mixedTypes = false;
	
	for RDMi = 1:nRDMs
		
		thisRDM = RDMs_struct(RDMi).RDM;
		
		% What type of RDM is this?
		if max(size(thisRDM)) == numel(thisRDM)
			% Then it's in ltv form
			thisType = 'ltv';
			thisSize = (1+sqrt(1-4*numel(thisRDM)))/2;
		else
			thisType = 'sq';
			thisSize = size(thisRDM,1);
		end%if:utv
		
		if RDMi == 1
			% What's the first type?
			firstType = thisType;
			firstSize = thisSize;
			maxSize = thisSize;
		elseif ~strcmp(thisType, firstType) || thisSize ~= firstSize;
			% Is each type the same as the first?
			mixedTypes = true;
			if thisSize ~= firstSize;
				maxSize = max(maxSize, thisSize);
			end%if:not the same size
		end%if:RDMi==1
	
		if ~mixedTypes
			switch thisType
				case 'ltv'
					RDMs(1,:,RDMi) = thisRDM;
				case 'sq'
					RDMs(:,:,RDMi) = thisRDM;
			end%switch:thisType
		end%if:~mixedTypes
		
	end%for:RDMi
	
	% If previous loop was broken...
	
	if mixedTypes
		clear RDMs;
		% All must be square
		RDMs = nan(maxSize, maxSize, nRDMs);
		for RDMi = 1:nRDMs
			thisRDM = RDMs_struct(RDMi).RDM;
			% What type of RDM is this?
			if max(size(thisRDM)) == numel(thisRDM)
				% Then it's in ltv form
				thisRDM = squareform(thisRDM);
			end%if:utv
			RDMs(1:size(thisRDM,1), 1:size(thisRDM,2), RDMi) = thisRDM;
		end%for:RDMi
	end%if:mixedTypes
	
else
    % bare already
    RDMs=RDMs_struct;
    nRDMs=size(RDMs,3);
end
