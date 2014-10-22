function [reducedRDMs,validConditionsLOG]=reduceRDMsToValidConditionSet(RDMs)
% reduces a set of RDMs to those experimental conditions that have all
% entries valied (i.e. non NaN) in all RDMs. the RDMs may be passed
% in square or upper triangular form and as array or struct with additional
% information. the reduced RDMs will have the same format as those
% passed.


%% convert from struct and/or utv form
% OLD version: RDMs_stacked_sq=squareAndStackRDMs(RDMs);
RDMs_stacked_sq=unwrapRDMs(squareRDMs(RDMs));
[n,n,nRDMs]=size(RDMs_stacked_sq);


%% determine valid conditions set
% reduce to a single nan-indication RDM
nanLOG_RDM=any(isnan(RDMs_stacked_sq),3);
% show(nanLOG_RDM);

% count diagonal elements among nans
nanLOG_RDM(logical(eye(n)))=true;

% find all-nan rows and columns
allNanColsLOG=sum(nanLOG_RDM,1)==n;
allNanRowsLOG=sum(nanLOG_RDM,2)==n;


% check if pattern of nans is consistent in rows and columns, terminate with error otherwise
if any(allNanColsLOG~=allNanRowsLOG')
    error('reduceRDMsToValidConditionSet: pattern of NaNs is not consistent in rows and columns.');
else
    validConditionsLOG=~allNanColsLOG;
end

% check if all nans have been removed, terminate with error otherwise
reducedNanLOG_RDM=nanLOG_RDM(validConditionsLOG,validConditionsLOG);

if any(isnan(reducedNanLOG_RDM(:)))
    error('reduceRDMsToValidConditionSet: isolated NaNs found.');
end


%% reduce all RDMs
RDMs_stacked_sq=RDMs_stacked_sq(validConditionsLOG,validConditionsLOG,:);


%% convert back to original format
if isstruct(RDMs)
    reducedRDMs=RDMs;
    
    for RDMI=1:nRDMs
        if length(RDMs(RDMI).RDM)==numel(RDMs(RDMI).RDM)
            % upper triangle form
            reducedRDMs(RDMI).RDM=vectorizeRDM(RDMs_stacked_sq(:,:,RDMI));
        else
            % square form
            reducedRDMs(RDMI).RDM=RDMs_stacked_sq(:,:,RDMI);
        end
    end
else
    if length(RDMs(:,:,1))==numel(RDMs(:,:,1))
        % upper triangular form
        reducedRDMs=vectorizeRDMs(RDMs_stacked_sq);
    else
        % square form
        reducedRDMs=RDMs_stacked_sq;
    end
end

