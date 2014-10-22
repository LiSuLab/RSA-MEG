function [RDMMask,condSet1_LOG,condSet2_LOG,nCondSets,nCond1,nCond2]=convertToRDMMask(condSetIndexVectorORRDMMask)
% converts a conditions-set index vector into a RDM mask. a conditions
% set index vector flags the conditions as 0, 1 or 2. conditions flagged as
% 0 are excluded in the mask. conditions flagged as 1 are vertically
% selected. conditions marked as 2 are horizontally selected. the
% RDMMask will have an entries set to true, if both its row and its
% column have been selected. if the conditions-set index vector contains no
% 2s, then the 1s define the vertical and the horizontal set. (if vertical
% and horizontal selection sets are to overlap, they are required to be
% identical here.) 

n=length(condSetIndexVectorORRDMMask);

if length(condSetIndexVectorORRDMMask)==numel(condSetIndexVectorORRDMMask)
    % it's a condition-set index vector...
    condSetIs_vector=condSetIndexVectorORRDMMask;
    condSetIndices=unique(condSetIs_vector(condSetIs_vector~=0)); % '0' indicates: to be excluded
    nCondSets=numel(condSetIndices);

    RDMMask=false(n,n);

    if nCondSets==1;
        % reduce RDMMask
        condSet1_LOG=(condSetIs_vector==condSetIndices(1));
        condSet2_LOG=condSet1_LOG;
        RDMMask(condSet1_LOG,condSet1_LOG)=true;
    elseif nCondSets==2;
        % reduce RDMMask (different input and output sets)
        condSet1_LOG=(condSetIs_vector==condSetIndices(1));
        condSet2_LOG=(condSetIs_vector==condSetIndices(2));
        RDMMask(condSet1_LOG,condSet2_LOG)=true;
        RDMMask(condSet2_LOG,condSet1_LOG)=true;
    end
else
    RDMMask=condsSetIndexVectorORRDMMask;
    condSet1_LOG=nan;
    condSet2_LOG=nan;
end

nCond1=sum(condSet1_LOG);
nCond2=sum(condSet2_LOG);
