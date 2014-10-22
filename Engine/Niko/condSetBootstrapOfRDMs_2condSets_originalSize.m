function resampledRDMs_utv=condSetBootstrapOfRDMs_2condSets_originalSize(RDMs,nResamplings,condSetIndexVec)
% uses bootstrap resampling of the conditions set to resample a set of RDMs.
% the resampled RDMs are returned in upper triangle form (rows), stacked
% along the 3rd (index of input RDM) and 4th (resampling index)
% dimensions (for compatibility with square RDMs).
%
% version: _2condSets_originalSize
%       - condSetIndexVec assigns each condition to either set 1 or set 2.
%       - the resampled RDMs have the same size as originals.
%       - the resampled RDMs have the two conditions covering the same
%         index sets as the original RDMs.

%% preparations
RDMs_sq=squareRDMs(unwrapRDMs(RDMs));
[nCond,nCond,nRDMs]=size(RDMs_sq);

[RDMMask,condSet1_LOG,condSet2_LOG,nCondSets,nCond1,nCond2]=convertToRDMMask(condSetIndexVec);

condSet1Is=find(condSet1_LOG);
condSet2Is=find(condSet2_LOG);

if numel(condSet1Is)+numel(condSet2Is)<nCond
    error('condSetBootstrapOfRDMs_2condSets_originalSize: condSetIndexVec must assign each condition to either set 1 or set 2.');
end

% showRDMs(RDMs,2);
% RDMCorrMat(RDMs,3);
% RDMCorrMat(RDMs_sq,4);



%% bootstrap resampling
% the RDM mask region need not be composed of squares on the diagonal.
% therefore we need to separately bootstrap resample two sets of images:
% the vertical and horizontal projection of the upper triangular part of
% the mask. this ensures that the combinations of conditions that the mask
% selects will be present in the resampled data.

resampledRDMs_utv=nan(1,(nCond^2-nCond)/2,nRDMs,nResamplings);

if nCondSets==1
    error('condSetBootstrapOfRDMs_2condSets_originalSize: there must be two condition sets.');
else % nConditionSets==2

    for resamplingI=1:nResamplings
        % bootstrap resample the two condition sets seaprately
        bootstrapCondIs1=condSet1Is(ceil(rand(1,nCond1)*nCond1));
        bootstrapCondIs2=condSet2Is(ceil(rand(1,nCond2)*nCond2));

        % combine the selected index sequences
        inds=nan(nCond,1);
        inds(condSet1Is)=bootstrapCondIs1; % make sure the resampled matrix has the two conditions covering the same index sets
        inds(condSet2Is)=bootstrapCondIs2;
        
        % bootstrap resample to obtain RDMs of equal size as the originals
        bootstrapRDMs_sq=RDMs_sq(inds,inds,:);
        %showRDMs(bootstrapRDMs_sq,10);

        % find off-diagonal matching conditions (occurring when the same
        % condition is selected more than once in the bootstrap sample) and
        % set those dissimilarities from zero to NaN.
        matchingCondI_matrix=squareRDM(pdist(inds,'euclidean')==0);
        bootstrapRDMs_sq(repmat(matchingCondI_matrix,[1 1 nRDMs]))=nan;
        
        resampledRDMs_utv(:,:,:,resamplingI)=vectorizeRDMs(bootstrapRDMs_sq);
    end
end

%% check proportion of off-diagonal NaNs
% NB: bootstrap resampling of the condition set moves zeros from the
% diagonal of the original dissimilarity matrix into off-diagonal locations
% of the bootstrap-resampled dissimilarity matrix. this happens whenever
% the same condition is selected more than once in the bootstrap sample.
% intuitively, the multiple instances of the same condition in the
% bootstrap sample serve to represent (to the degree possible with a
% limited data set) similar conditions that might be sampled if the
% experiment were repeated with a different set of conditions from the same
% population of conditions. however, where the same condition is compared
% to its other instances in the bootstrap sample, an off-diagonal zero
% appears. this aspect of the bootstrap simulation is unrealistic. since
% this affects only a small proportion for large condition sets, we
% nevertheless prefer condition-set bootstrapping to condition-pair-set
% bootstrapping (which would produce resampled similarity matrices with a
% dependence structure an actual similarity matrix cannot possibly have).
% to avoid spurious effects of the off-diagonal zeros (e.g. biased
% correlation estimates), we set the zeros to NaN here. when the bootstrapped
% RDMs are further analyzed (e.g. correlated), these NaNs should be
% treated as missing values.
disp(['Proportion of off-diagonal zeros (which are set to NaN): ',num2str(sum(isnan(resampledRDMs_utv(:)))/numel(resampledRDMs_utv))]);


