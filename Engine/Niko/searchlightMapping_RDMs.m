function [smm_bestModel,smm_rs,smm_ps,n,mappingMask_actual]=searchlightMapping_RDMs(t_pats,modelRDMs_utv,searchlightRad_mm,brainMask,voxSize_mm,monitor)
% USAGE
%               [smm_bestModel,smm_rs,smm_ps,n,mappingMask_actual]=
%               searchlightMapping_RDMs(t_pats,modelRDMs_utv,
%                                           searchlightRad_mm,brainMask,
%                                           voxSize_mm,monitor)
%
% ABREVIATIONS
% RDM        similarity matrix
% smm           similarity-matrix-match map
% rs            correlation coefficients
% ps            p values
% mask          3D logical brain mask marking voxels represented in voxel
%               vectors
%
% FUNCTION 
%               performs similarity-matrix-match mapping, a locally
%               multivariate mapping of an fMRI data set for one or more
%               model similarity matrices. at each voxel, the resulting
%               maps are based on the multivariate effects in that voxel's
%               local spherical neighborhood of voxels. the function out
%               puts a map for each model similarity matrix passed as well
%               as a best-model map of indices, indicating the best fitting
%               model at each location.
%               
% ARGUMENTS
% t_pats        condition-by-voxel matrix of spatial activity patterns
%               (usually t maps, computed e.g. by function tMapping.m). each
%               row represents one experimental condition and has as many
%               entries as there are voxels marked in the input data mask.
%               (alternatively t_pats may also contain all voxels within a
%               cuboid volume of the same size as inputDataMask. in that
%               case the voxels not marked in inputDataMask are
%               automatically discarded from t_pats before performing the
%               mapping.)
%
% modelRDMs_utv
%               the model similarity matrices (nModels many rows of
%               vectorized upper triangular similarity matrices).
%               similarity, here, is distance (so dissimilarity would be
%               more precise, but the concept is the same). note that the
%               distance measure used for the the activity-pattern
%               similarity matrix at each searchlight location is
%               1-correlation. to compute similarity matrices use matlab
%               function pdist (built in). to convert between square and
%               upper triangular vector form use squareform.m, or
%               squareRDM.m and vectorizeRDM.m.
%               
% searchlightRad_mm
%               the radius (in millimeters) of the spherical "searchlight"
%               defining the set of voxels to be considered multivariately
%               at each voxel position.
%
% brainMask     3D logical array with the dimensions of the data volumes.
%               this logical mask defines (1) the extent of the
%               information-based map to be computed (mappingMask_request)
%               and (2) the extent of the volume to be used as input data
%               (inputDataMask). if brainMask has 3 dimensions both of
%               these masks are set to the content of brainMask. this means
%               that all voxels marked in brainMask are mapped (if
%               possible) and that only data from voxels in brainMask
%               enters the computations. this is often reasonable. however,
%               note that it entails that searchlight voxels outside the
%               region marked in brainMask will not be included and thus
%               fringe voxels in the infomap are based on fewer voxels.
%               this may be the best solution, since including non-brain
%               voxels would be unfounded.
%
%               if brainMask has a 4th dimension with two levels, then the
%               two volumes define mappingMask_request and inputDataMask in
%               that order.
%               
%               more precisely, mappingMask_request is a logical map
%               indicating voxels, for which an infobased statistic should
%               be computed if possible. the set of voxels, for which the
%               infobased statistic has actually been computed will be
%               returned in mappingMask_actual. this contains the subset of
%               the voxels in mappingMask_request. though
%               mappingMask_request and mappingMask_actual should usually
%               be identical. however, sometimes computation of the
%               infobased statistic may not be possible in some voxels
%               because of a combination of two factors: (a) needed input
%               data voxels may not all be declared input data in
%               inputDataMask and (2) there may be all-zero time-courses
%               which are automatically eliminated. (these can result, for
%               example, from head-motion-correction shifting in zeros from
%               outside the measured slab.) most statistics are still
%               computed if there are missing voxels within the
%               searchlight. the number of input voxels to the infobased
%               statistic is noted in the returned volume n for each mapped
%               voxel in the volume.
%               
%               the size of brainMask also defines volSize_vox, the
%               dimensions assumed in interpreting the spatial structure of
%               the data matrix Y.
%
% voxSize_mm    a triple defining the size of the voxels (in millimeters)
%               along each of the dimensions in order x, y, z. for
%               nonisotropic voxels the spherical voxel searchlight will be
%               appropriately defined to have constant radius (in
%               millimeters) in all directions, though the radius in voxels
%               may then depend on the direction.
%
% OPTIONAL ARGUMENTS 
% [monitor]     if nonzero, the function outputs results as text and
%               figures for monitoring purposes. higher values activate
%               more detailed monitoring output. by default, no monitoring
%               is performed. (output for settings: 0: no output, 1:
%               progress bar and resulting maps.)
%
% RETURN VALUES
% smm_bestModel 3D map of indices indicating the best fitting model
%               similarity matrix. indices reflect the order in which the
%               model similarity matrices appear in modelRDMs_utv.
%
% smm_rs        4D array of 3D maps (x by y by z by model index) of
%               correlations between the searchlight pattern similarity
%               matrix and each of the model similarity matrices.
%
% smm_ps        4D array of 3D maps (x by y by z by model index) of p
%               values computed for each corresponding entry of smm_rs.
%
% n             an array of the same dimensions as the volume, which
%               indicates for each position how many voxels contributed
%               data to the corresponding values of the infomaps.
%               this is the number of searchlight voxels, except at the
%               fringes, where the searchlight may illuminate voxels
%               outside the input-data mask or voxel with all-zero
%               time-courses (as can arise from head-motion correction).
%
% mappingMask_actual
%               3D mask indicating locations for which valid searchlight
%               statistics have been computed.




%% preparations
if ~exist('monitor','var'), monitor=0; end

if ndims(brainMask)==3
	inputDataMask=logical(brainMask);
    mappingMask_request=logical(brainMask);
else
    inputDataMask=logical(brainMask(:,:,:,1));
    mappingMask_request=logical(brainMask(:,:,:,2));
end

if (size(t_pats,2)>sum(inputDataMask(:)))
    t_pats=t_pats(:,inputDataMask(:)); % reduce to inputDataMask
end

volSize_vox=size(inputDataMask);
nModelRDMs=size(modelRDMs_utv,1);
rad_vox=searchlightRad_mm./voxSize_mm;
minMargin_vox=floor(rad_vox);


%% create spherical multivariate searchlight
[x,y,z]=meshgrid(-minMargin_vox(1):minMargin_vox(1),-minMargin_vox(2):minMargin_vox(2),-minMargin_vox(3):minMargin_vox(3));
sphere=((x*voxSize_mm(1)).^2+(y*voxSize_mm(2)).^2+(z*voxSize_mm(3)).^2)<=(searchlightRad_mm^2);  % volume with sphere voxels marked 1 and the outside 0
sphereSize_vox=[size(sphere),ones(1,3-ndims(sphere))]; % enforce 3D (matlab stupidly autosqueezes trailing singleton dimensions to 2D, try: ndims(ones(1,1,1)). )

if monitor, figure(50); clf; showVoxObj(sphere); end % show searchlight in 3D

% compute center-relative sphere SUBindices
[sphereSUBx,sphereSUBy,sphereSUBz]=ind2sub(sphereSize_vox,find(sphere)); % (SUB)indices pointing to sphere voxels
sphereSUBs=[sphereSUBx,sphereSUBy,sphereSUBz];
ctrSUB=sphereSize_vox/2+[.5 .5 .5]; % (c)en(t)e(r) position (sphere necessarily has odd number of voxels in each dimension)
ctrRelSphereSUBs=sphereSUBs-ones(size(sphereSUBs,1),1)*ctrSUB; % (c)en(t)e(r)-relative sphere-voxel (SUB)indices

nSearchlightVox=size(sphereSUBs,1);


%% define masks
validInputDataMask=inputDataMask;
sumAbsY=sum(abs(t_pats),1);
validYspace_logical= (sumAbsY~=0) & ~isnan(sumAbsY); clear sumAbsY;
validInputDataMask(inputDataMask)=validYspace_logical; % define valid-input-data brain mask

t_pats=t_pats(:,validYspace_logical); % reduce t_pats to the valid-input-data brain mask
nVox_validInputData=size(t_pats,2);

mappingMask_request_INDs=find(mappingMask_request);
nVox_mappingMask_request=length(mappingMask_request_INDs);

if monitor
    disp([num2str(round(nVox_mappingMask_request/prod(volSize_vox)*10000)/100),'% of the cuboid volume requested to be mapped.']);
    disp([num2str(round(nVox_validInputData/prod(volSize_vox)*10000)/100),'% of the cuboid volume to be used as input data.']);
    disp([num2str(nVox_validInputData),' of ',num2str(sum(inputDataMask(:))),' declared input-data voxels included in the analysis.']);
end

volIND2YspaceIND=nan(volSize_vox);
volIND2YspaceIND(validInputDataMask)=1:nVox_validInputData;

% n voxels contributing to infobased t at each location
n=nan(volSize_vox);




%% similarity-graph-map the volume with the searchlight
smm_bestModel=nan(volSize_vox);
smm_ps=nan([volSize_vox,nModelRDMs]);
smm_rs=nan([volSize_vox,nModelRDMs]);

if monitor
    h_progressMonitor=progressMonitor(1, nVox_mappingMask_request,  'Similarity-graph-mapping...');
end

for cMappingVoxI=1:nVox_mappingMask_request
    
    if monitor && mod(cMappingVoxI,1000)==0
        progressMonitor(cMappingVoxI, nVox_mappingMask_request, 'Searchlight mapping Mahalanobis distance...', h_progressMonitor);
        %                 cMappingVoxI/nVox_mappingMask_request
    end

    [x, y, z]=ind2sub(volSize_vox,mappingMask_request_INDs(cMappingVoxI));

    % compute (sub)indices of (vox)els (c)urrently (ill)uminated by the spherical searchlight
    cIllVoxSUBs=repmat([x,y,z],[size(ctrRelSphereSUBs,1) 1])+ctrRelSphereSUBs;

    % exclude out-of-volume voxels
    outOfVolIs=(cIllVoxSUBs(:,1)<1 | cIllVoxSUBs(:,1)>volSize_vox(1)|...
                cIllVoxSUBs(:,2)<1 | cIllVoxSUBs(:,2)>volSize_vox(2)|...
                cIllVoxSUBs(:,3)<1 | cIllVoxSUBs(:,3)>volSize_vox(3));

    cIllVoxSUBs=cIllVoxSUBs(~outOfVolIs,:);

    % list of (IND)ices pointing to (vox)els (c)urrently (ill)uminated by the spherical searchlight
    cIllVox_volINDs=sub2ind(volSize_vox,cIllVoxSUBs(:,1),cIllVoxSUBs(:,2),cIllVoxSUBs(:,3));

    % restrict searchlight to voxels inside validDataBrainMask
    cIllValidVox_volINDs=cIllVox_volINDs(validInputDataMask(cIllVox_volINDs));
    cIllValidVox_YspaceINDs=volIND2YspaceIND(cIllValidVox_volINDs);

    % note how many voxels contributed to this locally multivariate stat
    n(x,y,z)=length(cIllValidVox_YspaceINDs);
    
    [rs,ps,bestModelI]=findBestModelSimilarityMatrix(t_pats(:,cIllValidVox_YspaceINDs),modelRDMs_utv);
    smm_bestModel(x,y,z)=bestModelI;
    smm_ps(x,y,z,:)=ps;
    smm_rs(x,y,z,:)=rs;
    
end

if monitor
	fprintf('\n');%cw2-2010
    close(h_progressMonitor);
end

mappingMask_actual=mappingMask_request;
mappingMask_actual(isnan(sum(smm_rs,4)))=0;



%% save results
% save cuboid maps
save('smm','smm_bestModel','smm_rs','smm_ps','n','mappingMask_actual');


%% visualize
if monitor
    aprox_p_uncorr=0.001;
    singleModel_p_crit=aprox_p_uncorr/nModelRDMs; % conservative assumption model proximities nonoverlapping

    smm_min_p=min(smm_ps,[],4);
    smm_significant=smm_min_p<singleModel_p_crit;

    vol=map2vol(brainMask);
	vol2=map2vol(brainMask);
    
    colors=[1 0 0
            0 1 0
            0 1 1
            1 1 0
            1 0 1];
    
    for modelRDMI=1:nModelRDMs
        vol=addBinaryMapToVol(vol, smm_significant&(smm_bestModel==modelRDMI), colors(modelRDMI,:));
% 		vol2=addBinaryMapToVol(vol2, smm_bestModel==modelRDMI, colors(modelRDMI,:));
    end
    
    showVol(vol);
	
% 	vol2 = vol2*0.1;
% 	vol2(vol<1) = vol(vol<1);
% 	
% 	showVol(vol2);
    
end
