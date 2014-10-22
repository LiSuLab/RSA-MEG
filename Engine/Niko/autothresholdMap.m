% USAGE:            [threshold,binMap]=autothresholdMap(map,nVox,twosided=0)
%
% ARGUMENTS:
%     map:          This is a statistical map, X x Y x Z.  For example, a
%                   32x64x64 map of p values.
%     nVox:         This is the number of significant voxels which WILL be
%                   selected.
%     twosided:     If true, this looks at the absolute values of map, if false,
%                   the true values are taken.
%
% OUTPUTS:
%     threshold:    The value such that exactly nVox voxels' statistics in "map"
%                   are greater.
%     binMap:       This is a X x Y x Z matrix (the same size as map) with 1s
%                   wherever voxels in map have been selected, and 0s wherever
%                   they have not.
%
% GENERAL IDEA:     In a situation where we only want to select the nVox most
%                   significant voxels in a map, this function will give us
%                   those voxels as well as a value for the threshold which they
%                   fall above.

function [threshold,binMap]=autothresholdMap(map,nVox,twosided)

% returns the threshold that highlights nVox voxels in the statistical map
% map. if twosided is passed and 1, then the returned threshold highlights
% nVox voxels when applied to the absolute values of map.

sz=size(map);
map=map(:); % convert to a column vector (explicit conversion only needed because map can be a row vector)

if ~exist('twosided','var'), twosided=0; end;

if isnan(sum(map(:)))
    disp('autothresholdMap: NaNs were found and set to zero.');
    map(isnan(map))=0;
end

if twosided
    map=abs(map);
end

% to speed this up...
% eligibilityFactor=0.1; 
%eligibilityFactor=0.05; % should always work
%eligibleVoxelI=find(map>eligibilityFactor*std(map(:)));
% if size(eligibleVoxelI,1)<=nVox
%     disp('nEligibleVoxels:'); size(eligibleVoxelI,1)
%     disp('nVox:'); nVox
%     disp('eligibilityFactor:'); eligibilityFactor
%     error('ERROR: number of eligible voxels needs to be larger than nVox. please decrease the eligibilityFactor.');
% end

%eligibleVoxelI=find(map); % consider all voxels with a stat value greater than zero
eligibleVoxelI=(1:numel(map))'; % consider all voxels
if nVox>numel(eligibleVoxelI)
    nVox=numel(eligibleVoxelI);
end

eligibleVoxelI_statVal=[eligibleVoxelI, map(eligibleVoxelI)];
clear eligibleVoxelI;

eligibleVoxelI_statVal_sorted=flipud(sortrows(eligibleVoxelI_statVal,2));
eligibleVoxelI_statVal_sorted=[eligibleVoxelI_statVal_sorted;[nVox+1,-10e10]]; % add one below the lowest in case the threshold is supposed to mark all voxels
clear eligibleVoxelI_statVal;
%disp('sorted')

% select nVox many voxels from the top of the list
%topVoxelI=eligibleVoxelI_statVal_sorted(1:nVox,1);

if nVox>0
    threshold=(eligibleVoxelI_statVal_sorted(nVox,2)+eligibleVoxelI_statVal_sorted(nVox+1,2))/2;
else % nVox==0
    threshold=max(map(:))+1;
end
    
clear eligibleVoxelI_statVal_sorted;

map=reshape(map,sz);
binMap=map>threshold;