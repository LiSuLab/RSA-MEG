function newRoi=resizeRoi(roi, map, nVox, mask)

% USAGE
%           newRoi=resizeRoi(roi, map, nVox[, mask])
%
% FUNCTION
%           redefine a roi to make it conform to the size
%           given by nVox. the new roi is defined by a region
%           growing process, which (1) is seeded at the voxel
%           that has the maximal statistical parameter within
%           the passed statistical map and (2) is prioritized
%           by the map's values.
%
% ARGUMENTS
% roi       (r)egion (o)f (i)nterest
%           a matrix of voxel positions
%           each row contains ONE-BASED coordinates (x, y, z) of a voxel.
%
% map       a 3D statistical-parameter map
%           the map must match the volume, relative to which
%           the roi-voxel coords are specified in roi.
%
% nVox      number of voxels the resized roi is to have
%
% [mask]    optional binary mask. if present, the region growing is
%           restricted to the nonzero entries of it.
%
% GENERAL IDEA: Given a 3d statistical map and an roi inside (which
%               may be masked using "mask", which has 1s inside and
%               0s outside); we can create a new roi of size nVox
%               which captures the most statistically significant
%               voxels near the original roi.  This is done by
%               picking the most significant voxel inside the
%               original roi and then adding neighbouring voxels in
%               order of their significance.

% PARAMETERS
%maxNlayers=2;   %defines how many complete layers are maximally added


if ~exist('mask','var') || (exist('mask','var') && isempty(mask))
    mask=ones(size(map));
end

map(~mask)=min(map(:));

% DEFINE THE VOLUME
vol=zeros(size(map));


% FIND THE SEED (A MAXIMAL MAP VALUE IN ROI&mask)
mapINDs=sub2ind(size(vol),roi(:,1),roi(:,2),roi(:,3)); %single indices to MAP specifying voxels in the roi
roimap=map(mapINDs);                                      %column vector of statistical-map subset for the roi
[roimax,roimax_roimapIND]=max(roimap);                    %the maximal statistical map value in the roi and its index within roimap
seed_mapIND=mapINDs(roimax_roimapIND);                    %seed index within map


newRoi=[];
if nVox==0; return; end;
    

if mask(seed_mapIND)
    vol(seed_mapIND)=1;
    [x,y,z]=ind2sub(size(vol),seed_mapIND);
    newRoi=[newRoi;[x,y,z]];
end

if nVox==1 || isempty(newRoi)
    return;
end

% GROW THE REGION
for i=2:nVox
    
    % DEFINE THE FRINGE
    cFringe=vol;
    [ivolx,ivoly,ivolz]=ind2sub(size(vol),find(vol));
    
    superset=[ivolx-1,ivoly,ivolz;
              ivolx+1,ivoly,ivolz;
              ivolx,ivoly-1,ivolz;
              ivolx,ivoly+1,ivolz;
              ivolx,ivoly,ivolz-1;
              ivolx,ivoly,ivolz+1];
    
    
    % exclude out-of-volume voxels
    outgrowths = superset(:,1)<1 | superset(:,2)<1 | superset(:,3)<1 | ...
                 superset(:,1)>size(vol,1) | superset(:,2)>size(vol,2) | superset(:,3)>size(vol,3);
    
    superset(find(outgrowths),:)=[];
             
    
    % draw the layer (excluding multiply defined voxels)
    cFringe(sub2ind(size(vol),superset(:,1),superset(:,2),superset(:,3)))=1;
    cFringe=cFringe&mask;
    cFringe=cFringe-vol;
    
    if size(find(cFringe),1)==0
        break; % exit the loop (possible cause of empty fringe: the whole volume is full)
    end
    
    % FIND A MAXIMAL-MAP-VALUE FRINGE VOXEL...
    mapINDs=find(cFringe);                                    %single indices to MAP specifying voxels in the fringe
    fringemap=map(mapINDs);                                   %column vector of statistical-map subset for the fringe
    [fringemax,fringemax_fringemapIND]=max(fringemap);        %the maximal statistical map value in the roi and its index within roimap
    fringemax_mapIND=mapINDs(fringemax_fringemapIND);         %seed index within map
       
    % ...INCLUDE IT
    vol(fringemax_mapIND)=1;
    [x,y,z]=ind2sub(size(vol),fringemax_mapIND);
    newRoi=[newRoi;[x,y,z]];
    
end
   
%newRoi=newRoi-1; %return zero-based resized roi
