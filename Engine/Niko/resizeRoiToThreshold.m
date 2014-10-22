function newRoi=resizeRoiToThreshold(roi, map, threshold)

% USAGE
%           newRoi=resizeRoiToThreshold(roi, map, threshold)
%
% FUNCTION
%           redefine a region of interest (ROI) to the largest contiguous
%           suprathreshold voxel set including the previous roi's
%           maximum-map-value voxel. the new roi is defined by a region
%           growing process, which (1) is seeded at the voxel that has the
%           maximal statistical parameter within the passed statistical map
%           and (2) is prioritized by the map's values.
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
% threshold 
%           the threshold voxel above which are included
%
% GENERAL IDEA: Given an roi inside a statistical map, this funciton will
%               resize the original roi to a new roi by starting with the
%               most significant map value inside the original roi (if it
%               is higher than the threshold) and then picking neighbouring
%               voxels whose statistics are also higher than the threshold
%               given in the argument "threshold".

% PARAMETERS
%maxNlayers=2;   %defines how many complete layers are maximally added



% DEFINE THE VOLUME
vol=zeros(size(map));


% FIND THE SEED (A MAXIMAL MAP VALUE IN ROI)
mapINDs=sub2ind(size(vol),roi(:,1),roi(:,2),roi(:,3)); %single indices to MAP specifying voxels in the roi
roimap=map(mapINDs);                                      %column vector of statistical-map subset for the roi
[roimax,roimax_roimapIND]=max(roimap);                    %the maximal statistical map value in the roi and its index within roimap
seed_mapIND=mapINDs(roimax_roimapIND);                    %seed index within map


newRoi=[];

% the first candidate is the seed voxel (max in original roi)
fringemax_mapIND=seed_mapIND;
fringemax=roimax;


% GROW THE REGION
while fringemax>threshold
    
    % ...INCLUDE IT
    vol(fringemax_mapIND)=1;
    [x,y,z]=ind2sub(size(vol),fringemax_mapIND);
    size(newRoi,1)
    newRoi=[newRoi;[x,y,z]];

    
    % DEFINE THE FRINGE
    cFringe=vol;
    [ivolx,ivoly,ivolz]=IND2SUB(size(vol),find(vol));
    
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
    cFringe=cFringe-vol;
    
    if size(find(cFringe),1)==0
        break; % exit the loop (possible cause of empty fringe: the whole volume is full)
    end
    
    % FIND A MAXIMAL-MAP-VALUE FRINGE VOXEL...
    mapINDs=find(cFringe);                                    %single indices to MAP specifying voxels in the fringe
    fringemap=map(mapINDs);                                   %column vector of statistical-map subset for the fringe
    [fringemax,fringemax_fringemapIND]=max(fringemap);        %the maximal statistical map value in the roi and its index within roimap
    fringemax_mapIND=mapINDs(fringemax_fringemapIND);         %seed index within map
       
    
end
   
%newRoi=newRoi-1; %return zero-based resized roi
