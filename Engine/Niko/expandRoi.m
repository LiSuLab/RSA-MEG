function newRoi=expandRoi(roi, nVox, sizeVol)

%USAGE:     newRoi=expandRoi(roi, nVox)
%FUNCTION:  expand a region of interest (roi) isotropically in all directions
%           (by adding nVox many voxels, first filling an adjacent layer
%           before starting the next one)
%
%roi        (r)egion (o)f (i)nterest
%           a matrix of voxel positions
%           each row contains ONE-BASED coordinates (x, y, z) of a voxel.
%
%nVox       number of voxels to expand the passed roi by
%
%sizeVol    a row triple defining the dimensions of the volume in which the
%           region is to grow: [sizeX sizeY sizeZ].
% GENERAL IDEA:  Given an roi "roi" in a volume of size "sizeVol", if you
%                want to expand it in all directions (isotropically) by nVox,
%                this function will give you a new RoI "newRoi".  This links
%                well with addRoiToVol which will visualize the old and new
%                RoIs.


% PARAMETERS

% DEBUG
%  roi=[10 10 10;
%     11 11 11;
%     10 11 10;
%     11 12 11;
%     11 13 11;
%     11 14 11];
%roi=[10 10 10; 9 9 9];
%roi=[10 10 10];
%nVox=8;


% DEFINE THE VOLUME
vol=zeros(sizeVol);
roi_ob=roi; %+1; %(o)ne-(b)ased

% DRAW THE ROI INTO THE VOLUME
vol(SUB2IND(size(vol),roi_ob(:,1),roi_ob(:,2),roi_ob(:,3)))=1;
orivol=vol;

while true
    % DEFINE THE CURRENT OUTER LAYER
    cLayer=vol;
    [ivolx,ivoly,ivolz]=ind2sub(size(vol),find(vol));

    superset=[ivolx-1,ivoly,ivolz;
              ivolx+1,ivoly,ivolz;
              ivolx,ivoly-1,ivolz;
              ivolx,ivoly+1,ivolz;
              ivolx,ivoly,ivolz-1;
              ivolx,ivoly,ivolz+1];
    
    
    % exclude out-of-volume voxels
    outgrowths = superset(:,1)<1 | superset(:,2)<1 | superset(:,3)<1 | ...
                 superset(:,1)>sizeVol(1) | superset(:,2)>sizeVol(2) | superset(:,3)>sizeVol(3);
    
    superset(find(outgrowths),:)=[];
             
    
    % draw the layer (excluding multiply defined voxels)
    cLayer(SUB2IND(size(vol),superset(:,1),superset(:,2),superset(:,3)))=1;
    cLayer=cLayer-vol;
    
    % ADD THE WHOLE LAYER IF APPROPRIATE
    cLayerSize=size(find(cLayer),1);

    if(cLayerSize > nVox )
        break % exit the while loop
    end
    
    vol=vol+cLayer;
    nVox=nVox-cLayerSize;
end

% ADD THE REMAINING VOXELS BY CHOOSING RANDOM ELEMENTS OF THE CURRENT LAYER
indices=find(cLayer);
indices=indices(randperm(size(indices,1)));
indices=indices(1:nVox);
vol(indices)=1;

%DEBUG
%vol+orivol

[ivolx,ivoly,ivolz]=IND2SUB(size(vol), find(vol));

newRoi=[ivolx-1, ivoly-1, ivolz-1]; % return zero-based roi


