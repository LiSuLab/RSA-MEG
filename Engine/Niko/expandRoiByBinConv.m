function newRoi=expandRoiByBinConv(roi, kernelRelRoi, sizeVol)

% USAGE:     newRoi=expandRoiByBinConv(roi, kernelRoi, sizeVol)
%
% FUNCTION:  expand a 3D region of interest (roi) isotropically in all
%            directions by binary convolution with another region
%            (binary 3D "kernel").
% 
% ARGUMENTS:
% roi        (r)egion (o)f (i)nterest
%            a matrix of voxel positions
%            each row contains ONE-BASED coordinates (x, y, z) of a voxel.
%
% kernelRelRoi the binary kernel (another 3D roi) as a roi matrix (nVox by 3)
%            of voxel positions.
%            each row [x, y, z] contains the RELATIVE coordinates (center of 
%            the sphere at (0, 0, 0)).
%
% sizeVol    a row triple defining the dimensions of the volume in which the
%            region is to grow: [sizeX sizeY sizeZ].
%
% RETURN VALUES:
% newRoi     the resulting expanded roi (ONE-BASED)
%
% SIMILAR FUNCTION:
% expandRoi  expands a roi by adding as specified number of voxels, first
%            filling an adjacent layer before starting the next one


% DEFINE THE VOLUME
vol=zeros(sizeVol);


% DRAW THE EXPANDED ROI INTO THE VOLUME
%draw the original roi
vol(sub2ind(size(vol),roi(:,1),roi(:,2),roi(:,3)))=1;

redExpRoi=[];              % REDundant EXPanded ROI
for roiVoxI=1:size(roi,1)
    redExpRoi=[redExpRoi; ones(size(kernelRelRoi,1),1)*roi(roiVoxI,:)+kernelRelRoi];
end

%DEBUG: redExpRoi

% ILLEGAL
% i=1:size(roi,1);
% j=1:size(kernelRelRoi,1);
% redExpRoi=zeros(size(roi,1)*size(kernelRelRoi,1),3);
% redExpRoi(j*size(roi,1)+i)=roi(i,:)+kernelRelRoi(j,:)

vol(SUB2IND(size(vol),redExpRoi(:,1),redExpRoi(:,2),redExpRoi(:,3)))=1;

% EXPRESSIONS BELOW HAVE DIFFERENT MEANINGS (which i don't understand)
% vol(redExpRoi(:,1),redExpRoi(:,2),redExpRoi(:,3))=1; %equivalent?
% showVol(vol, 346, 'right');
% vol(redExpRoi)=1; %equivalent?
% showVol(vol, 347, 'right');

[ivolx,ivoly,ivolz]=ind2sub(size(vol), find(vol));

newRoi=[ivolx, ivoly, ivolz]; % return ONE-BASED roi

