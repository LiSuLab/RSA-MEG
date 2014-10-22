% roi2mask is a function with two arguments:
%  ARGUMENTS:
%    roi: this is a 3xn matrix where each row contains the coordinates for
%         a point inside the roi
%    volSize_vox: this is a 1x3 vector containing the dimensions of the
%                 scanned volume.  E.g., [32 64 64]
%  RETURNS:
%    mask: a volume of size volSize_vox which is all 0s, except for the
%          points indicated by roi, which are 1s.

function mask=roi2mask(roi,volSize_vox)

roi_INDs=sub2ind(volSize_vox,roi(:,1),roi(:,2),roi(:,3)); %single indices to MAP specifying voxels in the roi
mask=false(volSize_vox);
mask(roi_INDs)=true;
