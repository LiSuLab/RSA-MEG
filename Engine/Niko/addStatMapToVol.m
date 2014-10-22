function vol=addStatMapToVol(vol, map, critVal, extremeVal, lessIsMore, showNeg, transparency)

% USAGE:        vol=addStatMapToVol(vol, map, [critVal, extremeVal, lessIsMore=0, showNeg=1, transparency=0])
%
% FUNCTION:     to superimpose the statistical map "map" to the true-color
%               volume "vol".
%
% ARGUMENTS:
% vol           the volume as a stack of true-color slices: X by Y by 3 by Z.
%               the third dimension encodes the color component (red, green,
%               blue).
%               anatomically, the X axis points to the left, the Y axis to
%               the back of the brain, and the Z axis up.
%               if vol contains a scalar zero, the map itself is used as a
%               grayscale background to its colored peaks.
%
% map           a statistical map as a 3D array of double-precision floats
%
% critVal       threshold determining what part of the map is superimposed:
%               only voxels whose ABSOLUTE map value exceeds critVal are
%               marked
%
% extremeVal    absolute value of the map entries above which the color
%               scale does not differentiate anymore
%
% [lessIsMore=0]  if this optional argument is nonzero, then instead of
%               marking voxels whose absolute map value EXCEEDS the
%               threshold, the function marks the voxels whose absolute map
%               value IS SMALLER than the threshold. this should be nonzero
%               for p maps.
%
% [showNeg=1]   if this optional argument is nonzero or missing (defaults to 1),
%               negative map values whose absolute value exceeds the
%               threshold are highlighted. if showNeg is zero, only
%               positive values exceeding the threshold are highlighted.
%
% RETURN VALUE: true-color volume (X by Y by 3 by Z) with the thresholded
%               map superimposed
%
% GENERAL IDEA: Given a volume of colour information and a statistical map
%               (for example, r values), this function will overwrite the
%               colour information in the original volume at the voxels
%               where the statistics in the map are significant.  In these
%               voxels, the colour will reflect the significance level, but
%               can be capped using extremeVal.  So, for statistics in the
%               map which are less than critVal, no colour will be changed.
%               For values between critVal and extremeVal, the colour will
%               be given by the statistics; and for values greater than
%               extremeVal, the colour will be at the maximum but
%               unchanging.  Finally, if lessIsMore is used, the relations
%               are reversed (since, for example using p values, to be
%               significant is to be *less* than a given threshold).




%% RESCALE THE STATISTICAL MAP
if ~exist('critVal','var')
    critVal=0;
end

if ~exist('extremeVal','var')
    extremeVal=max(abs(map(:)));
end

if ~exist('lessIsMore','var')
    lessIsMore=0;
end

if ~exist('showNeg','var')
    showNeg=1;
end

if ~exist('transparency','var')
    transparency=0;
end

if numel(vol)==1 && vol==0
    vol=map2vol(map);
end

if lessIsMore
    % flip positive and negative part of the map value axis
    % such that the maximum and minimum fall on zero
    % and the extreme values (positive and negative)
    % correspond to small values in the original map
    mx=max(abs(map(:)));
    mn=-mx;
    map(map>=0)=mx-map(map>=0);
    map(map<0)=mn-map(map<0);
    
    critVal=mx-critVal;
    extremeVal=mx-extremeVal;
end


NaN_flags=isnan(map);
if sum(NaN_flags(:))
    disp('NaNs were found in the map and set to 0.')
    map(NaN_flags)=0;
end


imagemap=double(map);

%shift to-be-shown portions (abs(map) between critVal and extremeVal) to span
%[1,31] (negative map values) and [33,64] (positive map values) 
imagemap(map>critVal)=(imagemap(map>critVal)-critVal)/(extremeVal-critVal)*(64-33)+33;
imagemap(map<-critVal)=(imagemap(map<-critVal)+critVal)/(extremeVal-critVal)*(31-1)+31;
%figure(iFig-2); clf; imagesc(imagemap(:,:,5));

imagemap(abs(map)<=critVal)=32;
imagemap(map>extremeVal)=64;
imagemap(map<-extremeVal)=1;
%figure(iFig-1); clf; imagesc(imagemap(:,:,5));

if ~showNeg
    imagemap(map<0)=32;
end


%% DEFINE THE COLORMAP
bgc=[0 0 0]; %background color
critposc=[1 0.3 0]; %color at critical positive value
critnegc=[0 0.3 1]; %color at critical negative value
maxposc=[1 1 0]; %color at maximum positive value
maxnegc=[0 1 0.3]; %color at maximum negative value

for i=1:31
    distToMin=(i-1)/30;
    distToMid=(31-i)/30;
    cm(i,:)=distToMin*critnegc+distToMid*maxnegc;
end

cm(32,:)=bgc;

for i=33:64
    distToMax=(64-i)/31;
    distToMid=(i-33)/31;
    cm(i,:)=distToMax*critposc+distToMid*maxposc;
end


%% MARK THE VOXELS
toBeMarked_INDs=find(imagemap~=32);

volXYZ3=permute(vol,[1 2 4 3]);

  redmap=volXYZ3(:,:,:,1);
greenmap=volXYZ3(:,:,:,2);
 bluemap=volXYZ3(:,:,:,3);

  redmap(toBeMarked_INDs)=  redmap(toBeMarked_INDs)*transparency + (1-transparency)*cm(round(imagemap(toBeMarked_INDs)),1);
greenmap(toBeMarked_INDs)=greenmap(toBeMarked_INDs)*transparency + (1-transparency)*cm(round(imagemap(toBeMarked_INDs)),2);
 bluemap(toBeMarked_INDs)= bluemap(toBeMarked_INDs)*transparency + (1-transparency)*cm(round(imagemap(toBeMarked_INDs)),3);

volXYZ3(:,:,:,1)=  redmap;
volXYZ3(:,:,:,2)=greenmap;
volXYZ3(:,:,:,3)= bluemap;

vol=permute(volXYZ3,[1 2 4 3]);
