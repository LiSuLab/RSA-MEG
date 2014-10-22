function threshold=showNvoxThresholdedMapOnAnatomy(statMap,nVoxToBeMarked,anatomyVolORmap,brainMask,figI,title,twosided,skipNslices)

% shows the statistical statMap on the anatomical background volume anatomyVol
% (x,y,RGB,z) at the threshold marking nVoxToBeMarked voxels.

if ~exist('twosided','var'), twosided=0; end
if ~exist('title','var') || (exist('title','var') && isempty(title)), title=''; end
if ~exist('figI','var'), figI=0; end
if ~exist('brainMask','var')|| (exist('brainMask','var') && isempty(brainMask)), brainMask=true(size(statMap)); end
if ~exist('nVoxToBeMarked','var'), nVoxToBeMarked=2000; end
if ~exist('anatomyVolORmap','var') || (exist('anatomyVolORmap','var') && isempty(anatomyVolORmap)), anatomyVolORmap=statMap; end
if ~exist('skipNslices','var'), skipNslices=1; end

brainMask=logical(brainMask);

if ndims(anatomyVolORmap)==3 % if its a map...
    anatomyVol=map2vol(anatomyVolORmap);
else % must be a vol
    anatomyVol=anatomyVolORmap;
end

if length(statMap)==prod(size(statMap))
    % statMap is a brain-mask vector
    statMap_brainMaskVec=statMap;
    statMap=zeros(size(brainMask));
    statMap(brainMask)=statMap_brainMaskVec;
else
    % statMap is not a brain-mask vector, but a whole volume
    statMap_brainMaskVec=statMap(brainMask);
end

%% determine the threshold marking nVoxToBeMarked voxels
threshold=autothresholdMap(statMap_brainMaskVec,nVoxToBeMarked,twosided);


%% display the thresholded map
lessIsMore=0;
if twosided
    extremeVal=max(abs(statMap_brainMaskVec(:)));
    nMarkedVoxels=sum(abs(statMap_brainMaskVec)>threshold);
else
    extremeVal=max(statMap_brainMaskVec(:));
    nMarkedVoxels=sum(statMap_brainMaskVec>threshold);
end
statMap(~brainMask)=nan;
volWithMap=addStatMapToVol(anatomyVol,statMap,threshold,extremeVal,lessIsMore,twosided);


% disp('min(statMap(:))')
% min(statMap(:))

%h=figure(figI); set(h,'Position',[(contrastIsI-1)*scrsz(3)*2/length(contrastIs) 4*scrsz(4)/8 scrsz(3)*2/length(contrastIs) scrsz(4)/8]); %[left, bottom, width, height]

%showVol(volWithMap,[title,' (',num2str(nMarkedVoxels),' marked)'],figI,'right',1,10,skipNslices);
%showVol(volWithMap,[title,' (',num2str(nMarkedVoxels),' marked)'],figI,'right as in BV');

showVol_titleBar(volWithMap,[title,' (',num2str(nMarkedVoxels),' marked)'],figI,'right as in BV');
