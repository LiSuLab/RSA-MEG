function showMap(map, iFig, titleString, userChosenColormap)
% USAGE
%       showMap(map[, iFig, titleString])
%
% NOTE
%       very similar to: showBVmap(map, iFig)
%
% FUNCTION
%       displays a 3D statistical map (represented as 3D matrix)
%
% GENERAL IDEA: Use this to visualise a 3dimensional statistical map.
%               The map goes in "map", the figure number goes in
%               "iFig", the title goes in "titleString" and if a
%               colormap other than "gray" is wanted, this goes in
%               "userChosenColormap".


%% preparations
if ~exist('iFig','var')
    iFig=0;
end

if ~exist('userChosenColormap', 'var')
    userChosenColormap = gray;
end

if iFig
    h=figure(iFig); 
else
    h=figure;
end
clf; set(h,'Color','w');
sizeZ=size(map,3);
nHorVerPanels=ceil(sqrt(sizeZ));

mapmax=max(max(max(map)));
mapmin=min(min(min(map)));

%DEBUG
%map(3,1,1)=mapmax;

imagemap=(map-mapmin)/(mapmax-mapmin)*64;


margin=0;
width=(1-(nHorVerPanels-1)*margin)/nHorVerPanels;
height=(1-(nHorVerPanels-1)*margin)/nHorVerPanels;


%% draw slices
for z=1:sizeZ
    
    %subplot(nHorVerPanels,nHorVerPanels,z);
    
    %subplot(nVerPanels,nHorPanels,sliceI);
    left=mod(z-1,nHorVerPanels)*(width+margin);
    bottom=1-ceil(z/nHorVerPanels)*(height+margin);
    subplot('Position',[left bottom width height]);

    image(fliplr(imagemap(:,:,z)'));
    %imagesc(map(:,:,z)');

    %colormap bone;
    colormap(userChosenColormap);
    
    axis([1 size(map,1) 1 size(map,2)]);
    axis equal;
    axis off;

    %set(gca,'Visible','off');

end


%% add title
if ~exist('titleString','var')
    titleString=['black=',num2str(mapmin),', white=',num2str(mapmax)];
end
title(titleString,'Color',[.5 .5 .5]);

% h=axes('Parent',gcf); hold on;
% set(h,'Visible','off');

%colorbar;