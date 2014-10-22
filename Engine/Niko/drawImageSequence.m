function drawImageSequence(imageStruct,sequence,start_xy,end_xy,nRows,transparentCol)
% USAGE
%         drawImageSequence(imageStruct,sequence,start_xy,end_xy[,transparentCol])
% FUNCTION
%         places a set of images at equal distances along a line in a 2d
%         coordinate system. the images are in field 'image' or the
%         structured array 'imageStruct'. the images selected and their
%         sequence is determined by argument 'sequence', which contains a
%         sequence of indices referring to the array 'imageStruct'. the
%         first image will appear centered on coordinates start_xy, the
%         last centered on end_xy (2-element vectors containing x and y
%         coordinates in that order).


%% preparations
if ~exist('transparentCol','var'), transparentCol=[128 128 128]; end
if ~exist('nRows','var'), nRows=2; end
nImagesInSeq=numel(sequence);

xlim=get(gca,'XLim');
ylim=get(gca,'YLim');

%daspect(daspect);
% looks moot, but causes the current data aspect ratio to be preserved when the figure is resized

da=daspect; % need this to render image pixels square

set(gca,'XTick',[]); % switch off horizontal-axis ticks and image index numbers
hold on;



%% compute image size
% images assumed to be square
seqVec_sqA=(end_xy-start_xy)./da(1:2); % vector in sequence direction (in square axis)
imageWidth=max(abs(seqVec_sqA))/(nImagesInSeq-1)*nRows;
imW_ax=imageWidth*da(1);
imW_ay=imageWidth*da(2);


%% prepare multi-row arrangement
% compute vector orthogonal to sequence direction
orthVec=-(null(seqVec_sqA)'*imageWidth).*da(1:2);

% shift start and end positions to center the multiple rows on the
% requested sequence line
% start_xy=start_xy-orthVec*(nRows-1)/2;
% end_xy=end_xy-orthVec*(nRows-1)/2;


%% arrange the images
for sequenceI=1:nImagesInSeq
    imageI=sequence(sequenceI);
    xy=start_xy+(end_xy-start_xy)/(nImagesInSeq-1)*(sequenceI-1);
    
    xy=xy+mod(sequenceI,nRows)*orthVec; % lateral displacement for multiple rows
    
    %[xs,ys,rgb3]=size(imageStruct(imageI).image); % assuming square images for now, so this isn't needed
    transparent=imageStruct(imageI).image(:,:,1)==transparentCol(1) & imageStruct(imageI).image(:,:,2)==transparentCol(2) & imageStruct(imageI).image(:,:,3)==transparentCol(3);

    % black disks underneath
    angles=0:0.1:2*pi;
    X=sin(angles)*imW_ax/2+xy(1);
    Y=cos(angles)*imW_ay/2+xy(2);
    Z=-2*ones(size(X));
    patch(X,Y,Z,[0 0 0]);
    
    %rectangle('Position',[xy(1)-imW_ax/2  xy(2)-imW_ay/2 imW_ax imW_ay],'Curvature',[1 1],'FaceColor','k','EdgeColor','none','EraseMode','normal');
    %plot(xy(1),xy(2),'o','MarkerSize',10,'MarkerFaceColor','k');
    
    % low-level version of image function
    image('CData',imageStruct(imageI).image,'XData',xy(1)+[-imW_ax/2, imW_ax/2],'YData',xy(2)+[imW_ay/2, -imW_ay/2],'AlphaData',~transparent,'EraseMode','normal');
    
    % high-level version of image function
    % image(xy(1)+[-imW_ax/2, imW_ax/2],xy(2)+[imW_ay/2, -imW_ay/2],imageStruct(imageI).image,'AlphaData',~transparent,'EraseMode','normal');
    
    xlim(1)=min(xlim(1),xy(1)-imW_ax/2);
    xlim(2)=max(xlim(2),xy(1)+imW_ax/2);
    
    ylim(1)=min(ylim(1),xy(2)-imW_ay/2);
    ylim(2)=max(ylim(2),xy(2)+imW_ay/2);
    
end

set(gca, 'XLim',xlim, 'YLim',ylim);
