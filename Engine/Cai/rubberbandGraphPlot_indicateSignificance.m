function rubberbandGraphPlot_indicateSignificance(coords_xy,distmat,pmat)

% Like Niko's rubberbandGraphPlot, but colours the rubberbands according to
% significance values held in the argument pmat
%
% Edited by Cai Wingfield 3-2010

%% control variables
nDistortionBins=11;


%% preparations
distmat_utv=vectorizeRDM(distmat);
nPoints=size(coords_xy,1);

hold on;
plot(coords_xy(:,1),coords_xy(:,2),'LineStyle','none');
axis tight equal;
set(gca,'Units','points'); lbwh_pts=get(gca,'Position');
unitsPerPoint=range(get(gca,'XLim'))/lbwh_pts(3);


%% compute 2D distmat
distmat2D_utv=pdist(coords_xy,'euclidean');


%% compute connection thicknesses
boundingBoxArea=prod(range(coords_xy));
areas_utv=distmat_utv/sum(distmat_utv)*boundingBoxArea*connectionAreaProportion;
thickness_utv=areas_utv./distmat2D_utv; % area=distmat2D*thickness, proportional to distmat


%% map from upper-triangular-vector (utv) form of distance matrix to point indices i, j
[i,j]=ndgrid(1:nPoints,1:nPoints);
i_utv=vectorizeRDM(i);
j_utv=vectorizeRDM(j);


%% draw the connections
%lc=[.9 .9 .9]; % line color
pmat_utv = vectorizeRDM(pmat);
p0colour = [0.90; 0.90; 0.90];
p1colour = [0.90; 0.68; 0.68];
p2colour = [0.90; 0.45; 0.45];
p3colour = [0.90; 0.00; 0.00];
lc_utm = repmat(p0colour, 1, numel(pmat_utv));
for i = 1:numel(lc_utm)
	if pmat_utv(i) < 0.05
		lc_utm(:, i) = p1colour;
	end%if
	if pmat_utv(i) < 0.001
		lc_utm(:, i) = p2colour;
	end%if
	if pmat_utv(i) < 0.0001
		lc_utm(:, i) = p3colour;
	end%if
end%for:i

mnt=min(thickness_utv);
mxt=max(thickness_utv);
thicknessCtrs=linspace(mnt,mxt,nDistortionBins);
binWidth=thicknessCtrs(2)-thicknessCtrs(1);
thicknessEdges=[thicknessCtrs-binWidth/2,thicknessCtrs(end)+binWidth/2];

if mxt-mnt<1e-6 % if no distortion...
    % select coords of all pairs
    sourcePositions_xy=coords_xy(i_utv,:);
    targetPositions_xy=coords_xy(j_utv,:);
    Z=[repmat(-1,size(sourcePositions_xy(:,1)')); repmat(-1,size(sourcePositions_xy(:,1)'))];

    % draw the lines
	for i = 1:size(coords_xy,1)
		line([sourcePositions_xy(i,1)'; targetPositions_xy(i,1)'],[sourcePositions_xy(i,2)'; targetPositions_xy(i,2)'],Z,'Color',lc_utm(:,i),'LineWidth',thickness_pts);
	end%for:i


else % if distortions present...
    for distortionBinI=1:nDistortionBins

        % find pairs to be connected by lines of this thickness
        pairs_utvLOG=thicknessEdges(distortionBinI)<thickness_utv & thickness_utv<=thicknessEdges(distortionBinI+1);
        thickness_pts=mean(thicknessEdges(distortionBinI:distortionBinI+1))/unitsPerPoint;

        is=i_utv(pairs_utvLOG);
        js=j_utv(pairs_utvLOG);

        % select coords of those pairs
        sourcePositions_xy=coords_xy(is,:);
        targetPositions_xy=coords_xy(js,:);
        Z=[repmat(-1,size(sourcePositions_xy(:,1)')); repmat(-1,size(sourcePositions_xy(:,1)'))];

        % draw the lines
	for i = 1:size(sourcePositions_xy, 1)
		line([sourcePositions_xy(i,1)'; targetPositions_xy(i,1)'],[sourcePositions_xy(i,2)'; targetPositions_xy(i,2)'],Z,'Color',lc_utm(:,i),'LineWidth',thickness_pts);
	end%for:i
    end
end
