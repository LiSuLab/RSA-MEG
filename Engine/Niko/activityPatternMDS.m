%  Edited by Cai Wingfield 11-2009 [options.randomOrderAmongstEquals]

function activityPatternMDS(RDM,options)

%% define defaults
options=setIfUnset(options,'figI_catCols', 1000);      
options=setIfUnset(options,'figI_images', 1001);
options=setIfUnset(options,'randomOrderAmongstEquals', false);

jetCols=jet;

options=setIfUnset(options,'MDScriterion','metricstress');
%criterion='stress'; % very good (distribution less even than with metricstress)
%criterion='sstress'; % pushes everything toward a circular fringe
%criterion='metricstress'; % very good (more even distribution than with stress)
%criterion='metricsstress'; % pushes everything toward a circular fringe
%criterion='sammon'; % looks like metricstress (i.e. good)
%criterion='strain'; % worse

if options.randomOrderAmongstEquals
	RDMs = rankTransformRDMs(RDMs,'randomOrderAmongEquals', true);
end%if

if isstruct(RDM)
    RDMName=RDM.name
else
    RDMName='[unnamed dissimilarity matrix]';
end
description{1}=['\fontsize{12}MDS(',options.MDScriterion,'): ',RDMName];


%% perform multidimensional scaling (MDS)
D = unwrapRDMs(squareRDMs(RDM));

nDims=2;

try
    [pats_mds_2D, stress, disparities] = mdscale(D, nDims,'criterion',options.MDScriterion);
catch
    try
        [pats_mds_2D, stress, disparities] = mdscale(D, nDims,'criterion','stress');
        description{1}=['\fontsize{12}MDS(reverted to stress, ',options.MDScriterion,' failed): ',RDMName];
    catch
        try
            D2=D+0.2;
            D2(logical(eye(length(D))))=0;
            [pats_mds_2D, stress, disparities] = mdscale(D2, nDims,'criterion',options.MDScriterion);
            description{1}=['\fontsize{12}MDS(added .2 to dissims to avoid colocalization)',RDMName];
        catch
            description{1}=['\fontsize{12}MDS failed:',RDMName];
        end
    end   
end    
% Y = mdscale(D, p) performs non-metric multidimensional scaling on
% the n-by-n dissimilarity matrix D, and returns Y, a configuration
% of n points (rows) in p dimensions (columns). The Euclidean
% distances between points in Y approximate a monotonic
% transformation of the corresponding dissimilarities in D. By
% default, mdscale uses Kruskal's normalized stress1
% criterion. mdscale treats NaNs in D as missing values, and
% ignores those elements.
%
% [Y, stress] = mdscale(D, p) returns the minimized stress, i.e.,
% the stress evaluated at Y.
% [Y, stress, disparities] = mdscale(D, p) returns the disparities,
% that is, the monotonic transformation of the dissimilarities D.
if isfield(options,'figI_shepardPlots')&&~isempty(options.figI_shepardPlots)
    pageFigure(options.figI_shepardPlots);  clf;
    shepardPlot(D,disparities,pdist(pats_mds_2D),figI_shepardPlots,['MDS(',description{1},')']);
end


%% plot MDS arrangement with category-color coding
if isfield(options,'contrasts')&&~isempty(options.contrasts)
    nCategories=size(options.contrasts,2);
    options=setIfUnset(options,'categoryColors',jetCols(round(linspace(1,64,nCategories)),:));
    %options=setIfUnset(options,'categoryColors',randomColor(nCategories));

    defaultLabels=cell(nCategories,1);
    for i=1:nCategories
        defaultLabels(i)={num2str(i)};
    end
    options=setIfUnset(options,'categoryLabels',defaultLabels);

    pageFigure(options.figI_catCols(1)); [hf,ha]=selectPlot(options.figI_catCols);

    if ~isfield(options,'categoryIs')
        options.categoryIs=1:size(options.contrasts,2);
    end
    
    if exist('pats_mds_2D','var')
        plotDots(pats_mds_2D,options.contrasts(:,options.categoryIs),options.categoryColors(options.categoryIs,:),options.categoryLabels(options.categoryIs));
        axis tight equal off; %title({'\fontsize{12}MDS(',options.MDScriterion,')'],description{1}});
        if isstruct(RDM), title(['\bf',deunderscore(RDM.name)]); end
    end
end


%% plot stimulus images (or condition icons) in MDS arrangement
if isfield(options,'imageData')&&~isempty(options.imageData)

    pageFigure(options.figI_images(1)); [hf,ha]=selectPlot(options.figI_images);

    if ~isfield(options,'rubberbandGraphPlot')
        if size(D,2)<10
            options.rubberbandGraphPlot=1;
        else
            options.rubberbandGraphPlot=0;
        end
    end
    
    if options.rubberbandGraphPlot
        rubberbandGraphPlot(pats_mds_2D,D);
    end

    drawImageArrangement(options.imageData,pats_mds_2D);
    axis tight equal off;
    %title({'\fontsize{14}multidim. scaling\fontsize{11}',['(',options.MDScriterion,')'],description{1}});

end


%% label and export the last figure plotted to
if isfield(options,'postscriptAppendFilespec')||isfield(options,'postscriptOverwriteFilespec')||isfield(options,'pdfFilespec')
    labelAndExportFig(gcf,description,options);
    % POLICY CHANGE: these functions don't automatically export their figures.
end

