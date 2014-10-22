function showMDSarrangement(RDM,options)

% FUNCTION
%       draws a multidimensional scaling (MDS) arrangement reflecting the
%       dissimilarity structure of the items whose
%       representational dissimilarity matrix is passed in argument RDM.
%       the function can draw the MDS result as an arrangement of text
%       labels (default), colored dots, or icons (e.g. the experimental
%       stimuli or, more generally, icons denoting the experimental
%       conditions). which ones of these visualizations are produced is
%       controlled by setting the fields of the options argument.
%
% USAGE
%       showMDSarrangement(RDM[,options])
%
% ARGUMENTS
% RDM   
%       the items' dissimilarity matrix as a struct RDM,
%       or in square-matrix or upper-triangular-vector format.
%
% [options]
%       optional struct whose fields control which visualizations are
%       produced and provide the required additional information. the
%       fields (like the struct as a whole) are optional. if options is not
%       passed or none of the fields are set, the MDS arrangement will be
%       drawn using text labels, where each label is a number indexing an
%       activity pattern in the order defined by the RDM.
%
%       the the key fields are:
%       [figI_textLabels] figure index for visualization as text-label
%                       arrangement. defaults to 1.
%       [figI_catCols]  figure index for visualization as colored-dot
%                       arrangement, where the colors code for category.
%                       this visualization is omitted if this argument is
%                       missing.
%       [figI_icons]    figure index for visualization as an icon
%                       arrangement. this visualization is omitted if this 
%                       argument is missing.
%       [figI_shepardPlots] figure index for shepard plot (a scatterplot
%                       relating the dissimilarities in the original space
%                       to the distances in the 2S MDS arrangement. 
%       please note: if the figI_... arguments are quadruples instead of
%                       single figure indices, then the last three numbers
%                       specify the subplot. see selectPlot.m for details.
%       [MDScriterion]  the cost function minimized by the MDS. the default
%                       value is 'metricstress'. see documentation of
%                       mdscale.m for details.
%       [rubberbandGraphPlot] boolean argument. if this is set to 'true',
%                       then a rubberband graph is plotted in the
%                       background to visualize the distortions incurred by
%                       the dimensionality reduction (for details, see
%                       kriegeskorte et al. 2008).
%       [postscriptAppendFilespec] file specification (i.e. [path,filename])
%                       for figure output in postscript format. the figures
%                       will be appended to the postscript file if the file
%                       already exists.
%       [postscriptOverwriteFilespec] file specification (i.e. [path,filename])
%                       for figure output in postscript format. the file
%                       will be overwritten if it already exists.
%       [pdfFilespec]   file specification (i.e. [path,filename]) for 
%                       figure output in pdf format. 
%
%       further fields needed for visualization as category-color-coded
%       arrangement of either dots or text labels:
%       [contrasts]     a matrix, whose columns define categories of 
%                       items as index vectors (column height
%                       == number of items, 1 indicates
%                       present, 0 indicates absent). (these category
%                       definitions are a special case of a general-linear-
%                       model contrast -- hence the name of this argument.)
%       [categoryIs]    list of intergers referring to columns of argument 
%                       'contrasts' and thereby selecting which categories 
%                       of items are to be included in the 
%                       visualization.
%       [categoryColors]nCategories-by-3 matrix, whose rows define the
%                       colors as RGB triples. there is one row per
%                       category.
%       [categoryLabels] text labels for the categories. if these are
%                       provided a legend will show what the color-coded
%                       categories are.
%
%       further field needed for visualization as icon arrangement:
%       [icons]         the icon images for visualization as an icon
%                       arrangement.

%
% TIP
%       use TEST_showMDSarrangement.m to try this function out.

%% define defaults
if ~exist('options','var')||isempty(options), options=struct; end
options=setIfUnset(options,'figI_textLabels',1);      
options=setIfUnset(options,'textLabels',[]);      
jetCols=jet;
options=setIfUnset(options,'MDScriterion','metricstress');
%criterion='stress'; % very good (distribution less even than with metricstress)
%criterion='sstress'; % pushes everything toward a circular fringe
%criterion='metricstress'; % very good (more even distribution than with stress)
%criterion='metricsstress'; % pushes everything toward a circular fringe
%criterion='sammon'; % looks like metricstress (i.e. good)
%criterion='strain'; % worse

if isstruct(RDM)
    RDMname=RDM.name
else
    RDMname='[unnamed dissimilarity matrix]';
end
description{1}=['\fontsize{12}MDS(',options.MDScriterion,'): ',RDMname];

figIs=[];

%% perform multidimensional scaling (MDS)
%D = unwrapSimmats(squareSimmats(RDM));
D = unwrapRDMs(squareRDMs(RDM));

nDims=2;

try
    [pats_mds_2D, stress, disparities] = mdscale(D, nDims,'criterion',options.MDScriterion);
catch
    try
        [pats_mds_2D, stress, disparities] = mdscale(D, nDims,'criterion','stress');
        description{1}=['\fontsize{12}MDS(reverted to stress, ',options.MDScriterion,' failed): ',RDMname];
    catch
        try
            D2=D+0.2;
            D2(logical(eye(length(D))))=0;
            [pats_mds_2D, stress, disparities] = mdscale(D2, nDims,'criterion',options.MDScriterion);
            description{1}=['\fontsize{12}MDS(added .2 to dissims to avoid colocalization)',RDMname];
        catch
            description{1}=['\fontsize{12}MDS failed:',RDMname];
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
    figIs=[figIs,pageFigure(options.figI_shepardPlots)];  clf;
    shepardPlot(D,disparities,pdist(pats_mds_2D),figI_shepardPlots,['MDS(',description{1},')']);
end


%% plot MDS arrangement using text labels
if isfield(options,'figI_textLabels')&&~isempty(options.figI_textLabels)
    %if isfield(options,'icons')&&~isempty(options.icons)

    figIs=[figIs,pageFigure(options.figI_textLabels(1))]; [hf,ha]=selectPlot(options.figI_textLabels);

    % plot rubberband plot in the background
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

    if ~exist('contrasts','var')
        % categories undefined: plot all text labels in black
        plotTextLabels(pats_mds_2D,options.textLabels);
        axis equal off;
    else
        % categories defined: plot text labels in category colors
        nCategories=size(options.contrasts,2);
        options=setIfUnset(options,'categoryColors',jetCols(round(linspace(1,64,nCategories)),:));
        %options=setIfUnset(options,'categoryColors',randomColor(nCategories));

        defaultLabels=cell(nCategories,1);
        for i=1:nCategories
            defaultLabels(i)={num2str(i)};
        end
        options=setIfUnset(options,'categoryLabels',defaultLabels);

        figIs=[figIs,pageFigure(options.figI_catCols(1))]; [hf,ha]=selectPlot(options.figI_catCols);

        if ~isfield(options,'categoryIs')
            options.categoryIs=1:size(options.contrasts,2);
        end

        plotTextLabels(pats_mds_2D,options.textLabels,[],options.contrasts(:,options.categoryIs),options.categoryColors(options.categoryIs,:),options.categoryLabels(options.categoryIs));
        axis equal off; %title({'\fontsize{12}MDS(',options.MDScriterion,')'],description{1}});
        if isstruct(RDM), title(['\bf',deunderscore(RDM.name)]); end
    end

    axis([min(pats_mds_2D(:,1)) max(pats_mds_2D(:,1)) min(pats_mds_2D(:,2)) max(pats_mds_2D(:,2))]);
end


%% plot MDS arrangement using category-color-coded dots
if isfield(options,'figI_catCols')&&~isempty(options.figI_catCols)
    nCategories=size(options.contrasts,2);
    options=setIfUnset(options,'categoryColors',jetCols(round(linspace(1,64,nCategories)),:));
    %options=setIfUnset(options,'categoryColors',randomColor(nCategories));

    defaultLabels=cell(nCategories,1);
    for i=1:nCategories
        defaultLabels(i)={num2str(i)};
    end
    options=setIfUnset(options,'categoryLabels',defaultLabels);

    figIs=[figIs,pageFigure(options.figI_catCols(1))]; [hf,ha]=selectPlot(options.figI_catCols);

    if ~isfield(options,'categoryIs')
        options.categoryIs=1:size(options.contrasts,2);
    end
    
    if exist('pats_mds_2D','var')
        plotDots(pats_mds_2D,options.contrasts(:,options.categoryIs),options.categoryColors(options.categoryIs,:),options.categoryLabels(options.categoryIs));
        axis tight equal off; %title({'\fontsize{12}MDS(',options.MDScriterion,')'],description{1}});
        if isstruct(RDM), title(['\bf',deunderscore(RDM.name)]); end
    end
end


%% plot MDS arrangement using icons
if isfield(options,'figI_icons')&&~isempty(options.figI_icons)
%if isfield(options,'icons')&&~isempty(options.icons)

    figIs=[figIs,pageFigure(options.figI_icons(1))]; [hf,ha]=selectPlot(options.figI_icons);

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

    drawImageArrangement(options.icons,pats_mds_2D);
    axis tight equal off;
    %title({'\fontsize{14}multidim. scaling\fontsize{11}',['(',options.MDScriterion,')'],description{1}});

end


%% label and export the last figure plotted to
if isfield(options,'postscriptAppendFilespec')||isfield(options,'postscriptOverwriteFilespec')||isfield(options,'pdfFilespec')
    if ~isempty(figIs)
        for figIsI=1:numel(figIs);
            labelAndExportFig(figIs(figIsI),description,options);
        end
    end
end

