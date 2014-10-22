function showRDMRelationships(RDMs,options)
% uses multidimensional scaling to simultaneously relate all RDMs passed
% in struct RDMs in terms of their similarity (i.e. second-order
% similarity). additionally visualizes the dissimilarity of each RDM to
% the first one (by default) or the one specified as the reference RDM
% in the options. 

% Edited by Cai Wingfield 27-10-2009

%% preparations
RDMs=squareRDMs(RDMs);
[n,n]=size(squareRDM(RDMs(1).RDM));

if ~exist('options','var'), options.rankTransform=true; end

if ~isfield(options,'rankTransform'), options.rankTransform=true; end
if ~isfield(options,'distanceMeasure'), options.distanceMeasure='euclidean'; end % alternatives: 'correlation','spearman','seuclidean','mahalanobis'
if ~isfield(options,'isomap'), options.isomap=false; end
if ~isfield(options,'nmMDS'), options.nmMDS=true; end
if ~isfield(options,'rubberbands'), options.rubberbands=true; end
if ~isfield(options,'barGraph'), options.barGraph=false; end
if ~isfield(options,'barGraphWithBootStrapErrorBars'), options.barGraphWithBootStrapErrorBars=true; end
if ~isfield(options,'figI'), options.figI=[3600 2 1 1; 3600 2 1 2]; end
if ~isfield(options,'conditionSetIndexVec')||(isfield(options,'conditionSetIndexVec')&&isempty(options.conditionSetIndexVec)), options.conditionSetIndexVec=ones(1,n); end
if ~isfield(options,'criterion'), options.criterion='metricstress'; end
if ~isfield(options,'referenceRDMI'), options.referenceRDMI=1; end

options.conditionSetIndexVec=options.conditionSetIndexVec(:)'; % convert to row

distanceMeasureNote='';

nRDMs=size(RDMs,2);
figIsI=1; 

%RDMCorrMat(RDMs,1);


%% grant index 1 to the reference RDM
RDM_temp=RDMs(1);
RDMs(1)=RDMs(options.referenceRDMI);
RDMs(options.referenceRDMI)=RDM_temp;

%% reduce RDMs to rows and cols defined (non-nan) in all of them
[RDMs,validConditionsLOG]=reduceRDMsToValidConditionSet(RDMs);
conditionSetIs_vector=options.conditionSetIndexVec;
conditionSetIs_vector=conditionSetIs_vector(validConditionsLOG);

RDMCorrMat(RDMs,1,'Spearman'); % This is just for display!


%% prepare for bootstrapping with reduced conditions sets
[RDMMask,condSet1_LOG,condSet2_LOG,nCondSets,nCond1,nCond2]=convertToRDMMask(conditionSetIs_vector);
RDMMask=tril(RDMMask,-1); % retain only upper triangular part (rest set to zero)
%show(RDMMask);

%% optional rank transform and RDM vectorization (RDMMask)
% showRDMs(RDMs,2,0);
RDM_rowvecs=nan(nRDMs,sum(RDMMask(:)));
for RDMI=1:nRDMs
    cRDM_maskContents=RDMs(RDMI).RDM(RDMMask);
    if options.rankTransform
        scale01=true;
        cRDM_maskContents=rankTransform_randomOrderAmongEquals(cRDM_maskContents,scale01);
    end
    RDMs(RDMI).RDM(RDMMask)=cRDM_maskContents; % copy back into RDMs
    RDM_rowvecs(RDMI,:)=cRDM_maskContents(:);        %  ...and RDM_rowvecs
end
% showRDMs(RDMs,2,0);
% showRDMs(RDMs,2,1);
%RDMCorrMat(RDMs,4);
%show(corrcoef(RDM_rowvecs'));

if options.rankTransform
    rankTransformString='rank-transf. (rand. among eq.)';
else
    rankTransformString='non-rank-transf.';
end

%% compute dissimilarities
try
    D=pdist(RDM_rowvecs,options.distanceMeasure); % compute pairwise distances
catch
    D=pdist(RDM_rowvecs,'euclidean'); % compute pairwise distances
    distanceMeasureNote=[' (',options.distanceMeasure,' dist. FAILED)'];
    options.distanceMeasure='euclidean';
end

% alternatives for options.distanceMeasure: 'correlation', 'euclidean', 'seuclidean', 'mahalanobis', 'spearman', 'cosine', 'cityblock', 'hamming' and others
D=squareRDM(D);

%show(D);

%% isomap
if options.isomap
    % plots similarity matrices as points in an isomap representation of
    % similarity structure of a set of similarity matrices (2nd-order similarity).
    isomap_options.dims=[1:2];
    isomap_options.display=1;
    [Y, R, E] = isomap_nk(D,'k',6,isomap_options); %[Y, R, E] = isomap(D, n_fcn, n_size, options);
    %[Y, R, E] = isomap(D,'epsilon',2); %[Y, R, E] = isomap(D, n_fcn, n_size, options);

    % plot the isomap arrangement
    coords=Y.coords{2}'; % 2 dimensional coords of isomap arrangement

    selectPlot(options.figI(figIsI,:)); cla; figIsI=figIsI+1;
    if options.rubberbands
        rubberbandGraphPlot(coords,D);
        %         rubberbandGraphPlot(coords,D,'color');
    end
    
    downShift=0;
    for RDMI=1:nRDMs
        plot(coords(RDMI,1),coords(RDMI,2),'o','MarkerFaceColor',RDMs(RDMI).color,'MarkerEdgeColor','none','MarkerSize',8); hold on;
        text(coords(RDMI,1),coords(RDMI,2)-downShift,{'',RDMs(RDMI).name},'Color',RDMs(RDMI).color,'HorizontalAlignment','Center','FontSize',12,'FontName','Arial');
    end
    axis equal tight; zoomOut(10);
    set(gca,'xtick',[],'ytick',[]);
    %xlabel('isomap of RDM space');
    title({'\fontsize{12}isomap\fontsize{9}', deunderscore(['(',options.distanceMeasure,' dist.',distanceMeasureNote,', ',rankTransformString,' RDMs)'])});

    shepardPlot(D,[],pdist(coords),986,['isomap(RDMs)']);
    
end % isomap

%% nonmetric multidimensional scaling
if options.nmMDS
    % plots similarity matrices as points in an nmMDS representation of
    % similarity structure of a set of similarity matrices (2nd-order similarity).
    nDims=2;
    
    selectPlot(options.figI(figIsI,:)); cla; figIsI=figIsI+1;
    
    %try
        [coords, stress, disparities] = mdscale(D, nDims,'criterion',options.criterion);

        % plot the mds arrangement
        if options.rubberbands
            rubberbandGraphPlot(coords,D);
            %         rubberbandGraphPlot(coords,D,'color');
        end

        downShift=0;
        for RDMI=1:nRDMs
            plot(coords(RDMI,1),coords(RDMI,2),'o','MarkerFaceColor',RDMs(RDMI).color,'MarkerEdgeColor','none','MarkerSize',8); hold on;
	    %text(coords(RDMI,1),coords(RDMI,2)-downShift,{'',deunderscore(RDMs(RDMI).name)},'Color',RDMs(RDMI).color,'HorizontalAlignment','Center','FontSize',12,'FontName','Arial'); % text.m wasn't accepting all these inputs, so...
	    text(coords(RDMI,1),coords(RDMI,2)-downShift,deunderscore(RDMs(RDMI).name));
        end
        axis equal tight; %zoomOut(10); % Don't have zoomOut.m
        set(gca,'xtick',[],'ytick',[]);
        %xlabel('nmMDS of RDM space');
        title({['\fontsize{12}\bfdissimilarity-matrix MDS\rm (',options.criterion,')'],['\fontsize{9}(',options.distanceMeasure,' dist.',distanceMeasureNote,', ',rankTransformString,' RDMs)']});
        axis off;
        
        shepardPlot(D,disparities,pdist(coords),987,['non-metric MDS(RDMs)']);
    %catch
    %    title('MDS failed.')
    %end
end % nmMDS

%% horizontal bar graph of the distances of each RDM from the reference RDM
% (by default the first RDM receives this treatment)
if options.barGraph
    selectPlot(options.figI(figIsI,:)); figIsI=figIsI+1;
    distsToData=D(1,1:end);
    [sortedDistsToData,sortedIndices]=sort(distsToData);

    hb=barh(sortedDistsToData(end:-1:2));
    set(gca,'YTick',[1:nRDMs-1]);
    set(gca,'YTickLabel',{RDMs(sortedIndices(end:-1:2)).name},'FontUnits','normalized','FontSize',1/nRDMs);

    if strcmp(options.distanceMeasure,'euclidean')
        xlabel(['euclidean dist.',distanceMeasureNote,' from ',RDMs(1).name,' in ',rankTransformString,' RDM space']);
    elseif strcmp(options.distanceMeasure,'spearman')
        xlabel(['(1 - Spearman''s rank corr.) dist.',distanceMeasureNote,' from ',RDMs(1).name,' in ',rankTransformString,' RDM space']);
    else
        xlabel([options.distanceMeasure,' dist.',distanceMeasureNote,' from ',RDMs(1).name,' in ',rankTransformString,' RDM space']);
    end

    axis tight;
end % bar graph

%% bar graph with bootstrap error bars of the distances of each RDM from the first
if options.barGraphWithBootStrapErrorBars

    % preparations
    if strcmp(options.distanceMeasure,'euclidean')
        titleString=['euclidean dist.',distanceMeasureNote,' from \bf',RDMs(1).name,'\rm in ',rankTransformString,' RDM space'];
    elseif strcmp(options.distanceMeasure,'spearman')
        titleString=['(1 - Spearman''s rank corr.) dist.',distanceMeasureNote,' from \bf',RDMs(1).name,'\rm in ',rankTransformString,' RDM space'];
    else
        titleString=[options.distanceMeasure,' dist.',distanceMeasureNote,' from \bf',RDMs(1).name,'\rm in ',rankTransformString,' RDM space'];
    end
    titleString=deunderscore(titleString);
    
    
    % bootstrap resample the similarity matrices
    nBootstrapResamplings=100;

    resampledRDMs_utv=condSetBootstrapOfRDMs_condSetRestriction(RDMs,nBootstrapResamplings,conditionSetIs_vector);
    % uses bootstrap resampling of the conditions set to resample a set of RDMs.
    % the resampled RDMs are returned in upper triangular form (rows), stacked
    % along the 3rd (index of input RDM) and 4th (resampling index)
    % dimensions (for compatibility with square RDMs).
    
    % compute second-order similarity matrix for each bootstrap resampling
    bootstrapDistsToRefRDM=nan(nBootstrapResamplings,nRDMs-1);
    
    for boostrapResamplingI=1:nBootstrapResamplings
         cResampling_RDMs_utv=resampledRDMs_utv(:,:,:,boostrapResamplingI);
         cResampling_RDMs_utv(:,isnan(mean(cResampling_RDMs_utv,3)),:)=[];
         
         %          showRDMs(cResampling_RDMs_utv(:,:,[1,5]));
         %          RDMCorrMat(cResampling_RDMs_utv(:,:,[1,5]),334);

         % line'em up along the first dimension to please pdist
         RDMs_rowvecs=permute(cResampling_RDMs_utv,[3 2 1]);
                  
         cResamplingD=squareRDM(pdist(RDMs_rowvecs,options.distanceMeasure)); % compute pairwise distances

         % DEBUG: 
         %          show([cResamplingD,D],1) % TEST PASSED
         %          cResamplingD(1,5)
         %          D(1,5)
         %          pause
         
         bootstrapDistsToRefRDM(boostrapResamplingI,:)=cResamplingD(1,2:end); % bootstrap distances to the reference RDM (RDM #1)
         % alternatives for options.distanceMeasure: 'correlation', 'euclidean', 'seuclidean', 'mahalanobis', 'spearman', 'cosine', 'cityblock', 'hamming' and others
    end
    
    % sort by the means (bootstrapped: bs)
    [sortedMeanDistsToData_bs,sortedIndices_bs]=sort(mean(bootstrapDistsToRefRDM,1));
    sortedBootstrapDistsToRefRDM=bootstrapDistsToRefRDM(:,sortedIndices_bs);
    
    
    
    % plot bar graph with bootstrap standard-error bars
    col=[0 0 0];
    barNOTplot=true;
    [mean_Y,se_mean_Y]=plotMeanWithStandardDeviationBars(1:nRDMs-1,sortedBootstrapDistsToRefRDM,options.figI(figIsI,:),titleString,col,barNOTplot);
    %figIsI=figIsI+1;

    % for original RDMs (not bootstrapped: nbs)
    distsToData_nbs=D(1,2:end);
    %[sortedDistsToData_nbs,sortedIndices_nbs]=sort(distsToData_nbs);
    plot(1:nRDMs-1,distsToData_nbs(sortedIndices_bs),'.','Color',[.8 .8 .8]);
    %sortedBootstrapDistsToRefRDM=bootstrapDistsToRefRDM(:,sortedIndices_nbs);

    
    set(gca,'XTick',[]);
    
    for RDMI=1:nRDMs-1
        col=RDMs(sortedIndices_bs(RDMI)+1).color;
        text(RDMI,0,['\bf',deunderscore(RDMs(sortedIndices_bs(RDMI)+1).name)],'Rotation',90,'Color',col);
    end

    axis_Xmin(0); axis_Xmax(nRDMs);

end % bar graph with bootstrap error bars