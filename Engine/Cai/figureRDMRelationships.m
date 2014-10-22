% figureRDMRelationships is a function originally authored by Niko which
% uses multidimensional scaling to simultaneously relate all RDMs passed
% in struct RDMs in terms of their similarity (i.e. second-order
% similarity). additionally visualizes the dissimilarity of each RDM to
% the first one (by default) or the one specified as the reference RDM
% in the localOptions.
%
% Cai's new version adds the ability to save and close figures as they are generated.
%
%        RDMs --- A struct of RDMs.
%
%        userOptions --- The options struct.
%                userOptions.analysisName
%                        A string which is prepended to the saved files.
%                userOptions.rootPath
%                        A string describing the root path where files will be
%                        saved (inside created directories).
%                userOptions.saveFigurePDF
%                        A boolean value. If true, the figure is saved as a PDF.
%                userOptions.saveFigurePS
%                        A boolean value. If true, the figure is saved as a PS.
%                userOptions.saveFigureFig
%                        A boolean value. If true, the figure is saved as a
%                        MATLAB .fig file.
%                userOptions.displayFigures
%                        A boolean value. If true, the figure remains open after
%                        it is created.
%
%        localOptions --- Further options.
%
% Edited by Cai Wingfield 10-2009
% Edited by Cai Wingfield 11-2009
% Edited by Cai Wingfield 3-2010

function figureRDMRelationships(RDMs, userOptions, localOptions)

%% preparations

% warning('off', 'IgnoringExtraEntries');

RDMs=squareRDMs(RDMs);
[n,n]=size(squareRDM(RDMs(1).RDM));

if ~exist('localOptions','var'), localOptions.rankTransform=true; end

if ~isfield(localOptions,'rankTransform'), localOptions.rankTransform=true; end
if ~isfield(localOptions,'distanceMeasure'), localOptions.distanceMeasure='euclidean'; end % alternatives: 'correlation','spearman','seuclidean','mahalanobis'
if ~isfield(localOptions,'isomap'), localOptions.isomap=false; end
if ~isfield(localOptions,'nmMDS'), localOptions.nmMDS=true; end
if ~isfield(localOptions,'rubberbands'), localOptions.rubberbands=true; end
if ~isfield(localOptions,'barGraph'), localOptions.barGraph=false; end
if ~isfield(localOptions,'barGraphWithBootstrapErrorBars'), localOptions.barGraphWithBootstrapErrorBars=false; end
if ~isfield(localOptions,'figI'), localOptions.figI=[3600 2 1 1; 3600 2 1 2]; end
if ~isfield(localOptions,'figI_shepardPlot'), localOptions.figI_shepardPlot=localOptions.figI+1; end
if ~isfield(localOptions,'conditionSetIndexVec')||(isfield(localOptions,'conditionSetIndexVec')&&isempty(localOptions.conditionSetIndexVec)), localOptions.conditionSetIndexVec=ones(1,n); end
if ~isfield(localOptions,'criterion'), localOptions.criterion='metricstress'; end
if ~isfield(localOptions,'referenceRDMI'), localOptions.referenceRDMI=1; end

if ~isfield(localOptions, 'fileName'), localOptions.fileName = 'unnamedFile!'; end

if isfield(localOptions, 'appendFlag'), appendFlag = localOptions.appendFlag; else, appendFlag = 0; end

localOptions.conditionSetIndexVec=localOptions.conditionSetIndexVec(:)'; % convert to row

fileName = [userOptions.analysisName '_' localOptions.fileName];

distanceMeasureNote='';

nRDMs=size(RDMs,2);
figIsI=1; 

%RDMCorrMat(RDMs,1);


%% grant index 1 to the reference RDM
RDM_temp=RDMs(1);
RDMs(1)=RDMs(localOptions.referenceRDMI);
RDMs(localOptions.referenceRDMI)=RDM_temp;

%% reduce RDMs to rows and cols defined (non-nan) in all of them
[RDMs,validConditionsLOG]=reduceRDMsToValidConditionSet(RDMs);
conditionSetIs_vector=localOptions.conditionSetIndexVec;
conditionSetIs_vector=conditionSetIs_vector(validConditionsLOG);

%% prepare for bootstrapping with reduced conditions sets
[RDMMask,condSet1_LOG,condSet2_LOG,nCondSets,nCond1,nCond2]=convertToRDMMask(conditionSetIs_vector);
RDMMask=tril(RDMMask,-1); % retain only upper triangular part (rest set to zero)
%show(RDMMask);

%% optional rank transform and RDM vectorization (RDMMask)
% showRDMs(RDMs,2,0);
RDM_rowvecs=nan(nRDMs,sum(RDMMask(:)));
for RDMI=1:nRDMs
    cRDM_maskContents=RDMs(RDMI).RDM(RDMMask);
    if localOptions.rankTransform
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

if localOptions.rankTransform
    rankTransformString='rank-transf. (rand. among eq.)';
else
    rankTransformString='non-rank-transf.';
end

%% compute dissimilarities
try
    D=pdist(RDM_rowvecs,localOptions.distanceMeasure); % compute pairwise distances
catch
    D=pdist(RDM_rowvecs,'euclidean'); % compute pairwise distances
    distanceMeasureNote=[' (',localOptions.distanceMeasure,' dist. FAILED)'];
    localOptions.distanceMeasure='euclidean';
end

% alternatives for localOptions.distanceMeasure: 'correlation', 'euclidean', 'seuclidean', 'mahalanobis', 'spearman', 'cosine', 'cityblock', 'hamming' and others
D=squareRDM(D);

%show(D);

%  %% isomap
%  if localOptions.isomap
%      % plots similarity matrices as points in an isomap representation of
%      % similarity structure of a set of similarity matrices (2nd-order similarity).
%      isomap_localOptions.dims=[1:2];
%      isomap_localOptions.display=1;
%      [Y, R, E] = isomap_nk(D,'k',6,isomap_localOptions); %[Y, R, E] = isomap(D, n_fcn, n_size, localOptions);
%      %[Y, R, E] = isomap(D,'epsilon',2); %[Y, R, E] = isomap(D, n_fcn, n_size, localOptions);
%  
%      % plot the isomap arrangement
%      coords=Y.coords{2}'; % 2 dimensional coords of isomap arrangement
%  
%      selectPlot(localOptions.figI(figIsI,:)); cla; figIsI=figIsI+1;
%      if localOptions.rubberbands
%          rubberbandGraphPlot(coords,D);
%          %         rubberbandGraphPlot(coords,D,'color');
%      end
%      
%      downShift=0;
%      for RDMI=1:nRDMs
%          plot(coords(RDMI,1),coords(RDMI,2),'o','MarkerFaceColor',RDMs(RDMI).color,'MarkerEdgeColor','none','MarkerSize',8); hold on;
%          text(coords(RDMI,1),coords(RDMI,2)-downShift,{'',RDMs(RDMI).name},'Color',RDMs(RDMI).color,'HorizontalAlignment','Center','FontSize',12,'FontName','Arial');
%      end
%      axis equal tight; zoomOut(10);
%      set(gca,'xtick',[],'ytick',[]);
%      %xlabel('isomap of RDM space');
%      title({'\fontsize{12}isomap\fontsize{9}', deunderscore(['(',localOptions.distanceMeasure,' dist.',distanceMeasureNote,', ',rankTransformString,' RDMs)'])});
%  
%      shepardPlot(D,[],pdist(coords),986,['isomap(RDMs)']);
%      
%  end % isomap

%% nonmetric multidimensional scaling
if localOptions.nmMDS
    % plots similarity matrices as points in an nmMDS representation of
    % similarity structure of a set of similarity matrices (2nd-order similarity).
    nDims=2;
    
    selectPlot(localOptions.figI(figIsI,:)); cla; figIsI=figIsI+1;
    
    %try
        [coords, stress, disparities] = mdscale(D, nDims,'criterion',localOptions.criterion);

        % plot the mds arrangement
        if localOptions.rubberbands
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
        title({['\fontsize{12}\bfdissimilarity-matrix MDS\rm (',localOptions.criterion,')'],['\fontsize{9}(',localOptions.distanceMeasure,' dist.',distanceMeasureNote,', ',rankTransformString,' RDMs)']});
        axis off;

	thisFileName = [fileName '_MDS'];
	handleCurrentFigure(thisFileName, userOptions);
	clear thisFileName
        
        shepardPlot(D,disparities,pdist(coords),localOptions.figI_shepardPlot,['non-metric MDS(RDMs)']);
    %catch
    %    title('MDS failed.')
    %end
    thisFileName = [fileName '_ShepardPlot'];
    handleCurrentFigure(thisFileName, userOptions);
	clear thisFileName
end % nmMDS

%% horizontal bar graph of the distances of each RDM from the reference RDM
% (by default the first RDM receives this treatment)
if localOptions.barGraph
	selectPlot(localOptions.figI(figIsI,:)); figIsI=figIsI+1;
	distsToData=D(1,1:end);
	[sortedDistsToData,sortedIndices]=sort(distsToData);
	
	fontSize = 0.3/nRDMs;
	
	hb=barh(sortedDistsToData(end:-1:2));
	set(gca,'YTick',[1:nRDMs-1]);
	set(gca,'YTickLabel',{RDMs(sortedIndices(end:-1:2)).name},'FontUnits','normalized','FontSize',fontSize);
	
	if strcmpi(localOptions.distanceMeasure,'euclidean')
		xlabel(['euclidean dist.',distanceMeasureNote,' from ',RDMs(1).name,' in ',rankTransformString,' RDM space'], 'fontsize', 0.5/nRDMs);
	elseif strcmpi(localOptions.distanceMeasure,'spearman')
		xlabel(['(1 - Spearman''s rank corr.) dist.',distanceMeasureNote,' from "',RDMs(1).name,'" in ',rankTransformString,' RDM space'], 'fontsize', fontSize);
	else
		xlabel([localOptions.distanceMeasure,' dist.',distanceMeasureNote,' from ',RDMs(1).name,' in ',rankTransformString,' RDM space'], 'fontsize', fontSize);
	end%if
	
	if isfield(localOptions, 'titleString')
		title(['\bf' localOptions.titleString]);
	else
		title(['\bfDistance bar graph of various RDMs from "' RDMs(1).name '"']);
	end%if
	
	thisFileName = [fileName '_barGraph'];
	handleCurrentFigure(thisFileName, userOptions);
	clear thisFileName
	
	axis tight;
end % bar graph

%% bar graph with bootstrap error bars of the distances of each RDM from the first
if localOptions.barGraphWithBootstrapErrorBars

    % preparations
    if strcmp(localOptions.distanceMeasure,'euclidean')
        titleString=['euclidean dist.',distanceMeasureNote,' from \bf',RDMs(1).name,'\rm in ',rankTransformString,' RDM space'];
    elseif strcmp(localOptions.distanceMeasure,'spearman')
        titleString=['(1 - Spearman''s rank corr.) dist.',distanceMeasureNote,' from \bf',RDMs(1).name,'\rm in ',rankTransformString,' RDM space'];
    else
        titleString=[localOptions.distanceMeasure,' dist.',distanceMeasureNote,' from \bf',RDMs(1).name,'\rm in ',rankTransformString,' RDM space'];
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

		try
			cResamplingD=squareRDM(pdist(RDMs_rowvecs,localOptions.distanceMeasure)); % compute pairwise distances
		catch
			fprintf([ ...
				'Note: Using MATLAB''s "pdist" to compute bootstrap resamplings\n' ...
				'      for error bars just *FAILED*. Working around the problem\n' ...
				'      (this may take a long time).\n']);
			cResamplingD=squareRDM(pdist_workaround(RDMs_rowvecs,localOptions.distanceMeasure));
			fprintf([ ...
				'Note: Bootstrap error bars may have broken the bar graph figure.\n' ...
				'      Consider re-running this without bootstrap error bars :(\n']);
		end%try

         bootstrapDistsToRefRDM(boostrapResamplingI,:)=cResamplingD(1,2:end); % bootstrap distances to the reference RDM (RDM #1)
         % alternatives for localOptions.distanceMeasure: 'correlation', 'euclidean', 'seuclidean', 'mahalanobis', 'spearman', 'cosine', 'cityblock', 'hamming' and others
    end
    
    % sort by the means (bootstrapped: bs)
    [sortedMeanDistsToData_bs,sortedIndices_bs]=sort(mean(bootstrapDistsToRefRDM,1));
    sortedBootstrapDistsToRefRDM=bootstrapDistsToRefRDM(:,sortedIndices_bs);
    
    
    
    % plot bar graph with bootstrap standard-error bars
    col=[0 0 0];
    barNOTplot=true;
    [mean_Y,se_mean_Y]=plotMeanWithStandardDeviationBars(1:nRDMs-1,sortedBootstrapDistsToRefRDM,localOptions.figI(figIsI,:),titleString,col,barNOTplot);
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

    thisFileName = [fileName '_bootstrap'];
 	handleCurrentFigure(thisFileName, userOptions);
	clear thisFileName

end % bar graph with bootstrap error bars


% warning('on', 'IgnoringExtraEntries');
