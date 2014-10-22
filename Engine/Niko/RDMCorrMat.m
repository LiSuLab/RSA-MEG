function corrMat=RDMCorrMat(RDMs,figPlotSpec,type)
% USAGE
% corrMat=RDMCorrMat(RDMs[,figPlotSpec])
%
% FUNCTION
% returns and optionally displays the correlation matrix (spearman) of a set
% of RDMs (can be square or upper triangle form and wrapped or bare).
%
% 08-2011 CW: slight modifications for how nans are handled

if ~exist('type','var'),type='Spearman'; end; % Spearman rank should be default?

RDMs_bareVecs=unwrapRDMs(vectorizeRDMs(RDMs));%reduceRDMsToValidConditionSet(RDMs))); % This is of size [1 utv nRDMs]

[one,nRDMParams,nRDMs]=size(RDMs_bareVecs);

RDMs_cols=permute(RDMs_bareVecs,[2 3 1]); % This is of size [utv nRDMs (1)]

% For each pair of RDMs, ignore missing data only for this pair of RDMs
% (unlike just using corr, which would ignore it if ANY RDM had missing
% data at this point).
%corrMat=corrcoef(RDMs_cols)
for RDMI1 = 1:nRDMs
	for RDMI2 = 1 : nRDMs
		corrMat(RDMI1,RDMI2)= 1 - corr(RDMs_cols(:,RDMI1), RDMs_cols(:,RDMI2), 'type', type, 'rows', 'pairwise');
    end
    if RDMI1==RDMI2
        corrMat(RDMI1,RDMI2) = 0;
    end
end

% for RDMI1 = 1:nRDMs
% 	corrMat(RDMI1,RDMI1) = 0; % make the diagonal artificially zero
% end
displayRDM.RDM = corrMat;
displayRDM.name = ['\bfDistance matrix (',type,')'];

if exist('figPlotSpec','var')
    %selectPlot(figPlotSpec);
    showRDMs(displayRDM);
    axis square off;
    
	if isstruct(RDMs)
		for RDMI=1:nRDMs
			text(RDMI,RDMI,RDMs(RDMI).name,'HorizontalAlignment','center','FontWeight','bold','Color',RDMs(RDMI).color,'FontUnits','normalized','FontSize',1/max(nRDMs,30));
		end
	end

end