function distanceBarRDMs(referenceRDMsCell, comparedRDMsCell, userOptions, localOptions)

% distanceBarRDMs( ...
%                 {referenceRDMs[, referenceRDMs2, ...]}, ...
%                 {comparedRDMs[, comparedRDMs2, ...]}, ...
%                 userOptions, ...
%                 [localOptions] ...
%                )
%
% Draws a distance bar graph for RDMs.
%
%        referenceRDMs, referenceRDMs2, ... --- Structs of RDMs.
%                One bar graph will be drawn for each of structs in this cell.
%                Each struct in this cell must be either two-dimensional, or
%                three-dimensional, or have exactly one entry. In the case where
%                it's two- or three-dimensional, the first dimension (which may
%                be singleton) is for different masks, the second is for
%                subjects, and the third (if it exists) is for sessions (as per
%                convention). In the case where there's exactly one RDM,
%                subjects will not be bootstrapped. This is because the error
%                bars may be formed by bootstrapping the subject set (according
%                to preferences).  If not resampling subjects, the RDMs in each
%                struct will be averaged together. See the description for
%                userOptions.resampleSubjects for help.
%
%        comparedRDMs, comparedRDMs2, ... ---- Structs of RDMs.
%                All RDMs in here will be concatenated and compared to the RDMs
%                in the referenceRDMss.
%
%        userOptions --- The options struct.
%                userOptions.analysisName
%                        A string which is prepended to the saved files.
%                userOptions.rootPath
%                        A string describing the root path where files will be
%                        saved (inside created directories).
%                userOptions.saveFigurePDF
%                        A boolean value. If true, the figure is saved as a PDF.
%                        Defaults to false.
%                userOptions.saveFigurePS
%                        A boolean value. If true, the figure is saved as a PS.
%                        Defaults to false.
%                userOptions.saveFigureFig
%                        A boolean value. If true, the figure is saved as a
%                        MATLAB .fig file. Defaults to false.
%                userOptions.displayFigures
%                        A boolean value. If true, the figure remains open after
%                        it is created. Defaults to true.
%                userOptions.distanceMeasure
%                        A string descriptive of the distance measure to be used
%                        to compare two RDMs. Defaults to 'Spearman'.
%                userOptions.nResamplings
%                        How many bootstrap resamplings shoule be performed?
%                        Defaults to 1000.
%                userOptions.resampleSubjects
%                        Boolean. If true, subjects will be bootstrap resampled.
%                        Defaults to false.
%                userOptions.resampleConditions
%                        Boolean. If true, conditions will be resampled.
%                        Defaults to true.
%
%        localOptions --- Further options.
%                localOptions.titleString
%                        A string which can override the automatic string used
%                        to title the distance bar graph.
%                localOptions.pIndication
%                        Formats the p-value labels on the x-axis in one of the
%                        following ways:
%                                '='
%                                '<'
%                                '*'
%                        Defuaults to '*'.
%                localOptions.pairwiseSignificanceThreshold
%                        Defaults to 0.05.
%
% Cai Wingfield 5-2010, 6-2010, 7-2010

returnHere = pwd;

%% Set defaults and check options struct
if nargin == 3, localOptions = struct(); end%if:nargin
if ~isfield(userOptions, 'analysisName'), error('distanceBarRDMs:NoAnalysisName', 'analysisName must be set. See help'); end%if
if ~isfield(userOptions, 'rootPath'), error('distanceBarRDMs:NoRootPath', 'rootPath must be set. See help'); end%if
userOptions = setIfUnset(userOptions, 'saveFigurePDF', false);
userOptions = setIfUnset(userOptions, 'saveFigurePS', false);
userOptions = setIfUnset(userOptions, 'saveFigureFig', false);
userOptions = setIfUnset(userOptions, 'displayFigures', true);
userOptions = setIfUnset(userOptions, 'distanceMeasure', 'Spearman');
userOptions = setIfUnset(userOptions, 'nResamplings', 1000);
userOptions = setIfUnset(userOptions, 'resampleSubjects', false);
userOptions = setIfUnset(userOptions, 'resampleConditions', true);
localOptions = setIfUnset(localOptions, 'pIndication', '*');
localOptions = setIfUnset(localOptions, 'pairwiseSignificanceThreshold', 0.05);

nReferenceRDMStructs = numel(referenceRDMsCell);
nComparedRDMStructs = numel(comparedRDMsCell);

%% De-cell and concatenate testRDMs

% For compared ones
for RDMStructI = 1:nComparedRDMStructs
	thisRDMStruct = comparedRDMsCell{RDMStructI};
	if RDMStructI == 1
		comparedRDMs = thisRDMStruct;
	else
		comparedRDMs = concatenateRDMs(comparedRDMs, thisRDMStruct);
	end%if
end%for:RDMStructI

nComparedRDMs = numel(comparedRDMs);

% Split up RoIs for each of the referenceRDMs
oldReferenceRDMsCell = referenceRDMsCell;
clear referenceRDMsCell;
overallReferenceI = 1;
for referenceI = 1:nReferenceRDMStructs
	thisRDMStruct = oldReferenceRDMsCell{referenceI};
	nRoIsHere = size(thisRDMStruct, 1);
	for RoII = 1:nRoIsHere
		referenceRDMsCell{overallReferenceI} = thisRDMStruct(RoII, :, :);
		overallReferenceI = overallReferenceI + 1;
	end%for:RoII
end%for:referenceI
clear oldReferenceRDMsCell;

nReferenceRDMStructs = numel(referenceRDMsCell);

% Split up the subjects for each of the referenceRDMs
oldReferenceRDMsCell = referenceRDMsCell;
clear referenceRDMsCell;
overallReferenceI = 1;
for referenceI = 1:nReferenceRDMStructs
	thisRDMStruct = oldReferenceRDMsCell{referenceI};
	nSessionsHere = size(thisRDMStruct, 3);
	for sessionI = 1:nSessionsHere
		referenceRDMsCell{overallReferenceI} = thisRDMStruct(:, :, sessionI);
		overallReferenceI = overallReferenceI + 1;
	end%for:RoII
end%for:referenceI
clear oldReferenceRDMsCell;

nReferenceRDMStructs = numel(referenceRDMsCell);

% Some preferences

figureNumberBase = 1000000*floor(100*rand);

cd(fullfile(userOptions.rootPath, 'Figures'));

for referenceRDMi = 1:nReferenceRDMStructs

	thisFigureNumber = figureNumberBase + referenceRDMi;
	thisReferenceRDMs = referenceRDMsCell{referenceRDMi};
	
	if size(thisReferenceRDMs, 2) == 1 && numel(thisReferenceRDMs) == 1% If just one reference RDM...
		RDMName = thisReferenceRDMs.name;
		if userOptions.resampleSubjects
			warning('distanceBarRDMs:ResampleSubjectsWithOneSubject', ['(!) You have specified to resample subjects, but "' RDMName '" doesn''t contain\n    multiple subjects. Subjects will not be resampled, conditions will be.']);
			userOptions.resampleSubjects = false;
			userOptions.resampleConditions = true;
		end%if
	else % if more than one RDM in this reference
		thisAverageReferenceRDMs = averageRDMs_subjectSession(thisReferenceRDMs, 'subject');
		RDMName = thisAverageReferenceRDMs.name;
	end%if
	thisReferenceStack = unwrapRDMs(thisReferenceRDMs);

	barOptions.fileName = ['distance-from-' RDMName];
	barOptions.figI = thisFigureNumber;
	if isfield(localOptions, 'titleString')
		barOptions.titleString = localOptions.titleString;
	else
		barOptions.titleString = ['Distance from ' RDMName];
	end%if
	barOptions.referenceName = RDMName;
	barOptions.pIndication = localOptions.pIndication;
	barOptions.pairwiseSignificanceThreshold = localOptions.pairwiseSignificanceThreshold;
	
	fprintf(['Drawing bar graph of distances from "' RDMName '"...\n        "' barOptions.titleString '" [figure ' num2str(thisFigureNumber) ']\n']);

	%figureRDMRelationships(theseRDMs, userOptions, barOptions);
	figureDistanceBarGraph(thisReferenceStack, comparedRDMs, userOptions, barOptions);

end%for:referenceRDMi

cd(returnHere);
