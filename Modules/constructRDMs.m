% constructRDMs is a function which takes a matrix of imaging data and computes
% RDMs from it.  One RDM for each mask region, subject and scanning session.
% Depending on the passed options, these may or may not be averaged over
% subjects or sessions. RDMs will end up appropriately named and in a struct
% which is then saved.
%
% [RDMs =] constructRDMs(maskedBrains, betaCorrespondence, userOptions)
%
%        maskedBrains --- The input response patterns.
%                Contains the masked brains in a struct such that
%                maskedBrains.(mask).(subject) is a [nMaskedVoxels nConditions
%                nSessions] matrix.
%
%        betaCorrespondence --- The array of beta filenames.
%                betas(condition, session).identifier is a string which referrs
%                to the filename (not including path) of the SPM beta image.
%                Alternatively, this can be the string 'SPM', in which case the
%                SPM metadata will be used to infer this information, provided
%                that userOptions.conditionLabels is set, and the condition
%                labels are the same as those used in SPM.
%
%        userOptions --- The options struct.
%                userOptions.analysisName
%                        A string which is prepended to the saved files.
%                userOptions.rootPath
%                        A string describing the root path where files will be
%                        saved (inside created directories).
%                userOptions.maskNames
%                        A cell array containing strings identifying the mask
%                        names. Defaults to the fieldnames of the first subject
%                        of maskedBrains.
%                userOptions.subjectNames
%                        A cell array containing strings identifying the subject
%                        names. Defaults to the fieldnames in (the first mask
%                        of) maskedBrains.
%                userOptions.distance
%                        A string indicating the distance measure with which to
%                        calculate the RDMs. Defaults to 'Correlation'.
%                userOptions.RoIColor
%                        A triple indicating the [R G B] value of the colour
%                        which should be used to indicated RoI RDMs on various
%                        diagrams. Defaults to black ([0 0 0]).
%
% The following files are saved by this function:
%        userOptions.rootPath/RDMs/
%                userOptions.analysisName_RDMs.mat
%                        Contains a structure of RoI RDMs which is of size
%                        [nMasks, nSubjects, nSessions] and with fields:
%                                RDM
%                                name
%                                color
%        userOptions.rootPath/Details/
%                userOptions.analysisName_constructRDMs_Details.mat
%                        Contains the userOptions for this execution of the
%                        function and a timestamp.
%
% Cai Wingfield 11-2009, 12-2009, 3-2010, 6-2010
% Update: Isma Zulfiqar, Added regularized searchlight pattern option 11-12 

function [varargout] = constructRDMs(maskedBrains, betaCorrespondence, userOptions)

returnHere = pwd; % We'll come back here later

%% Set defaults and check options struct
if ~isfield(userOptions, 'analysisName'), error('constructRDMs:NoAnalysisName', 'analysisName must be set. See help'); end%if
if ~isfield(userOptions, 'rootPath'), error('constructRDMs:NoRootPath', 'rootPath must be set. See help'); end%if
%userOptions = setIfUnset(userOptions, 'maskNames', fieldnames(maskedBrains));
%userOptions = setIfUnset(userOptions, 'subjectNames', fieldnames(maskedBrains.(userOptions.maskNames{1})));
userOptions = setIfUnset(userOptions, 'distance', 'Correlation');
userOptions = setIfUnset(userOptions, 'RoIColor', [0 0 0]);

% The analysisName will be used to label the files which are eventually saved.
RDMsFilename = [userOptions.analysisName, '_RDMs.mat'];
DetailsFilename = [userOptions.analysisName, '_constructRDMs_Details.mat'];

promptOptions.functionCaller = 'constructRDMs';
promptOptions.defaultResponse = 'S';
promptOptions.checkFiles(1).address = fullfile(userOptions.rootPath, 'RDMs', RDMsFilename);
promptOptions.checkFiles(2).address = fullfile(userOptions.rootPath, 'Details', DetailsFilename);

if userOptions.slidingTimeWindow , overwriteFlag=1;
else overwriteFlag = overwritePrompt(userOptions, promptOptions); end

if overwriteFlag
	
	%% Get Data

	if ischar(betaCorrespondence) && strcmpi(betaCorrespondence, 'SPM')
		betas = getDataFromSPM(userOptions);
	else
		betas = betaCorrespondence;
	end%if:SPM
	
	nSubjects = numel(userOptions.subjectNames);
	nSessions = size(betas, 1);
	nConditions = size(betas, 2);
    
    if userOptions.sensorLevelAnalysis
        nMasks = 1;
        disp('in sensor space!');
        maskNames = userOptions.maskSpec.maskName;
    else
        nMasks = numel(userOptions.maskNames);
        disp('in source space!');
        maskNames = userOptions.maskNames;
    end
    
    fprintf('Constructing RDMs for ROIs:');  
    
	for mask = 1:nMasks % For each mask...

		thisMask = dashToUnderscores(maskNames{mask});         
        
		for subject = 1:nSubjects % and for each subject...

			% Figure out which subject this is
			thisSubject = userOptions.subjectNames{subject};
        
          if ~userOptions.regularized || userOptions.sensorLevelAnalysis

			for session = 1:nSessions % and each session...

				% Get the brain scan vol
				thisActivityPattern = maskedBrains.(thisMask).(thisSubject);

				% Calculate the RDM               
				localRDM = squareform( ...
					pdist( ...
						squeeze(thisActivityPattern(:, :, session))', userOptions.distance));
               
				% Store the RDM in a struct with the right names and things!
				RDMs(mask, subject, session).RDM = localRDM;
				RDMs(mask, subject, session).name = [deunderscore(thisMask) ' | ' thisSubject ' | Session: ' num2str(session)];
				RDMs(mask, subject, session).color = userOptions.RoIColor;

				clear localRDM;

			end%for:session
          else
            thisActivityPattern = maskedBrains.(thisMask).(thisSubject); 
            r_matrix = g_matrix(zscore(squeeze(thisActivityPattern))',nConditions, nSessions);
            localRDM = (1 - r_matrix);
            if isnan(localRDM) % sessions and conditions should be optimal
              error('Cannot calculate g-matrix. Try reducing number of conditions');
            end
             
              session=1;
              % Store the RDM in a struct with the right names and things!
              RDMs(mask, subject, session).RDM = localRDM;
              RDMs(mask, subject, session).name = [deunderscore(thisMask) '-regularised | ' thisSubject ' | all'];
              RDMs(mask, subject, session).color = userOptions.RoIColor;
              
              clear localRDM;
          end % if userOptions.regularized
             
		end%for:subject
        fprintf('\b.:');
	end%for:mask
    fprintf('\bDone.\n');

	%% Save relevant info

	timeStamp = datestr(now);

	fprintf(['Saving RDMs to ' fullfile(userOptions.rootPath, 'RDMs', RDMsFilename) '\n']);
	gotoDir(userOptions.rootPath, 'RDMs');
	save(RDMsFilename, 'RDMs');
	
	fprintf(['Saving Details to ' fullfile(userOptions.rootPath, 'Details', DetailsFilename) '\n']);
	gotoDir(userOptions.rootPath, 'Details');
	save(DetailsFilename, 'timeStamp', 'userOptions');
	
else
	fprintf(['Loading previously saved RDMs from ' fullfile(userOptions.rootPath, 'RDMs', RDMsFilename) '...\n']);
	load(fullfile(userOptions.rootPath, 'RDMs', RDMsFilename));
end%if

if nargout == 1
	varargout{1} = RDMs;
elseif nargout > 0
	error('0 or 1 arguments out, please.');
end%if:nargout

cd(returnHere); % And go back to where you started
