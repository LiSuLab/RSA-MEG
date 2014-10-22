%  figureInterleavedRDMs(RDMs, [userOptions[, localOptions]])
%
%  figureInterleavedRDMs is a function which accepts a multidimensional struct of
%  RDMs, interleaves them into a 1D struct, and then shows them.
%
%        RDMs --- A struct of RDMs.
%                All RDMs in here will be concatenated and displayed.
%
%        userOptions --- The options struct.  This is optional but if not provided
%                        the function will revert to default options and nothing
%                        will be saved.
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
%                userOptions.imagelables
%                        Defaults to empty (no image labels).
%                userOptions.rankTransform
%                        Boolean value. If true, values in each RDM are
%                        separately transformed to lie uniformly in [0,1] with
%                        ranks preserved. If false, true values are represented.
%                        Defaults to true. This is for display only.
%                userOptions.colourScheme
%                        A colour scheme for the RDMs. Defualts to jet(64).
%                        
%        localOptions --- Further options.
%                localOptions.figureNumber
%                        If specified, this will set the figure number of the
%                        produced figure. Otherwise the figure number will be
%                        randomly generated (and probably large).
%                localOptions.fileName
%
% May save figures according to preferences.
%
%  Cai Wingfield 11-2009, 6-2010

function figureInterleavedRDMs(RDMs, userOptions, localOptions)

returnHere = pwd;

%% Set defaults and check options struct
if (nargin == 1)
	userOptionsProvided = false;
	userOptions = struct();
	localOptions = struct();
	
else
	userOptionsProvided = true;
	if (nargin == 2)
		localOptions = struct();
	end%if:nargin2
end%if:nargin1
if (userOptionsProvided && ~isfield(userOptions, 'analysisName')), error('figureInterleavedRDMs:NoAnalysisName', 'analysisName must be set. See help'); end%if
if (userOptionsProvided && ~isfield(userOptions, 'rootPath')), error('figureInterleavedRDMs:NoRootPath', 'rootPath must be set. See help'); end%if
userOptions = setIfUnset(userOptions, 'saveFigurePDF', false);
userOptions = setIfUnset(userOptions, 'saveFigurePS', false);
userOptions = setIfUnset(userOptions, 'saveFigureFig', false);
userOptions = setIfUnset(userOptions, 'displayFigures', true);
userOptions = setIfUnset(userOptions, 'rankTransform', true);
userOptions = setIfUnset(userOptions, 'colourScheme', jet(64));

if ~isfield(userOptions,'dpi'), userOptions.dpi=300; end;
if ~isfield(userOptions,'tightInset')
    userOptions.tightInset = false;
end

appendFlag = 0; % Put this into a prompt depending on existant files?

if ~isfield(localOptions, 'figureNumber')
	localOptions.figureNumber = 1;
end
if ~isfield(localOptions, 'fileName')
	localOptions.fileName = 'RDMs';
end

RDMs = interleaveRDMs(RDMs); % Pull the RDMs into a 1-d structured array

%% Now display

if isfield(userOptions, 'imagelabels')
    imagelabels = userOptions.imagelabels;
else
    imagelabels = [];
end

if userOptions.rankTransform
	showRDMs(RDMs, localOptions.figureNumber, true, [0 1], true, [], imagelabels, userOptions.colourScheme);
else
	showRDMs(RDMs, localOptions.figureNumber, false, [], true, [], imagelabels, userOptions.colourScheme);
end%rankTransform

if userOptionsProvided

	fileName = [userOptions.analysisName '_' localOptions.fileName];
	handleCurrentFigure(fullfile(userOptions.rootPath, 'Figures',fileName), userOptions);

	cd(returnHere);
	
end%if:userOptionsProvided
