%  constructModelRDMs is a function which parses the User/modelRDMs.m file and
%  saves a struct which is readable by the body of Niko's functions.
%
% [RDMs =] constructModelRDMs(rawModels, userOptions)
%
%        rawModels --- Naturally specified models.
%                A struct in which rawModels.(modelName) is the model RDM.
%
%        userOptions --- The options struct.
%                userOptions.analysisName
%                        A string which is prepended to the saved files.
%                userOptions.rootPath
%                        A string describing the root path where files will be
%                        saved (inside created directories).
%                userOptions.ModelColor
%                        A triple indicating the [R G B] value of the colour
%                        which should be used to indicated model RDMs on various
%                        diagrams. Defaults to black ([0 0 0]).
%
% The following files are saved by this function.
%        userOptions.rootPath/RDMs/
%                userOptions.analysisName_Models.mat
%                        Contains a structure of model RDMs, one for each of the
%                        ones from rawModels, but in the form preferred by the
%                        toolbox.
%  
%  Cai Wingfield 11-2009, updated by Li Su 3-2012

function [varargout] = constructModelRDMs(userOptions)

returnHere = pwd;

ModelsFilename = [userOptions.analysisName, '_Models.mat'];

promptOptions.functionCaller = 'constructModelRDMs';
promptOptions.defaultResponse = 'S';
promptOptions.checkFiles(1).address = fullfile(userOptions.rootPath, 'RDMs', ModelsFilename);

overwriteFlag = overwritePrompt(userOptions, promptOptions);

if overwriteFlag

    fprintf('Constructing models based on modelRDMs.m...\n');
    rawModels = modelRDMs();
    % Parse
    modelNames = fields(rawModels);
    nModelNames = numel(modelNames);

    % Reconstruct
    for model = 1:nModelNames
        Models(1,model).name = underscoresToSpaces(modelNames{model});
        Models(1,model).RDM = rawModels.(modelNames{model});
        Models(1,model).color = userOptions.ModelColor;
        
    end%for
    
    % And save
    fprintf(['Saving Model RDMs to ' fullfile(userOptions.rootPath, 'RDMs', ModelsFilename) '\n']);
    gotoDir(userOptions.rootPath, 'RDMs');
    save(ModelsFilename, 'Models');
    
    if nModelNames>1
        disp('Correlating Models to see their dependency: ')
        pairwiseCorrelateRDMs({Models},userOptions);
    end
else
    fprintf('Loading saved modelRDMs...\n');
    load(promptOptions.checkFiles(1).address);
end

if nargout == 1
	varargout{1} = Models;
elseif nargout > 0
	error('0 or 1 arguments out, please.');
end%if:nargout

cd(returnHere);
