% Recipe_fMRI
%
% Cai Wingfield 5-2010, 6-2010, 7-2010, 8-2010

%%%%%%%%%%%%%%%%%%%%
%% Initialisation %%
%%%%%%%%%%%%%%%%%%%%

toolboxRoot = '/home/cw04/RSA/svn/devel/toolbox/'; addpath(genpath(toolboxRoot));
userOptions = projectOptions();

%%%%%%%%%%%%%%%%%%%%%%
%% Data preparation %%
%%%%%%%%%%%%%%%%%%%%%%

fullBrainVols = fMRIDataPreparation('SPM', userOptions);
binaryMasks_nS = fMRIMaskPreparation(userOptions);
maskedBrains = fMRIDataMasking(fullBrainVols, binaryMasks_nS, 'SPM', userOptions);

%%%%%%%%%%%%%%%%%%%%%
%% RDM calculation %%
%%%%%%%%%%%%%%%%%%%%%

RDMs = constructRDMs(maskedBrains, 'SPM', userOptions);
sRDMs = averageRDMs_subjectSession(RDMs, 'session');
RDMs = averageRDMs_subjectSession(RDMs, 'session', 'subject');

Models = constructModelRDMs(modelRDMs(), userOptions);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% First-order visualisation %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figureInterleavedRDMs(RDMs, userOptions, struct('fileName', 'RoIRDMs', 'figureNumber', 1));
figureInterleavedRDMs(Models, userOptions, struct('fileName', 'ModelRDMs', 'figureNumber', 2));

MDSConditions(RDMs, userOptions);
dendrogramConditions(RDMs, userOptions);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Second-order analysis %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

pairwiseCorrelateRDMs({RDMs, Models}, userOptions);
MDSRDMs({RDMs, Models}, userOptions);
distanceBarRDMs({sRDMs}, {Models}, userOptions);

testSignificance({RDMs}, {Models}, userOptions);