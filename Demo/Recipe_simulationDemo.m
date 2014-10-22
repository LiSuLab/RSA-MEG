% Recipe_simulationDemo is a script.  It is a sample recipe file which will
% simulate some fMRI data and then run through an "RoI-based" RSA pipeline for
% the simulated data
%
% Cai Wingfield 5-2010, 6-2010, 7-2010

%%%%%%%%%%%%%%%%%%%%
%% Initialisation %%
%%%%%%%%%%%%%%%%%%%%

toolboxRoot = '/home/iz01/toolbox/devel/toolbox/';
% cw04/RSA/svn/devel/toolbox/'; 
addpath(genpath(toolboxRoot));

% Generate a userOptions structure and then clone it for the two streams of data
% in this pipeline. Change only the names.
userOptions_common = projectOptions_demo();
userOptions_true = userOptions_common; userOptions_true.analysisName = [userOptions_true.analysisName 'True'];
userOptions_noisy = userOptions_common; userOptions_noisy.analysisName = [userOptions_noisy.analysisName 'Noisy'];

% Generate a simulationOptions structure.
simulationOptions = simulationOptions_demo();

%%%%%%%%%%%%%%%%
%% Simulation %%
%%%%%%%%%%%%%%%%

% Generate the SPM files for each subject containing conditions clustered
% according to preferences in the simulationOptions.
[betaCorrespondence_true betaCorrespondence_noisy] = simulateSPMFiles(userOptions_common, simulationOptions);

% Load in the 'true' fMRI data
fullBrainVols_true = fMRIDataPreparation(betaCorrespondence_true, userOptions_true);

% Load in the 'noisy' fMRI data
fullBrainVols_noisy = fMRIDataPreparation(betaCorrespondence_noisy, userOptions_noisy);

% Name the RoIs for both streams of data
RoIName = 'SimRoI';
maskedBrains_true.(['true' RoIName]) = fullBrainVols_true;
maskedBrains_noisy.(['noisy' RoIName]) = fullBrainVols_noisy;

%%%%%%%%%%
%% RDMs %%
%%%%%%%%%%

% Construct RDMs for the 'true' data. One RDM for each subject (sessions have
% not been simulated) and one for the average across subjects.
RDMs_true = constructRDMs(maskedBrains_true, betaCorrespondence_true, userOptions_true);
RDMs_true = averageRDMs_subjectSession(RDMs_true, 'session');
averageRDMs_true = averageRDMs_subjectSession(RDMs_true, 'subject');

% Do the same for the 'noisy' data.
RDMs_noisy = constructRDMs(maskedBrains_noisy, betaCorrespondence_noisy, userOptions_noisy);
RDMs_noisy = averageRDMs_subjectSession(RDMs_noisy, 'session');
averageRDMs_noisy = averageRDMs_subjectSession(RDMs_noisy, 'subject');

% Prepare the model RDMs.
RDMs_model = constructModelRDMs(userOptions_common);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%% First-order analysis %%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display the three sets of RDMs: true, noisy and model
figureInterleavedRDMs(concatenateRDMs(RDMs_true, averageRDMs_true), userOptions_true, struct('fileName', 'trueRDMs', 'figureNumber', 1));
figureInterleavedRDMs(concatenateRDMs(RDMs_noisy, averageRDMs_noisy), userOptions_noisy, struct('fileName', 'noisyRDMs', 'figureNumber', 2));
figureInterleavedRDMs(RDMs_model, userOptions_common, struct('fileName', 'modelRDMs', 'figureNumber', 3));

% Determine dendrograms for the clustering of the conditions for the two data
% streams
[blankConditionLabels{1:size(RDMs_model(1).RDM, 2)}] = deal(' ');
dendrogramConditions(averageRDMs_true, userOptions_true, struct('titleString', 'Dendrogram of conditions without simulated noise', 'useAlternativeConditionLabels', true, 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 4));
dendrogramConditions(averageRDMs_noisy, userOptions_noisy, struct('titleString', 'Dendrogram of conditions with simulated noise', 'useAlternativeConditionLabels', true, 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 5));

% Display MDS plots for the condition sets for both streams of data
MDSConditions(averageRDMs_true, userOptions_true, struct('titleString', 'MDS of conditions without simulated noise', 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 6));
MDSConditions(averageRDMs_noisy, userOptions_noisy, struct('titleString', 'MDS of conditions with simulated noise', 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 7));

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Second-order analysis %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display a second-order simmilarity matrix for the models and the true and noisy simulated pattern RDMs
pairwiseCorrelateRDMs({averageRDMs_true, averageRDMs_noisy, RDMs_model}, userOptions_common, struct('figureNumber', 8));

% Plot explained dissimilarity variance bar graphs for the true and the noisy RDMs and the models, compare the distances to the /other/ stream of data
explainedDissimilarityVarianceBarRDMs({RDMs_true}, {RDMs_model, averageRDMs_noisy}, userOptions_true, struct('titleString', 'Model distances from true RDMs', 'figureNumber', 9));
explainedDissimilarityVarianceBarRDMs({RDMs_noisy}, {RDMs_model, averageRDMs_true}, userOptions_noisy, struct('titleString', 'Model distances from noisy RDMs', 'figureNumber', 10));

% Plot all RDMs on a MDS plot to visualise pairwise distances.
MDSRDMs({averageRDMs_true, averageRDMs_noisy, RDMs_model}, userOptions_common, struct('titleString', 'MDS of noisy RDMs and models', 'figureNumber', 11));

% Calculate actual significance values and save them to a file.
testSignificance({RDMs_model}, {averageRDMs_true, averageRDMs_noisy}, userOptions_common);
