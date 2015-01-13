% [realD, bootstrapStdE] = bootstrapRDMComparison(bootstrapableReferenceRDMs,
%                                                testRDM,
%                                                userOptions)
%
%        bootstrapableReferenceRDMs --- The RDMs to bootstrap.
%                bootstrapableReferenceRDMs should be a [nConditions nConditions nSubjects]-sized matrix of stacked squareform RDMs.
%
%        testRDM --- The RDM to test against.
%                testRDM should be an [nConditions nConditions]-sized matrix.
%
%        userOptions --- The options struct.
%                userOptions.distanceMeasure
%                        A string descriptive of the distance measure to be
%                        used. Defaults to 'Spearman'.
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
%        realD
%                The true distance between the average of the
%                bootstrapableReferenceRDMs and the testRDM.
%
%        bootstrapStdE
%                The bootstrap standard error.
%
% Cai Wingfield 6-2010

function [realD, bootstrapStdE] = bootstrapRDMComparison(bootstrapableReferenceRDMs, testRDM, userOptions)

	% Sort out defaults
	userOptions = setIfUnset(userOptions, 'nResamplings', 1000);
	userOptions = setIfUnset(userOptions, 'resampleSubjects', false);
	userOptions = setIfUnset(userOptions, 'resampleConditions', true);
	userOptions = setIfUnset(userOptions, 'distanceMeasure', 'Spearman');
	
	% Constants
	nConditions = size(bootstrapableReferenceRDMs, 1);
	nSubjects = size(bootstrapableReferenceRDMs, 3);
	nDots = 50;
	
	averageReferenceRDM = sum(bootstrapableReferenceRDMs, 3) ./ nSubjects;
	
	dVector = nan(1, userOptions.nResamplings);
	
	if userOptions.resampleSubjects
		if userOptions.resampleConditions
			message = 'subjects and conditions';
		else
			message = 'subjects';
		end%if
	else
		if userOptions.resampleConditions
			message = 'conditions';
		else
			message = 'nothing';
			warning('(!) You''ve gotta resample something, else the bar graph won''t mean anything!');
		end%if
	end%if
	
	fprintf(['Resampling ' message ' ' num2str(userOptions.nResamplings) ' times']);
	
	tic; %1

	for b = 1:userOptions.nResamplings
	
		localReferenceRDMs = bootstrapableReferenceRDMs;
		localTestRDM = testRDM;
	
		% Sort out subjects
		if userOptions.resampleSubjects
			resampledSubjectIndices = resampleWithReplacement(nSubjects);
			localReferenceRDMs = localReferenceRDMs(:,:,resampledSubjectIndices);
		end%if:resampleSubjects
		
		% Sort out conditions
		if userOptions.resampleConditions
			if ~(size(localReferenceRDMs, 1) == size(localTestRDM, 1))
				error('bootstrapRDMComparison:DifferentSizedRDMs', 'Two RDMs being compared are of different sizes, so conditions cannot be boostrapped.');
			end%if
			resampledConditionIndices = resampleWithReplacement(nConditions);
			localReferenceRDMs = localReferenceRDMs(resampledConditionIndices,resampledConditionIndices,:);
			localTestRDM = localTestRDM(resampledConditionIndices,resampledConditionIndices);
		end%if:resampleConditions
		
		averageBootstrappedRDM = sum(localReferenceRDMs, 3) ./ nSubjects;
		
		dVector(b) = pdist([squareform(averageBootstrappedRDM); squareform(localTestRDM)], userOptions.distanceMeasure);
		
		if mod(b, ceil(userOptions.nResamplings/nDots)) == 0, fprintf('.'); end%if:print some dots
		
	end%for:b
	
	t = toc;%1
	
	fprintf([': [' num2str(ceil(t)) 's]\n']);
	
	dVector = dVector(~isnan(dVector));
	
	bootstrapStdE = std(dVector);
	
	realD = pdist([squareform(averageReferenceRDM); squareform(testRDM)], userOptions.distanceMeasure);

end%function:bootstrapRDMComparison

%%%%%%%%%%%%%%%%%%%
%% Subfunctions: %%
%%%%%%%%%%%%%%%%%%%

% indicesOut = resampleWithReplacement(n)
% indicesOut = resampleWithReplacement(n,m)
function indicesOut = resampleWithReplacement(varargin)
	switch nargin
		case 1
			n = varargin{1};
			m = n;
		case 2
			n = varargin{1};
			m = varargin{2};
		otherwise
			error('resampleWithReplacement:wrongNargin', 'Only 1 or 2 arguments, thanks.');
	end%switch:nargin
	
	indicesOut = zeros(1, m);
	for i = 1:m
		allIntegers = randomPermutation(n);
		indicesOut(i) = allIntegers(1);
	end%for:i
end%function:resampleWithReplacement
