% [realDs bootstrapEs pairwisePs] = ...
%                        bootstrapRDMComparisons(bootstrapableReferenceRDMs, ...
%                                                testRDMs, ...
%                                                userOptions ...
%                                                )
%
%        bootstrapableReferenceRDMs --- The RDMs to bootstrap.
%                bootstrapableReferenceRDMs should be a [nConditions nConditions
%                nSubjects]-sized matrix of stacked squareform RDMs.
%
%        testRDMs --- The RDMs to test against.
%                testRDM should be an [nConditions nConditions nTestRDMs]-sized
%                matrix where each leaf is one RDM to be tested.
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
%        realDs
%                The true distances between the average of the
%                bootstrapableReferenceRDMs and the testRDM.
%
%        bootstrapStdE
%                The bootstrap standard error.
%
%        pairwisePs
%                A matrix of p-values obtained my calculating the
%                proportion of bootstrap resamplings in which the test RDMs
%                are most similar to the reference RDMs, averaged across
%                subjects.
%
% Cai Wingfield 6-2010, 7-2010

function [realDs, bootstrapEs, pairwisePs] = bootstrapRDMComparisons(bootstrapableReferenceRDMs, testRDMs, userOptions)

	% Sort out defaults
	userOptions = setIfUnset(userOptions, 'nResamplings', 1000);
	userOptions = setIfUnset(userOptions, 'resampleSubjects', false);
	userOptions = setIfUnset(userOptions, 'resampleConditions', true);
	userOptions = setIfUnset(userOptions, 'distanceMeasure', 'Spearman');
	
	% Constants
	nConditions = size(bootstrapableReferenceRDMs, 1);
	nSubjects = size(bootstrapableReferenceRDMs, 3);
	nDots = 50; % For the display output!
	
	nTestRDMs = size(testRDMs, 3);
	
	if ~(size(bootstrapableReferenceRDMs, 1) == size(testRDMs, 1))
		error('bootstrapRDMComparison:DifferentSizedRDMs', 'Two RDMs being compared are of different sizes. This is incompatible\nwith bootstrap methods!');
	end%if
	
	averageReferenceRDM = sum(bootstrapableReferenceRDMs, 3) ./ nSubjects;
	
	% Decide what to say
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
	
	% Come up with the random samples (with replacement)
	if userOptions.resampleSubjects
		resampledSubjectIs = ceil(nSubjects * rand(userOptions.nResamplings, nSubjects));
	else
		resampledSubjectIs = repmat(1:nSubjects, userOptions.nResamplings, nSubjects);
	end%if:resampleSubjects
	
	if userOptions.resampleConditions
		resampledConditionIs = ceil(nConditions * rand(userOptions.nResamplings, nConditions));
	else
		resampledConditionIs = repmat(1:nConditions, userOptions.nResamplings, nConditions);
	end%if:resampleConditions

	% Preallocation
	realDs = nan(nTestRDMs, 1);
	bootstrapDs = nan(nTestRDMs, userOptions.nResamplings);
	bootstrapEs = nan(nTestRDMs, 1);
	pairwisePs = nan(nTestRDMs, nTestRDMs);
	
	% Bootstrap
	n = 0;
	for test = 1:nTestRDMs
		for b = 1:userOptions.nResamplings
		
			n = n + 1;
		
			localReferenceRDMs = bootstrapableReferenceRDMs(resampledConditionIs(b,:),resampledConditionIs(b,:),resampledSubjectIs(b,:));
			localTestRDM = testRDMs(resampledConditionIs(b,:), resampledConditionIs(b,:), test);
			
			averageBootstrappedRDM = mean(localReferenceRDMs, 3);
			
			bootstrapDs(test, b) = pdist([squareform(averageBootstrappedRDM); squareform(localTestRDM)], userOptions.distanceMeasure);
			
			if mod(n, ceil(userOptions.nResamplings*nTestRDMs/nDots)) == 0, fprintf('.'); end%if:print some dots
			
		end%for:b
	end%for:test
		
	bootstrapEs = std(bootstrapDs, 0, 2);
	
	for testi = 1:nTestRDMs
		realDs(testi) = pdist([squareform(averageReferenceRDM); squareform(testRDMs(:,:,testi))], userOptions.distanceMeasure);
		for testj = 1:nTestRDMs
			if testi == testj
				pairwisePs(testi, testj) = nan;
			else
				ijDifferences = bootstrapDs(testi, :) - bootstrapDs(testj, :);
				pairwisePs(testi, testj) = (numel(find(ijDifferences <= 0)) / userOptions.nResamplings);
			end%if:diagonal
		end%for:testj
		fprintf('.');
	end%for:testi
	
	t = toc;%1
	
	fprintf([': [' num2str(ceil(t)) 's]\n']);

end%function:bootstrapRDMComparisons
