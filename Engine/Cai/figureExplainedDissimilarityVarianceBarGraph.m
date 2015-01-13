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
%                userOptions.significanceTestPermutations
%                        An integer which describes the number of random
%                        permutations to be used to calculate significance.
%                        Defaults to 10,000.
%
%        localOptions --- Further options.
%                localOptions.referenceName
%                        The name to refer to referenceRDMStack.
%                localOptions.figI
%                        The figure number.
%                localOptions.titleString
%                        A string to overwride the default title.
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
% Files may be saved according to user preference.
%
% Cai Wingfield 6-2010


function figureExplainedDissimilarityVarianceBarGraph(referenceRDMStack, testRDMs, userOptions, localOptions)

fileName = [userOptions.analysisName '_' localOptions.fileName];

% Set defaults
localOptions = setIfUnset(localOptions, 'referenceName', '[unnamed RDM]');
localOptions = setIfUnset(localOptions, 'useStars', true);
localOptions = setIfUnset(localOptions, 'pairwiseSignificanceThreshold', 0.05);
localOptions = setIfUnset(localOptions, 'extraFigureHeight', 0);
userOptions = setIfUnset(userOptions, 'nResamplings', 1000);
userOptions = setIfUnset(userOptions, 'resampleSubjects', false);
userOptions = setIfUnset(userOptions, 'resampleConditions', true);
userOptions = setIfUnset(userOptions, 'significanceTestPermutations', 10000);

% Constants
nTestRDMs = numel(testRDMs);

fileName = [userOptions.analysisName '_' localOptions.fileName];

if isfield(localOptions, 'titleString')
	titleString = ['\fontsize{14}' localOptions.titleString];
else
	titleString = [];
end%if:titleString
titleString = {titleString; [' \fontsize{9}"' userOptions.distanceMeasure '" explained dissimilarity variances of \bf' localOptions.referenceName '\rm.']};

ds = zeros(1, nTestRDMs);
es = zeros(1, nTestRDMs);

nConditions = size(referenceRDMStack, 1);

%% Error bars

fprintf('Calculating error bars via bootstrap resampling. This may take a while...\n');

testRDMStack = zeros(nConditions, nConditions, nTestRDMs);
for test = 1:nTestRDMs
	testRDMStack(:,:,test) = testRDMs(test).RDM;
end%for:test

bootstrapWorked = true;

try
	[ds, es, pairwisePs] = bootstrapRDMComparisons(referenceRDMStack, testRDMStack, userOptions);
catch ex
	bootstrapWorked = false;
	fprintf(' Bootstrap failed! (Perhaps consider not resampling conditions?)\n');
	titleString{2} = [titleString{2} ' (Bootstrap FAILED)'];
	for test = 1:nTestRDMs
		ds(test) = pdist([squareform(mean(referenceRDMStack, 3)); squareform(testRDMs(test).RDM)], userOptions.distanceMeasure);
	end%for:test
end%try


[sortedRs, sortedIs] = sort(ds);
sortedEs = es(sortedIs);
if bootstrapWorked
	sortedPairwisePs = pairwisePs(sortedIs, sortedIs);
end%if:bootstrapWorked

%% Significance values

fprintf(['Calculating p-values via permutation tests (' num2str(userOptions.significanceTestPermutations) ' permutations). This may also take a while...\n']);

averageReferenceRDM = sum(referenceRDMStack, 3) ./ size(referenceRDMStack, 3);

for test = 1:nTestRDMs
	[~, ps(test), ~]=testRDMrelatedness_randomization_cw(averageReferenceRDM, testRDMs(test).RDM, struct('nSignificanceTestPermutations', userOptions.significanceTestPermutations));
end%for:test

sortedPs = ps(sortedIs);
pStringCell = cell(1, nTestRDMs);
for test = 1:nTestRDMs
	switch localOptions.pIndication
		case '*'
			if sortedPs(test) < 0.000001
				pStringCell{test} = '******';
			elseif sortedPs(test) < 0.00001
				pStringCell{test} = '*****';
			elseif sortedPs(test) < 0.0001
				pStringCell{test} = '****';
			elseif sortedPs(test) < 0.001
				pStringCell{test} = '***';
			elseif sortedPs(test) < 0.01
				pStringCell{test} = '**';
			elseif sortedPs(test) < 0.05
				pStringCell{test} = '*';
			else
				pStringCell{test} = ' ';
			end%if:significanceLevel
		case '<'
			if sortedPs(test) < 0.000001
				pStringCell{test} = 'p < 0.000001';
			elseif sortedPs(test) < 0.00001
				pStringCell{test} = 'p < 0.00001';
			elseif sortedPs(test) < 0.0001
				pStringCell{test} = 'p < 0.0001';
			elseif sortedPs(test) < 0.001
				pStringCell{test} = 'p < 0.001';
			elseif sortedPs(test) < 0.01
				pStringCell{test} = 'p < 0.01';
			elseif sortedPs(test) < 0.05
				pStringCell{test} = 'p < 0.05';
			else
				pStringCell{test} = ' ';
			end%if:significanceLevel
		case '='
			pStringCell{test} = ['p = ' num2str(sortedPs(test), 2)]; % 2 significant figures!
	end%switch:pIndication
end%for:test

%%
%% IS THIS RIGHT???
%%
sortedRs = (1-sortedRs).^2;
%%
%%

%% Plot

x = 1:nTestRDMs;

h = figure(localOptions.figI); clf;
set(h, 'Color', 'w');

bar(x, sortedRs, 'FaceColor', [68/225 131/225 149/225], 'EdgeColor', 'none');
hold on;
if bootstrapWorked
	errorbar(x, sortedRs, sortedEs, 'Color', [0 0 0], 'LineWidth', 2, 'LineStyle', 'none');
end%if:bootstrapWorked

%% %%
%% Plot the pairwise lines
%% %%

if bootstrapWorked

	% Adjust the height of the graph
	cYLim = get(gca, 'YLim');
	cYMax = cYLim(2);
	eachLineHeight = 0.03; % 1%
	nPairwiseLines = (nTestRDMs * (nTestRDMs - 1)) / 2;
	padding = 2;
	nYMax = ((nPairwiseLines + padding) * eachLineHeight + 1) * cYMax;
	nYLim = cYLim;
	nYLim(2) = nYMax;
	set(gca, 'YLim', nYLim);

	pairwiseI = 0;

	for i = 1:nTestRDMs
		for j = i+1:nTestRDMs
			pairwiseI = pairwiseI + 1;
			% The p value for i (the lower bar) being significantly smaller than j (the higher bar) is the p value for j being significantly higher than i. This is pairwisePs(j,i) (the lt?)
			thisPairwiseP = sortedPairwisePs(j,i);
			if thisPairwiseP < localOptions.pairwiseSignificanceThreshold / 2 % / 2 because it's 2-tailed!
				xx = [i j];
				y = cYMax;
				y = y + (cYMax * eachLineHeight * padding / 2); %padding
				y = y + (pairwiseI * eachLineHeight * cYMax); %line separation
				yy = [y y];
				line(xx, yy, 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'none', 'Color', [0 0 0]);
			end%if
		end%for:j
	end%for:i

end%if:bootstrapWorked

cYLim = get(gca, 'YLim');
cYMax = cYLim(2);
labelBase = cYMax + 0.1;
labelHeight = 0.6;
nYMax = labelBase + labelHeight + localOptions.extraFigureHeight;
nYLim = [cYLim(1), nYMax];
set(gca, 'YLim', nYLim);

for test = 1:nTestRDMs
	text(test, labelBase, ['\bf',deunderscore(testRDMs(sortedIs(test)).name)], 'Rotation', 90, 'Color', testRDMs(sortedIs(test)).color);
end

xticklabel_rotate(get(gca,'XTick'), 90, pStringCell,'Fontsize',14);

box off;

hold off;

axis_Xmin(0); axis_Xmax(nTestRDMs + 1); % +1 so we don't lose half a bar

title(titleString);

thisFileName = [fileName '_barGraph'];
handleCurrentFigure(thisFileName, userOptions);
clear thisFileName
