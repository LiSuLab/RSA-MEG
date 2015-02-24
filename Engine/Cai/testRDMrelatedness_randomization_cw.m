function [r, p_randCondLabels, p_conv] = testRDMrelatedness_randomization_cw(rdmA, rdmB, options)

% USAGE
%       [p_randCondLabels,r,p_randDissims,p_conv]=
%       testRDMRelatedness_rankCorr_randomization(rdmA,rdmB
%       [,options])
%
% EDITS!!!
%       1-2010:  Cai Wingfield: now accepts an options struct.
%       12-2010: Cai Wingfield: now doesn't return either conventional p
%                               values or the unused NaNs.
%       3-2010:  Cai Wingfield: now if < 8 conditions exist, all permutations
%                               are used exhaustively.
%
% FUNCTION
%       tests the null hypothesis that similarity matrices A and B are
%       unrelated. the test simulates a null distribution of correlations
%       between A and B by means of randomization of the conditions labels
%       of the similarity matrix.

if ~exist('options', 'var'), options = struct([]); end

%% preparations
% square the RDMs
rdmA=stripNsquareRDMs(rdmA);
rdmB=stripNsquareRDMs(rdmB);
[ignore,n]=size(rdmA);
[ignore,nB]=size(rdmB);
if n~=nB
    error('testRDMRelatedness_randomization: RDMs need to be of the same size.');
end

% defaults for options struct
nConditions = n;
exhaustPermutations = false;
if nConditions < 8
	allPermutations = exhaustivePermutations(nConditions);
	nRandomizations = size(allPermutations, 1);
	exhaustPermutations = true;
	warning('(!) Comparing RDMs with fewer than 8 conditions (per conditions set) will produce unrealiable results!\n  + I''ll partially compensate by using exhaustive instead of random permutations...'); %#ok<WNTAG>
elseif isfield('options', 'significanceTestPermutations')
	nRandomizations = options.significanceTestPermutations;
else
	nRandomizations = 10000;
end%if

if isfield(options, 'conditionSetIs_vector')
	conditionSetIs_vector = options.conditionSetIs_vector;
else
	conditionSetIs_vector=ones(1,n);
end%if
if isfield(options, 'corrType')
	corrType = options.corrType;
else
	corrType='Spearman';
end%if
	
conditionSetIndices=unique(conditionSetIs_vector(conditionSetIs_vector~=0)); % '0' indicates: to be excluded
nConditionSets=numel(conditionSetIndices);

%% test relatedness within a single conditions set
if nConditionSets==1;

    % reduce the RDMs
    conditionSet_LOG=(conditionSetIs_vector==conditionSetIndices(1));
    rdmA=rdmA(conditionSet_LOG,conditionSet_LOG);
    rdmB=rdmB(conditionSet_LOG,conditionSet_LOG);
    [ignore,n]=size(rdmA);
    
    % vectorize the RDMs
    rdmA_vec=vectorizeRDM(rdmA);
    rdmB_vec=vectorizeRDM(rdmB);
  
    % make space for null-distribution of correlations
    rs=nan(nRandomizations,1);
    
    % index method would require on the order of n^2*nRandomizations
    % memory, so i'll go slowly for now...
    %tic
    for randomizationI=1:nRandomizations
    
    	if exhaustPermutations
    		randomIndexSeq = allPermutations(randomizationI, :);
    	else
	        randomIndexSeq = randomPermutation(n);
    	end%if
    
        rdmA_rand_vec=vectorizeRDM(rdmA(randomIndexSeq,randomIndexSeq));
        
        rs(randomizationI)=corr(rdmA_rand_vec',rdmB_vec','type',corrType,'rows','pairwise');  % correlation types: Pearson (linear), Spearman (rank), Kendall (rank)
        % setting the 'rows' parameter to 'pairwise' makes the function ignore nans.
    end
    %toc

%% test relatedness between two disjoint conditions sets
elseif nConditionSets==2;

    % reduce the RDMs (they become nonsquare: different input and output sets)
    conditionSet1_LOG=(conditionSetIs_vector==conditionSetIndices(1));
    conditionSet2_LOG=(conditionSetIs_vector==conditionSetIndices(2));

    rdmA=rdmA(conditionSet1_LOG,conditionSet2_LOG);
    rdmB=rdmB(conditionSet1_LOG,conditionSet2_LOG);
    [n1,n2]=size(rdmA);
    
    % vectorize the RDMs (already no diagonal-including sections present)
    rdmA_vec=rdmA(:)';
    rdmB_vec=rdmB(:)';
  
    % make space for null-distribution of correlations
    rs=nan(nRandomizations,1);
    
    % index method would require on the order of n^2*nRandomizations
    % memory, so i'll go slowly for now...
    %tic
    for randomizationI=1:nRandomizations
        conditionsSet1_randomIndexSeq=randomPermutation(n1);
        conditionsSet2_randomIndexSeq=randomPermutation(n2);
        rdmA_rand=rdmA(conditionsSet1_randomIndexSeq,conditionsSet2_randomIndexSeq);
        rdmA_rand_vec=rdmA_rand(:)';
        
        rs(randomizationI)=corr(rdmA_rand_vec',rdmB_vec','type',corrType,'rows','pairwise');  % correlation types: Pearson (linear), Spearman (rank), Kendall (rank)
        % setting the 'rows' parameter to 'pairwise' makes the function ignore nans.
    end
    %toc

end


%% compute actual correlation and the corresponding conventional p value
[r, p_conv]=corr(rdmA_vec',rdmB_vec','type',corrType,'rows','pairwise');  % correlation types: Pearson (linear), Spearman (rank), Kendall (rank)
% setting the 'rows' parameter to 'pairwise' makes the function ignore nans.


%% compute p value for condition-label randomization
% this is a valid method.
p_randCondLabels = 1 - relRankIn_includeValue_lowerBound(rs,r); % conservative


%% compute p value for dissimilarity-set randomization
% valid method???
% nDissimilarities=numel(rdmA_vec);
% rs=nan(nRandomizations,1);
% 
% for randomizationI=1:nRandomizations
%     rdmA_rand_vec=rdmA_vec(randperm(nDissimilarities));
% 
%     rs(randomizationI)=corr(rdmA_rand_vec',rdmB_vec','type','Spearman','rows','pairwise');  % correlation types: Pearson (linear), Spearman (rank), Kendall (rank)
%     % setting the 'rows' parameter to 'pairwise' makes the function ignore nans.
% end
% 
% p_randDissims=1-relRankIn_includeValue_lowerBound(rs,r); % conservative

% p_randDissims=nan;

