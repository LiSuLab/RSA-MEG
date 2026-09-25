function tests=test_inference
tests=functiontests(localfunctions);
end

function testExactMaximumAndRng(testCase)
x=[1 1;2 -2;3 3];
state=rng;
o=rsa_meeg.inference.group(x,'Exact',true,'Tail','two-sided');
verifyEqual(testCase,rng,state);
verifyEqual(testCase,o.n_permutations,8);
verifyEqual(testCase,o.p_fwer(1),sum(o.null_max>=abs(o.statistic(1)))/8,'AbsTol',1e-12);
verifyGreaterThanOrEqual(testCase,min(o.p_fwer),1/8);
end

function testSharedFlipsAndMonteCarlo(testCase)
x=[1 1;2 2;3 3;4 4];
state=rng;
o=rsa_meeg.inference.group(x,'NumPermutations',19,'Seed',7);
verifyEqual(testCase,rng,state);
verifyEqual(testCase,o.p_fwer(1),o.p_fwer(2));
verifyGreaterThanOrEqual(testCase,min(o.p_fwer),1/20);
end

function testNegativeClusterAndEmptyNull(testCase)
adj=logical([0 1 0;1 0 0;0 0 0]);
x=[-4 -4 0;-5 -5 0;-6 -6 0;-7 -7 0];
o=rsa_meeg.inference.group(x,'Method','cluster-mass','Tail','two-sided',...
    'Adjacency',adj,'ClusterThreshold',1,'Exact',true);
verifyLessThan(testCase,o.cluster_labels(1),0);
verifyEqual(testCase,o.cluster_labels(1),o.cluster_labels(2));
verifyEqual(testCase,o.p_fwer(3),1);
empty=rsa_meeg.inference.group(zeros(4,3),'Method','cluster-mass',...
    'Adjacency',adj,'ClusterThreshold',1,'Exact',true);
verifyEqual(testCase,empty.null_max,zeros(16,1));
verifyEqual(testCase,empty.p_fwer,ones(1,3));
end

function testSignedTfce(testCase)
adj=logical([0 1;1 0]);
x=[-3 -3;-4 -4;-5 -5;-6 -6];
o=rsa_meeg.inference.group(x,'Method','tfce','Tail','two-sided','Adjacency',adj,'Exact',true);
verifyGreaterThan(testCase,min(o.enhanced_statistic),0);
verifyEqual(testCase,o.enhanced_statistic(1),o.enhanced_statistic(2));
end

function testBhEmptyAndNoRejections(testCase)
o=rsa_meeg.inference.bh([.2 .3 .9],.05);
verifyFalse(testCase,any(o.reject)); verifyEmpty(testCase,o.threshold);
e=rsa_meeg.inference.bh(zeros(0,1),.05);
verifyEmpty(testCase,e.reject);
end

function testConditionPermutationExactAndSharedLabels(testCase)
R=struct(); R.values=cat(3,reshape([1;2;3],3,1,1),reshape([3;2;1],3,1,1));
R.pairs=nchoosek(1:3,2); R.condition_ids={'A','B','C'}; R.center_ids=1;
R.windows=struct('center_s',[.1 .2]);
o=rsa_meeg.inference.condition_permutation(R,[1;2;3],...
    'Correlation','pearson','Exact',true);
verifyEqual(testCase,o.n_permutations,6);
verifyEqual(testCase,o.statistic(:),[1;-1],'AbsTol',1e-12);
verifyEqual(testCase,o.p_fwer(:),[1/3;1/3],'AbsTol',1e-12);
verifyEqual(testCase,sum(abs(o.null_max-1)<1e-12),2); % shared relabeling preserves the paired maps
end

function testConditionPermutationMonteCarloRng(testCase)
R=struct('values',reshape([1;2;3],3,1,1),'pairs',nchoosek(1:3,2),...
    'condition_ids',{{'A','B','C'}},'center_ids',1,'windows',struct('center_s',0));
prior=rng;
o=rsa_meeg.inference.condition_permutation(R,[1;2;3],...
    'Correlation','pearson','NumPermutations',19,'Seed',17,'Tail','negative');
verifyEqual(testCase,rng,prior);
verifyGreaterThanOrEqual(testCase,o.p_fwer,1/20);
verifyEqual(testCase,o.p_fwer,1); % positive observed correlation has negative-tail score zero
end

function testInvalidInferenceOptions(testCase)
x=[1 2;2 1;3 2];
verifyError(testCase,@()rsa_meeg.inference.group(x,'NumPermutations',Inf),...
    'rsa_meeg:inference:Permutations');
verifyError(testCase,@()rsa_meeg.inference.group(x,'Method','cluster-mass',...
    'Adjacency',[0 -1;-1 0],'ClusterThreshold',1),...
    'rsa_meeg:inference:Adjacency');
end
