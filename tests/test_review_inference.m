function tests=test_review_inference
%TEST_REVIEW_INFERENCE Independent cross-module scientific regression tests.
tests=functiontests(localfunctions);
end

function testGroupTScaleInvariance(testCase)
% A change of physical unit cannot alter the one-sample t statistic.
x=[1 2;2 1;3 2;4 1];
a=rsa_meeg.inference.group(x,'Exact',true);
b=rsa_meeg.inference.group(x*1e-20,'Exact',true);
verifyEqual(testCase,b.statistic,a.statistic,'AbsTol',1e-10);
verifyEqual(testCase,b.p_fwer,a.p_fwer,'AbsTol',1e-12);
c=rsa_meeg.inference.group(x*1e300,'Exact',true);
verifyEqual(testCase,c.statistic,a.statistic,'AbsTol',1e-10);
verifyEqual(testCase,c.p_fwer,a.p_fwer,'AbsTol',1e-12);
end

function testExactSignSymmetryAndFamily(testCase)
x=[1 2;2 -1;3 3];
a=rsa_meeg.inference.group(x,'Exact',true,'Tail','two-sided');
b=rsa_meeg.inference.group(-x,'Exact',true,'Tail','two-sided');
verifyEqual(testCase,a.p_fwer,b.p_fwer,'AbsTol',1e-12);
verifyEqual(testCase,a.null_max,b.null_max,'AbsTol',1e-12);
verifyEqual(testCase,a.family_size,2);
verifyEqual(testCase,a.n_permutations,8);
reference=zeros(8,1);
for i=0:7
    signs=2*double(bitget(uint32(i),1:3))-1;
    flipped=x.*signs(:);
    t=mean(flipped,1)./(std(flipped,0,1)/sqrt(3));
    reference(i+1)=max(abs(t));
end
verifyEqual(testCase,sort(a.null_max),sort(reference),'AbsTol',1e-12);
end

function testSignedClustersRespectAdjacency(testCase)
% Tests 1:2 and 3:4 are separate hemisphere blocks.
x=[4 4 -4 -4;5 5 -5 -5;6 6 -6 -6;7 7 -7 -7];
A=logical([0 1 0 0;1 0 0 0;0 0 0 1;0 0 1 0]);
r=rsa_meeg.inference.group(x,'Method','cluster-mass', ...
    'Tail','two-sided','Adjacency',A,'ClusterThreshold',1,'Exact',true);
verifyEqual(testCase,sign(r.cluster_labels),[1 1 -1 -1]);
verifyEqual(testCase,r.cluster_labels(1),r.cluster_labels(2));
verifyEqual(testCase,r.cluster_labels(3),r.cluster_labels(4));
verifyEqual(testCase,r.p_fwer(1),r.p_fwer(2));
verifyEqual(testCase,r.family_size,4);
end

function testTfceSameSignFlipFamily(testCase)
x=[2 0;3 1;4 2]; A=logical([0 1;1 0]);
r=rsa_meeg.inference.group(x,'Method','tfce','Tail','two-sided', ...
    'Adjacency',A,'Exact',true);
verifyTrue(testCase,all(isfinite(r.enhanced_statistic)));
verifyEqual(testCase,r.n_permutations,8);
verifyEqual(testCase,r.p_fwer(1),sum(r.null_max>=r.enhanced_statistic(1))/8);
end

function testMappingTiesAndCompleteCases(testCase)
R=sampleR([1;2;2]);
model=[1;2;2];
s=rsa_meeg.mapping.compare(R,model,'Correlation','spearman');
k=rsa_meeg.mapping.compare(R,model,'Correlation','kendall-a');
verifyEqual(testCase,s.values(1),1,'AbsTol',1e-12);
verifyEqual(testCase,k.values(1),2/3,'AbsTol',1e-12);
R.values(3)=NaN;
s=rsa_meeg.mapping.compare(R,model,'Correlation','spearman');
verifyEqual(testCase,s.values(1),1,'AbsTol',1e-12);
end

function testCellstrModelAlignment(testCase)
R=sampleR([1;2;3]); R.condition_ids={'cat','dog','fish'};
model=struct(); model.condition_ids={'fish','cat','dog'};
model.values=[2;3;1];
out=rsa_meeg.mapping.compare(R,model,'Correlation','pearson');
verifyEqual(testCase,out.values(1),1,'AbsTol',1e-12);
verifyEqual(testCase,out.condition_ids,{'cat','dog','fish'});
end

function testPublicCorrelationVectorOrientation(testCase)
x=[1 2 2 4]; y=[4;2;2;1];
for method={"pearson","spearman","kendall-a"}
    row=rsa_meeg.mapping.correlation(x,y.',method{1});
    mixed=rsa_meeg.mapping.correlation(x,y,method{1});
    verifyEqual(testCase,mixed,row,'AbsTol',1e-12);
end
verifyEqual(testCase,rsa_meeg.mapping.correlation([1 NaN 2],[1;9;2], ...
    'kendall-a'),1,'AbsTol',1e-12);
end

function testGLMRejectsNonBooleanStandardize(testCase)
R=sampleR([1;2;3]);
verifyError(testCase,@() rsa_meeg.mapping.glm(R,[1;2;3], ...
    'Standardize',2),'rsa_meeg:mapping:Standardize');
end

function testGLMRankDeficiency(testCase)
R=sampleR([1;2;3]);
verifyError(testCase,@() rsa_meeg.mapping.glm(R,[1 2;2 4;3 6]), ...
    'rsa_meeg:mapping:GLMRank');
end

function testGLMScaleAndIntercept(testCase)
R=struct(); R.pairs=nchoosek(1:4,2); R.condition_ids=1:4;
R.center_ids=1; R.windows=struct('center_s',0);
X=[0 1;1 0;1 2;2 1;3 2;2 3];
y=3+2*X(:,1)-X(:,2);
R.values=reshape(y,[6 1 1]);
a=rsa_meeg.mapping.glm(R,X);
b=rsa_meeg.mapping.glm(R,[10*X(:,1),X(:,2)]);
verifyEqual(testCase,reshape(a.beta,1,2),[2 -1],'AbsTol',1e-12);
verifyEqual(testCase,reshape(b.beta,1,2),[.2 -1],'AbsTol',1e-12);
verifyEqual(testCase,a.intercept,3,'AbsTol',1e-12);
verifyEqual(testCase,b.intercept,3,'AbsTol',1e-12);
end

function testTfcePiecewiseIntegral(testCase)
% For two adjacent positive t values, extent is 2 up to min(t), then 1.
x=[1 1;2 3;3 5;4 7]; A=logical([0 1;1 0]);
r=rsa_meeg.inference.group(x,'Method','tfce','Tail','positive', ...
    'Adjacency',A,'Exact',true);
t=sort(r.statistic);
expected=[sqrt(2)*t(1)^3/3, ...
    sqrt(2)*t(1)^3/3+(t(2)^3-t(1)^3)/3];
verifyEqual(testCase,sort(r.enhanced_statistic),expected,'AbsTol',1e-10);
end

function testBHNoDiscoveries(testCase)
r=rsa_meeg.inference.bh([.4 .5 .6],.05);
verifyFalse(testCase,any(r.reject));
verifyEmpty(testCase,r.threshold);
end

function R=sampleR(y)
R=struct(); R.values=reshape(y,[3 1 1]); R.pairs=[1 2;1 3;2 3];
R.condition_ids=[1 2 3]; R.center_ids=10;
R.windows=struct('center_s',0);
end
