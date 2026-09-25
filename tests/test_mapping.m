function tests=test_mapping
tests=functiontests(localfunctions);
end

function testCorrelationHandValues(testCase)
x=[1;2;2;4]; y=[1;2;3;4];
verifyEqual(testCase,rsa_meeg.mapping.correlation(x,y,'kendall-a'),5/6,'AbsTol',1e-12);
verifyEqual(testCase,rsa_meeg.mapping.correlation([1;2;3],[3;2;1],'spearman'),-1,'AbsTol',1e-12);
verifyTrue(testCase,isnan(rsa_meeg.mapping.correlation(ones(3,1),(1:3)','pearson')));
verifyEqual(testCase,rsa_meeg.mapping.correlation([1 2 3],[1;2;3],'kendall-a'),1);
verifyEqual(testCase,rsa_meeg.mapping.correlation([1 NaN 3],[1;2;3],'pearson'),1);
verifyError(testCase,@()rsa_meeg.mapping.correlation([1 2],[1;2;3],'pearson'),...
    'rsa_meeg:mapping:CorrelationInput');
end

function testConditionAlignmentAndCompare(testCase)
R=fixture;
% Neural [A B C] edge vector [AB AC BC] = [1 2 3].
% Model [C A B] edges [CA CB AB] = [2 3 1].
model=struct('condition_ids',{{'C','A','B'}},'values',[2;3;1]);
o=rsa_meeg.mapping.compare(R,model,'Correlation','pearson');
verifyEqual(testCase,o.values,1,'AbsTol',1e-12);
verifyError(testCase,@()rsa_meeg.mapping.compare(R,struct('condition_ids',{{'A','B','X'}},'values',[1;2;3])),...
    'rsa_meeg:mapping:ModelLabels');
end

function testDynamicLagSeconds(testCase)
R=fixture; R.values=cat(3,[1;2;3],[3;2;1]);
R.windows.center_s=[.1,.2];
model=reshape([1;2;3],3,1,1);
o=rsa_meeg.mapping.dynamic(R,model,0.1,0.1,'Correlation','pearson');
verifyTrue(testCase,isnan(o.values(1,1,1,1)));
verifyEqual(testCase,o.values(1,2,1,1),-1,'AbsTol',1e-12);
verifyEqual(testCase,o.model_index,[0;1]);
end

function testGLMAndRank(testCase)
R=fixture; R.condition_ids={'A','B','C','D'}; R.pairs=nchoosek(1:4,2);
X=[1 0;2 1;3 0;4 1;5 0;6 1];
R.values=reshape(7+2*X(:,1)-3*X(:,2),6,1,1);
o=rsa_meeg.mapping.glm(R,X);
verifyEqual(testCase,squeeze(o.beta),[2;-3],'AbsTol',1e-12);
verifyEqual(testCase,o.intercept,7,'AbsTol',1e-12);
verifyEqual(testCase,o.rank,3);
verifyError(testCase,@()rsa_meeg.mapping.glm(R,[X(:,1),2*X(:,1)]),'rsa_meeg:mapping:GLMRank');
end

function R=fixture
R=struct('values',reshape([1;2;3],3,1,1),'pairs',nchoosek(1:3,2),...
    'condition_ids',{{'A','B','C'}},'center_ids',1,...
    'windows',struct('center_s',.1));
end
