function tests=test_streaming_dynamic_glm
tests=functiontests(localfunctions);
end

function testDynamicGlmHandBetasAndLagDirection(testCase)
R=rdmfixture;
X0=[1 0;2 1;3 0;4 1;5 0;6 1];
X1=[2 1;1 0;4 0;3 1;6 1;5 0];
model=zeros(6,2,2); model(:,1,:)=X0; model(:,2,:)=X1;
R.values(:,:,1)=reshape(5-X1(:,1)+4*X1(:,2),6,1);
R.values(:,:,2)=reshape(7+2*X0(:,1)-3*X0(:,2),6,1);
R.values(:,:,3)=reshape(9+3*X1(:,1)+X1(:,2),6,1);
R.windows.center_s=[0 .1 .2];
out=rsa_meeg.mapping.dynamic_glm(R,model,[0 .1],[-.1 .1]);
verifyEqual(testCase,out.model_index,[2 0;0 1;0 2]);
verifyEqual(testCase,squeeze(out.beta(1,1,:,1)),[-1;4],'AbsTol',1e-12);
verifyEqual(testCase,squeeze(out.beta(1,2,:,2)),[2;-3],'AbsTol',1e-12);
verifyEqual(testCase,squeeze(out.beta(1,3,:,2)),[3;1],'AbsTol',1e-12);
verifyEqual(testCase,out.intercept(1,1,1),5,'AbsTol',1e-12);
verifyEqual(testCase,out.intercept(1,2,2),7,'AbsTol',1e-12);
verifyTrue(testCase,isnan(out.intercept(1,1,2)));
verifyTrue(testCase,isnan(out.beta(1,2,1,1)));
end

function testDynamicGlmConditionLabelsRankAndStandardize(testCase)
R=rdmfixture; X=[1 0;2 1;3 0;4 1;5 0;6 1];
R.values=reshape(2+3*X(:,1)-X(:,2),6,1,1); R.windows.center_s=.1;
% Reorder the six upper-triangle rows for model condition order D,A,B,C.
model=struct('condition_ids',{{'D','A','B','C'}},...
    'values',zeros(6,1,2));
pairs=nchoosek(1:4,2); neural={'A','B','C','D'};
for k=1:6
    a=find(strcmp(neural{pairs(k,1)},model.condition_ids));
    b=find(strcmp(neural{pairs(k,2)},model.condition_ids));
    source=find(pairs(:,1)==min(a,b) & pairs(:,2)==max(a,b));
    model.values(source,1,:)=X(k,:);
end
out=rsa_meeg.mapping.dynamic_glm(R,model,0,.1);
verifyEqual(testCase,squeeze(out.beta),[3;-1],'AbsTol',1e-12);
verifyError(testCase,@()rsa_meeg.mapping.dynamic_glm(R,cat(3,X(:,1),2*X(:,1)),0,.1),...
    'rsa_meeg:mapping:GLMRank');
verifyError(testCase,@()rsa_meeg.mapping.dynamic_glm(R,model,0,.1,'Standardize',2),...
    'rsa_meeg:mapping:Standardize');
end

function testStreamingEqualsFullMappingAndSlicesPrecision(testCase)
data=zeros(6,2,2);
data(:,1,1)=[0;2;4;0;4;8]; data(:,1,2)=2*data(:,1,1);
data(:,2,1)=[0;3;7;0;5;9]; data(:,2,2)=2*data(:,2,1);
ds=rsa_meeg.dataset(data,[0 .01],{'A','B','C','A','B','C'},...
    'FoldIds',[1 1 1 2 2 2],'FeatureTypes',{'eeg','eeg'});
hoods={[1 2],1,2}; ids=[10 30 50]; P={eye(2),2,3};
R=rsa_meeg.rdm.searchlight(ds,hoods,'Metric','crossnobis',...
    'Precision',P,'CenterIds',ids,'WindowSamples',1);
expected=rsa_meeg.mapping.compare(R,[1;3;2],'Correlation','pearson');
for b=[1 2 3]
    got=rsa_meeg.mapping.searchlight(ds,hoods,[1;3;2],...
        'BatchSize',b,'Metric','crossnobis','Precision',P,...
        'CenterIds',ids,'WindowSamples',1,'Correlation','pearson');
    verifyEqual(testCase,got.values,expected.values,'AbsTol',1e-12);
    verifyEqual(testCase,got.center_ids,ids);
    verifyEqual(testCase,got.pair_counts,R.pair_counts);
end
verifyError(testCase,@()rsa_meeg.mapping.searchlight(ds,hoods,[1;3;2],...
    'BatchSize',0),'rsa_meeg:mapping:BatchSize');
verifyError(testCase,@()rsa_meeg.mapping.searchlight(ds,hoods,[1;3;2],...
    'Precision',{1}),'rsa_meeg:mapping:Precision');
end

function testStreamingWindowModeAndLabeledModel(testCase)
data=zeros(6,2,4);
for t=1:4
    data(:,1,t)=t*[0;1;3;0;1;3];
    data(:,2,t)=t*[0;2;4;0;2;4];
end
ds=rsa_meeg.dataset(data,[0 .01 .02 .03],{'A','B','C','A','B','C'},...
    'FeatureTypes',{'eeg','eeg'});
hoods={[1 2],1};
model=struct('condition_ids',{{'C','A','B'}},'values',[3;2;1]);
R=rsa_meeg.rdm.searchlight(ds,hoods,'Metric','sqeuclidean','Mode','spatial',...
    'WindowSamples',2,'StepSamples',2,'CenterIds',[5 9]);
full=rsa_meeg.mapping.compare(R,model,'Correlation','spearman');
stream=rsa_meeg.mapping.searchlight(ds,hoods,model,'BatchSize',1,...
    'Metric','sqeuclidean','Mode','spatial','WindowSamples',2,...
    'StepSamples',2,'CenterIds',[5 9],'Correlation','spearman');
verifyEqual(testCase,stream.values,full.values,'AbsTol',1e-12);
verifyEqual(testCase,stream.windows.center_s,[.005 .025],'AbsTol',1e-12);
end

function R=rdmfixture
R=struct(); R.values=nan(6,1,3); R.pairs=nchoosek(1:4,2);
R.condition_ids={'A','B','C','D'}; R.center_ids=7;
R.windows=struct('center_s',[0 .1 .2]);
end
