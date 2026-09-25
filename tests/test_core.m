function tests=test_core
%TEST_CORE Hand-computed M/EEG RSA core regressions; base MATLAB only.
tests=functiontests(localfunctions);
end

function testDatasetAndUnits(testCase)
data=reshape(1:24,[4 3 2]);
ds=rsa_meeg.dataset(data,[0 .01],["z","a","z","a"], ...
    'FoldIds',[2 2 9 9],'FeatureIds',[101 17 900], ...
    'FeatureTypes',["EEG","EEG","EEG"], ...
    'FeatureUnits',["V","V","V"], ...
    'Positions',[0 0 0; .01 0 0; .02 0 0], ...
    'PositionUnit','m','BadFeatures',[false true false], ...
    'Reference',struct('scheme','average'));
verifyEqual(testCase,ds.condition_order,["z","a"]);
verifyEqual(testCase,ds.positions_mm(:,1),[0;10;20]);
verifyEqual(testCase,ds.feature_ids,[101 17 900]);
verifyEqual(testCase,ds.feature_units,["V","V","V"]);
verifyEqual(testCase,ds.reference.scheme,'average');
verifyError(testCase,@() rsa_meeg.dataset(data,[0 0],[1 1 2 2]),'rsa_meeg:dataset:Times');
verifyError(testCase,@() rsa_meeg.dataset(data,[0 .01],[1 1 2]),'rsa_meeg:dataset:Labels');
end

function testGeometryAndBoundary(testCase)
p=[0 0 0;1 0 0;2 0 0;1 0 0];
nb=rsa_meeg.geometry.neighborhoods(p,1,'Hemisphere',["L","L","L","R"], ...
    'Centers',[2 4]);
verifyEqual(testCase,nb{1},[1 2 3]);
verifyEqual(testCase,nb{2},4);
A=rsa_meeg.geometry.mesh_adjacency([1 2 3;3 4 5],5);
verifyTrue(testCase,A(1,3));
verifyFalse(testCase,A(1,5));
ST=rsa_meeg.geometry.spacetime_adjacency(A,2);
verifyTrue(testCase,ST(1,6));
verifyFalse(testCase,ST(1,10));
verifyEqual(testCase,size(ST),[10 10]);
end

function testWindowsInclusive(testCase)
w=rsa_meeg.rdm.windows([-.1 0 .1 .2 .3],3,2);
verifyEqual(testCase,w.indices,{1:3,3:5});
verifyEqual(testCase,w.center_s,[0 .2],'AbsTol',1e-12);
w=rsa_meeg.rdm.windows(0:.1:.3,4,1);
verifyEqual(testCase,numel(w.indices),1);
verifyEqual(testCase,w.indices{1},1:4);
end

function testEuclideanModesAndIds(testCase)
% Conditions A,B have patterns [1,2;3,4] and [0,0;0,0].
x=zeros(2,2,2); x(1,:,:)=[1 3;2 4];
ds=rsa_meeg.dataset(x,[0 .1],[88 7],'FeatureIds',[101 999]);
R=rsa_meeg.rdm.searchlight(ds,{[1 2]},'WindowSamples',2, ...
    'Metric','sqeuclidean','CenterIds',500);
verifyEqual(testCase,R.values(1,1,1),30/4,'AbsTol',1e-12);
verifyEqual(testCase,R.condition_ids,[88 7]);
verifyEqual(testCase,R.center_ids,500);
verifyEqual(testCase,R.pairs,[1 2]);
S=rsa_meeg.rdm.searchlight(ds,{[1 2]},'WindowSamples',2, ...
    'Mode','spatial','Metric','sqeuclidean');
verifyEqual(testCase,S.values(1), (2^2+3^2)/2,'AbsTol',1e-12);
T=rsa_meeg.rdm.searchlight(ds,{[1 2]},'WindowSamples',2, ...
    'Mode','temporal','Metric','sqeuclidean');
verifyEqual(testCase,T.values(1),(1.5^2+3.5^2)/2,'AbsTol',1e-12);
end

function testCorrelationAndMixedTypes(testCase)
x=zeros(2,2,1); x(1,:)=[1 2]; x(2,:)=[2 1];
ds=rsa_meeg.dataset(x,0,[1 2]);
R=rsa_meeg.rdm.searchlight(ds,{1:2});
verifyEqual(testCase,R.values(1),2,'AbsTol',1e-12);
x(1,:)=[1 1]; ds=rsa_meeg.dataset(x,0,[1 2]);
R=rsa_meeg.rdm.searchlight(ds,{1:2});
verifyTrue(testCase,isnan(R.values(1)));
verifyNotEmpty(testCase,R.diagnostic{1,1});
ds=rsa_meeg.dataset(x,0,[1 2],'FeatureTypes',["EEG","MAG"]);
verifyError(testCase,@() rsa_meeg.rdm.searchlight(ds,{1:2}), ...
    'rsa_meeg:rdm:MixedTypes');
R=rsa_meeg.rdm.searchlight(ds,{1:2},'FeatureTypes','EEG','Metric','sqeuclidean');
verifyEqual(testCase,R.values(1),1);
verifyError(testCase,@() rsa_meeg.rdm.searchlight(ds,{1:2}, ...
    'FeatureTypes',["EEG","MAG"],'Metric','sqeuclidean'), ...
    'rsa_meeg:rdm:MixedTypes');
ds=rsa_meeg.dataset(x,0,[1 2],'FeatureTypes',["EEG","EEG"], ...
    'FeatureUnits',["V","microV"]);
verifyError(testCase,@() rsa_meeg.rdm.searchlight(ds,{1:2},'Metric','sqeuclidean'), ...
    'rsa_meeg:rdm:MixedUnits');
end

function testCrossnobisIndependentFolds(testCase)
x=zeros(4,2,1); x(1,:)=[2 0]; x(3,:)=[3 0];
ds=rsa_meeg.dataset(x,0,[10 20 10 20],'FoldIds',[7 7 55 55]);
R=rsa_meeg.rdm.searchlight(ds,{1:2},'Metric','crossnobis','Precision',eye(2));
verifyEqual(testCase,R.values(1),3,'AbsTol',1e-12);
verifyEqual(testCase,R.fold_count,2);
verifyEqual(testCase,R.pair_counts,[2 2]);
ds2=rsa_meeg.dataset(x,0,[10 20 10 20],'FoldIds',[55 55 7 7]);
R2=rsa_meeg.rdm.searchlight(ds2,{1:2},'Metric','crossnobis','Precision',eye(2));
verifyEqual(testCase,R2.values,R.values,'AbsTol',1e-12);
verifyError(testCase,@() rsa_meeg.rdm.searchlight(ds,{1:2},'Metric','crossnobis'), ...
    'rsa_meeg:rdm:Precision');
verifyError(testCase,@() rsa_meeg.rdm.searchlight(ds,{1:2},'Metric','crossnobis','Precision',eye(3)), ...
    'rsa_meeg:rdm:Precision');
end

function testMissingFoldCellAndBadFeature(testCase)
x=zeros(3,2,1);
ds=rsa_meeg.dataset(x,0,[1 2 1],'FoldIds',[1 1 2]);
verifyError(testCase,@() rsa_meeg.rdm.searchlight(ds,{1:2},'Metric','crossnobis','Precision',eye(2)), ...
    'rsa_meeg:rdm:MissingCell');
ds=rsa_meeg.dataset(x,0,[1 2 1],'BadFeatures',[true false]);
verifyError(testCase,@() rsa_meeg.rdm.searchlight(ds,{1}), ...
    'rsa_meeg:rdm:EmptyNeighborhood');
end

function testCrossnobisScalarAndNegative(testCase)
% A scalar-pattern, three-fold example gives ordered-product mean 11/3.
x=reshape([1 0 2 0 3 0],[6 1 1]);
ds=rsa_meeg.dataset(x,0,[1 2 1 2 1 2],'FoldIds',[10 10 20 20 30 30]);
R=rsa_meeg.rdm.searchlight(ds,{1},'Metric','crossnobis','Precision',1);
verifyEqual(testCase,R.values(1),11/3,'AbsTol',1e-12);
% Opposite independent fold estimates may legitimately yield negative distance.
y=reshape([1 0 -1 0],[4 1 1]);
ds=rsa_meeg.dataset(y,0,[1 2 1 2],'FoldIds',[5 5 9 9]);
R=rsa_meeg.rdm.searchlight(ds,{1},'Metric','crossnobis','Precision',1);
verifyEqual(testCase,R.values(1),-1,'AbsTol',1e-12);
end

function testPairCenterWindowAxes(testCase)
x=zeros(3,2,3);
x(1,1,:)=[1 2 3]; x(2,1,:)=[2 3 4]; x(3,1,:)=[4 5 6];
x(1,2,:)=[1 1 1]; x(2,2,:)=[3 3 3]; x(3,2,:)=[6 6 6];
ds=rsa_meeg.dataset(x,[.01 .02 .04],["C","A","B"], ...
    'FeatureIds',[700 5]);
R=rsa_meeg.rdm.searchlight(ds,{1,2},'WindowSamples',2, ...
    'StepSamples',1,'Metric','sqeuclidean','CenterIds',[700 5]);
verifyEqual(testCase,size(R.values),[3 2 2]);
verifyEqual(testCase,R.pairs,[1 2;1 3;2 3]);
verifyEqual(testCase,R.center_ids,[700 5]);
verifyEqual(testCase,R.windows.indices,{1:2,2:3});
verifyEqual(testCase,R.values(1,1,1),1,'AbsTol',1e-12);
verifyEqual(testCase,R.values(1,2,2),4,'AbsTol',1e-12);
end

function testCellstrIdentifiersAndSingletons(testCase)
ds=rsa_meeg.dataset(reshape([1 2],[2 1 1]),0,{'left','right'}, ...
    'FoldIds',{'block-a','block-b'},'FeatureIds',{'channel-7'}, ...
    'FeatureTypes',{'EEG'},'Hemisphere',{'L'});
verifyTrue(testCase,isscalar(ds));
verifyEqual(testCase,ds.condition_order,{'left','right'});
verifyEqual(testCase,ds.feature_ids,{'channel-7'});
R=rsa_meeg.rdm.searchlight(ds,{1},'Metric','sqeuclidean', ...
    'CenterIds',{'channel-7'});
verifyTrue(testCase,isscalar(R));
verifyEqual(testCase,R.condition_ids,{'left','right'});
verifyEqual(testCase,R.center_ids,{'channel-7'});
verifyEqual(testCase,size(R.values),[1 1]);
verifyEqual(testCase,R.values(1),1);
end
