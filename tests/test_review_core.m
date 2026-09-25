function tests=test_review_core
% Independent integration checks across dataset, RDM, mapping, geometry, IO.
tests=functiontests(localfunctions);
end

function testFoldCrossnobisThroughMapping(testCase)
% Independent fold contrasts are [-2,-4] and [-4,-8] in one feature.
data=zeros(6,2,2);
data(:,1,1)=[0 2 4 0 4 8];
data(:,1,2)=2*data(:,1,1);
ds=rsa_meeg.dataset(data,[.1 .2],{'A','B','C','A','B','C'},...
    'FoldIds',[1 1 1 2 2 2],'FeatureTypes',{'eeg','eeg'});
R=rsa_meeg.rdm.searchlight(ds,{[1 2]},'Metric','crossnobis',...
    'Precision',eye(2),'WindowSamples',1);
verifyEqual(testCase,R.values(:,1,1),[4;16;4],'AbsTol',1e-12);
verifyEqual(testCase,R.values(:,1,2),[16;64;16],'AbsTol',1e-12);
verifyEqual(testCase,R.pair_counts,[2 2;2 2;2 2]);
verifyEqual(testCase,R.condition_ids,{'A','B','C'});
model=struct('values',[16;4;4],'condition_ids',{{'C','A','B'}});
map=rsa_meeg.mapping.compare(R,model,'Correlation','pearson');
verifyEqual(testCase,map.values,[1 1],'AbsTol',1e-12);
end

function testSingletonWindowsAndHemispheres(testCase)
w=rsa_meeg.rdm.windows(.1,1,1);
verifyEqual(testCase,numel(w.indices),1); verifyEqual(testCase,w.indices{1},1);
adj=rsa_meeg.geometry.spacetime_adjacency(logical([0 0;0 0]),2);
verifyTrue(testCase,adj(1,3)); verifyFalse(testCase,adj(1,2));
neigh=rsa_meeg.geometry.neighborhoods([0 0 0;0 0 .1],1,...
    'Hemisphere',{'lh','rh'});
verifyEqual(testCase,neigh,{1,2});
end

function testSTCZeroBasedOrderAndEEGShape(testCase)
tmp=[tempname '.stc']; cleanup=onCleanup(@()delete_if_exists(tmp));
rsa_meeg.io.write_stc(tmp,[2 0],[10 11;20 21],[0 .01]);
s=rsa_meeg.io.read_stc(tmp);
verifyEqual(testCase,s.vertices_zero_based,[2;0]);
verifyEqual(testCase,s.data,[10 11;20 21]);
ds=rsa_meeg.io.from_stc({tmp;tmp},{'A','B'},'Hemisphere',{'lh'});
verifyEqual(testCase,size(ds.data),[2 2 2]);
verifyEqual(testCase,squeeze(ds.data(1,:,:)),[10 11;20 21]);
verifyEqual(testCase,ds.feature_ids,{'lh:2','lh:0'});
EEG=struct('data',reshape(1:12,[2 3 2]),'times',[0 10 20]);
eeg=rsa_meeg.io.from_eeglab(EEG,{'A','B'});
verifyEqual(testCase,size(eeg.data),[2 2 3]);
verifyEqual(testCase,eeg.sample_times_s,[0 .01 .02],'AbsTol',1e-12);
end

function testMixedSensorSelectionAndPrecision(testCase)
data=zeros(6,2,2);
data(:,1,1)=[0;2;4;0;4;8]; data(:,1,2)=data(:,1,1);
data(:,2,1)=[0;20;40;0;40;80]; data(:,2,2)=data(:,2,1);
ds=rsa_meeg.dataset(data,[0 .01],{'A','B','C','A','B','C'},...
    'FoldIds',[1 1 1 2 2 2],...
    'FeatureTypes',{'eeg','meg'});
verifyError(testCase,@()rsa_meeg.rdm.searchlight(ds,{[1 2]},'Metric','sqeuclidean'),...
    'rsa_meeg:rdm:MixedTypes');
R=rsa_meeg.rdm.searchlight(ds,{[1 2]},'Metric','crossnobis',...
    'FeatureTypes','eeg','Precision',1,'WindowSamples',1);
verifyEqual(testCase,R.values(:,1,1),[8;32;8],'AbsTol',1e-12);
verifyError(testCase,@()rsa_meeg.rdm.searchlight(ds,{[1 2]},'Metric','crossnobis',...
    'FeatureTypes','eeg','Precision',-1),...
    'rsa_meeg:rdm:Precision');
end

function testFieldTripEpochAxis(testCase)
ft=struct('label',{{'E1','E2'}},'time',[0 .01],...
    'trial',reshape(1:12,[3 2 2]),'dimord','rpt_chan_time');
ds=rsa_meeg.io.from_fieldtrip(ft,{'A','B','A'},'FeatureTypes',{'eeg','eeg'});
verifyEqual(testCase,size(ds.data),[3 2 2]);
verifyEqual(testCase,ds.data,ft.trial);
end

function delete_if_exists(path)
if isfile(path), delete(path); end
end
