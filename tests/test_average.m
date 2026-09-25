function tests=test_average
%TEST_AVERAGE Fixed-effects descriptive RDM averaging, alignment and counts.
tests=functiontests(localfunctions);
end

function testReorderAndNaNCounts(testCase)
a=fixture({'A','B','C'},[1;NaN;3]);
b=fixture({'C','A','B'},[2;4;6]);
a.subject_id='s1'; b.subject_id='s2';
r=rsa_meeg.rdm.average({a,b});
verifyEqual(testCase,r.condition_ids,{'A','B','C'});
verifyEqual(testCase,r.values(:),[3.5;2;3.5],'AbsTol',1e-12);
verifyEqual(testCase,r.valid_counts(:),[2;1;2]);
verifyEqual(testCase,r.n_subjects,2);
verifyEqual(testCase,r.subject_ids,{'s1','s2'});
verifyEqual(testCase,r.analysis_level,'descriptive fixed-effects subject RDM mean');
mapped=rsa_meeg.mapping.compare(r,[3.5;2;3.5],'Correlation','pearson');
verifyEqual(testCase,mapped.values(1),1,'AbsTol',1e-12);
end

function testAllMissingRemainsNaN(testCase)
a=fixture({'A','B','C'},[NaN;2;NaN]);
b=fixture({'A','B','C'},[NaN;4;NaN]);
r=rsa_meeg.rdm.average({a,b});
verifyTrue(testCase,isnan(r.values(1)) && isnan(r.values(3)));
verifyEqual(testCase,r.values(2),3);
verifyEqual(testCase,r.valid_counts(:),[0;2;0]);
end

function testAlignmentAndMetadataErrors(testCase)
a=fixture({'A','B','C'},[1;2;3]); b=a;
a.values=repmat(a.values,[1 2 1]); b.values=a.values;
a.center_ids=[10 20]; b.center_ids=[10 20];
a.feature_types=["EEG","EEG"]; b.feature_types=a.feature_types;
a.feature_units=["V","V"]; b.feature_units=a.feature_units;
a.pattern_dimensions=[2;3]; b.pattern_dimensions=[4;5];
a.feature_indices={1:2,1:3}; b.feature_indices={1:4,1:5};
verifyError(testCase,@()rsa_meeg.rdm.average({a,b}), ...
    'rsa_meeg:rdm:Alignment');
r=rsa_meeg.rdm.average({a,b},'AlignedCenters',true);
verifyEqual(testCase,r.subject_pattern_dimensions(:,:,1),[2;3]);
verifyEqual(testCase,r.subject_pattern_dimensions(:,:,2),[4;5]);
b.center_ids=[20 10];
verifyError(testCase,@()rsa_meeg.rdm.average({a,b},'AlignedCenters',true), ...
    'rsa_meeg:rdm:Centers');
b=a; b.metric='correlation';
verifyError(testCase,@()rsa_meeg.rdm.average({a,b},'AlignedCenters',true), ...
    'rsa_meeg:rdm:Metadata');
b=a; b.feature_units=["uV","uV"];
verifyError(testCase,@()rsa_meeg.rdm.average({a,b},'AlignedCenters',true), ...
    'rsa_meeg:rdm:Metadata');
b=a; b.normalization='unscaled';
verifyError(testCase,@()rsa_meeg.rdm.average({a,b},'AlignedCenters',true), ...
    'rsa_meeg:rdm:Metadata');
end

function testTimeAndSourceErrors(testCase)
a=fixture({'A','B','C'},[1;2;3]); b=a;
b.windows.center_s=.011;
verifyError(testCase,@()rsa_meeg.rdm.average({a,b}), ...
    'rsa_meeg:rdm:Windows');
b=a; b.sample_times_s=[0 .021];
verifyError(testCase,@()rsa_meeg.rdm.average({a,b}), ...
    'rsa_meeg:rdm:Metadata');
b=a; b.source_tag='different-mesh';
verifyError(testCase,@()rsa_meeg.rdm.average({a,b}), ...
    'rsa_meeg:rdm:Metadata');
end

function testSearchlightProvenance(testCase)
x=zeros(2,3,2); x(1,:,:)=[1 2;3 4;5 6];
ds=rsa_meeg.dataset(x,[0 .01],{'A','B'}, ...
    'FeatureTypes',{'EEG','EEG','EEG'}, ...
    'FeatureUnits',{'V','V','V'},'SubjectId','S1');
r=rsa_meeg.rdm.searchlight(ds,{[1 2],[2 3]}, ...
    'Metric','sqeuclidean','WindowSamples',2,'Mode','spatiotemporal');
verifyEqual(testCase,r.feature_indices,{[1 2],[2 3]});
verifyEqual(testCase,r.pattern_dimensions,[4;4]);
verifyEqual(testCase,r.feature_units,["V","V"]);
verifyEqual(testCase,r.normalization,'per-pattern-dimension');
verifyEqual(testCase,r.sample_times_s,[0 .01]);
verifyEqual(testCase,r.subject_id,'S1');
end

function r=fixture(ids,values)
r=struct(); r.values=reshape(values,[3 1 1]);
r.pairs=[1 2;1 3;2 3]; r.condition_ids=ids; r.center_ids=10;
r.windows=struct('indices',{{1:2}},'start_s',0,'center_s',.01, ...
    'end_s',.02,'width_samples',2,'step_samples',1);
r.sample_times_s=[0 .02];
r.metric='sqeuclidean'; r.mode='spatiotemporal';
r.normalization='per-pattern-dimension';
r.fold_weighting='not applicable'; r.source_tag='registered-template-1';
r.feature_types="EEG"; r.feature_units="V"; r.reference=[];
r.feature_indices={1:2}; r.pattern_dimensions=4;
end
