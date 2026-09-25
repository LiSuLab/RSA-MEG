function summary=demo_public_sample(data_dir)
%DEMO_PUBLIC_SAMPLE Analyze exported, genuine MNE sample EEG/mag/grad epochs.
% Prepare input with examples/data/export_mne_sample.py; see REAL_DATA.md.
% One subject is an IO/execution validation, not population inference.
types={'eeg','mag','grad'};summary=struct;
for k=1:numel(types)
    type=types{k};s=load(fullfile(data_dir,['sample_audvis_',type,'_epochs.mat']));
    ds=rsa_meeg.dataset(s.data,s.sample_times_s,s.condition_ids,'FoldIds',s.fold_ids, ...
        'FeatureIds',s.feature_ids,'FeatureTypes',s.feature_types,'Positions',s.positions_mm, ...
        'Reference','as stored in source FIF; no additional rereference','SubjectId','MNE sample');
    h=rsa_meeg.geometry.neighborhoods(ds.positions_mm,45);
    R=rsa_meeg.rdm.searchlight(ds,h,'WindowSamples',9,'StepSamples',8,'Metric','correlation','CenterIds',ds.feature_ids);
    % 1/2 auditory, 3/4 visual: modality model, explicit condition order.
    ids=double(R.condition_ids);p=R.pairs;
    model.condition_ids=ids;model.values=double((ids(p(:,1))<=2)~=(ids(p(:,2))<=2));model.values=model.values(:);
    fit=rsa_meeg.mapping.compare(R,model,'Correlation','kendall-a');
    assert(all(isfinite(R.values(:)))&&all(isfinite(fit.values(:))),'Public example produced undefined values.');
    summary.(type)=struct('n_observations',size(ds.data,1),'n_features',size(ds.data,2), ...
        'n_samples',size(ds.data,3),'n_windows',numel(R.windows.center_s), ...
        'rdm_min',min(R.values(:)),'rdm_max',max(R.values(:)), ...
        'similarity_min',min(fit.values(:)),'similarity_max',max(fit.values(:)), ...
        'condition_ids',ids,'time_start_s',ds.sample_times_s(1),'time_end_s',ds.sample_times_s(end));
    fprintf('%s real data: %d trials, %d channels, %d windows; finite results.\n',type,size(ds.data,1),size(ds.data,2),numel(R.windows.center_s));
end
end
