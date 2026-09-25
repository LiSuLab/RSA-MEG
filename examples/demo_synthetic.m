function out=demo_synthetic(modality,make_figures)
%DEMO_SYNTHETIC Self-contained sensor/source RSA and random-effects example.
% out=demo_synthetic('eeg',false); modalities: eeg,mag,grad,source.
% Simulated data validate execution and known signal, not biological claims.
if nargin<1,modality='eeg';end
if nargin<2,make_figures=true;end
assert(ismember(modality,{'eeg','mag','grad','source'}),'Unknown modality');
prior=rng;cleanup=onCleanup(@()rng(prior));rng(1701,'twister');
times=-.10:.005:.20;nf=12;ns=8;conditions=repmat((1:4)',4,1);
folds=repelem((1:2)',8); c=[-2 -1 1 2];
position=[(0:nf-1)'*5,zeros(nf,2)];hemi=repmat({''},nf,1);
if strcmp(modality,'source'),hemi(1:6)={'lh'};hemi(7:12)={'rh'};end
hood=rsa_meeg.geometry.neighborhoods(position,11,'Hemisphere',hemi);
model.values=[];model.condition_ids=1:4;
pairs=nchoosek(1:4,2);model.values=(c(pairs(:,1))-c(pairs(:,2))).^2;model.values=model.values(:);
maps=[];
for s=1:ns
    x=randn(numel(conditions),nf,numel(times));
    spatial=sin((1:nf)*1.7);envelope=exp(-((times-.10)/.035).^2);
    for o=1:numel(conditions)
        x(o,:,:)=reshape(x(o,:,:),nf,[])+1.5*c(conditions(o))*spatial'*envelope;
    end
    ds=rsa_meeg.dataset(x,times,conditions,'FoldIds',folds, ...
        'FeatureIds',101+(0:nf-1)*7,'FeatureTypes',repmat({modality},nf,1), ...
        'Positions',position,'Hemisphere',hemi,'Reference','synthetic zero reference','SubjectId',s);
    R=rsa_meeg.rdm.searchlight(ds,hood,'WindowSamples',5,'StepSamples',4,'Metric','sqeuclidean','CenterIds',ds.feature_ids);
    fit=rsa_meeg.mapping.compare(R,model,'Correlation','spearman');
    maps(s,:)=fit.values(:)'; %#ok<AGROW>
end
inference=rsa_meeg.inference.group(maps,'Exact',true,'Method','max-t');
out=struct('rdm',R,'fit',fit,'subject_maps',maps,'inference',inference,'modality',modality);
fprintf('%s: %d subjects, %d centers, %d windows; %d/%d tests reject at FWER .05.\n',modality,ns,size(fit.values,1),size(fit.values,2),sum(inference.significant),numel(inference.significant));
if make_figures
    figure('Name',['RSA-MEEG synthetic ',modality]);
    ax1=subplot(1,2,1);rsa_meeg.plot.rdm(R,1,round(numel(R.windows.center_s)/2),ax1);
    ax2=subplot(1,2,2);rsa_meeg.plot.map(fit,1,ax2);
end
end
