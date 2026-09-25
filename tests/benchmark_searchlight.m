function report=benchmark_searchlight(n_conditions,n_centers)
%BENCHMARK_SEARCHLIGHT Explicit, reproducible full-vs-batched static mapping.
% Reports returned RDM bytes, not process peak resident memory. No downloads.
if nargin<1,n_conditions=80;end
if nargin<2,n_centers=16;end
prior=rng;cleanup=onCleanup(@()rng(prior));rng(271828,'twister');
nobs=2*n_conditions;nf=24;times=0:.005:.095;
ds=rsa_meeg.dataset(randn(nobs,nf,numel(times)),times,repmat((1:n_conditions)',2,1));
hood=cell(1,n_centers);
for k=1:n_centers,hood{k}=mod((k:k+4)-1,nf)+1;end
pairs=nchoosek(1:n_conditions,2);model=abs(pairs(:,1)-pairs(:,2));
tic;R=rsa_meeg.rdm.searchlight(ds,hood,'WindowSamples',4,'StepSamples',4,'Metric','sqeuclidean');
full=rsa_meeg.mapping.compare(R,model,'Correlation','pearson');full_seconds=toc;
info=whos('R');rdm_bytes=info.bytes;
tic;batch=rsa_meeg.mapping.searchlight(ds,hood,model,'BatchSize',1,'WindowSamples',4,'StepSamples',4,'Metric','sqeuclidean','Correlation','pearson');batch_seconds=toc;
error_max=max(abs(full.values(:)-batch.values(:)));assert(error_max<1e-12);
report=struct('n_conditions',n_conditions,'n_centers',n_centers,'n_features',nf, ...
    'n_windows',size(R.values,3),'full_seconds',full_seconds,'batch_seconds',batch_seconds, ...
    'full_returned_rdm_bytes',rdm_bytes,'max_abs_difference',error_max, ...
    'memory_note','Returned full RDM struct bytes only, not measured process peak memory. BatchSize=1.');
disp(report);
end
