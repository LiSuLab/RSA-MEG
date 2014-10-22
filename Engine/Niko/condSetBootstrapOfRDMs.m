function resampledRDMs_utv=condSetBootstrapOfRDMs(RDMs,nResamplings)
% uses bootstrap resampling of the conditions set to resample a set of RDMs.
% the resampled RDMs are returned in upper triangle form (rows), stacked
% along the 3rd (index of input RDM) and 4th (resampling index)
% dimensions (for compatibility with square RDMs).


RDMs_sq=squareRDMs(unwrapRDMs(RDMs));
nRDMs=size(RDMs_sq,3);
nCond=size(RDMs_sq,1);
resampledRDMs_utv=nan(1,size(vectorizeRDM(RDMs_sq(:,:,1)),2),nRDMs,nResamplings);

for resamplingI=1:nResamplings
    inds=ceil(rand(1,nCond)*nCond);
    resampledRDMs_utv(:,:,:,resamplingI)=vectorizeRDMs(RDMs_sq(inds,inds,:));
end
