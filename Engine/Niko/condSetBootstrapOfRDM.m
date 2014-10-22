function resampledRDMs_utv=condSetBootstrapOfRDM(RDM,nResamplings)

RDM=squareRDM(RDM);
RDM_utv=vectorizeRDM(RDM);
nCond=size(RDM,1);

resampledRDMs_utv=nan(nResamplings,size(RDM_utv,2));

for resamplingI=1:nResamplings
    inds=ceil(rand(1,nCond)*nCond);
    resampledRDMs_utv(resamplingI,:)=vectorizeRDM(RDM(inds,inds));
end
