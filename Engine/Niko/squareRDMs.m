function RDMs=squareRDMs(RDMs_utv)
% converts set of row-vector RDMs_utv to square form (despite being
% rows, RDMs are stacked along the 3rd dimension, just as square RDMs
% would be. this avoids ambiguity when the RDMs_utv is square and could
% be either a single RDM or a number of vectorized RDMs.)
% RDMs may be bare or wrapped with meta-data in a struct object. they
% will be returned in the same format as passed.

if isstruct(RDMs_utv)
    % wrapped
    RDMs_utv_struct=RDMs_utv;
    RDMs_utv=unwrapRDMs(RDMs_utv_struct);
    
    nRDMs=size(RDMs_utv,3);
    RDMs=[];
    for RDMI=1:nRDMs
        RDMs=cat(3,RDMs,squareRDM(RDMs_utv(:,:,RDMI)));
    end
    
    RDMs=wrapRDMs(RDMs,RDMs_utv_struct);
else
    % bare
    nRDMs=size(RDMs_utv,3);
    RDMs=[];
    for RDMI=1:nRDMs
        RDMs=cat(3,RDMs,squareform(vectorizeRDM(RDMs_utv(:,:,RDMI))));
    end
end
