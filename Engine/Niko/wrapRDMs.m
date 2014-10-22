function RDMs_struct=wrapRDMs(RDMs,refRDMs_struct)
% wraps similiarity matrices RDMs (in square or upper triangle form)
% into a structured array with meta data copied from refRDMs_struct
% (which needs to have the same number of RDMs).(if they are already
% wrapped already then the wrapping (metadata) is replaced by that of
% refRDMs_struct.
% is refRDMs_struct is omitted the RDMs are given a generic meta
% information.

if isstruct(RDMs)
    % wrapped already, but replace the wrapping
    nRDMs=numel(RDMs);
else
    nRDMs=size(RDMs,3);
end    

if ~exist('refRDMs_struct','var')
    for RDMI=1:nRDMs
        refRDMs_struct(RDMI).name='[unnamed RDM]';
        refRDMs_struct(RDMI).color=[0 0 0];
    end
end

RDMs_struct=refRDMs_struct;
if isstruct(RDMs)
    % wrapped already, but replace the wrapping
    for RDMI=1:nRDMs
        RDMs_struct(RDMI).RDM=RDMs(RDMI).RDM;
    end
else
    % RDMs need wrapping
    for RDMI=1:nRDMs
        RDMs_struct(RDMI).RDM=RDMs(:,:,RDMI);
    end
end
