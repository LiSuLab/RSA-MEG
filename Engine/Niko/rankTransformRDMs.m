function RDMs_rankTransformed=rankTransformRDMs(RDMs,rankTransformType,scale01)

%% preparations
if ~exist('rankTransformType','var')
    rankTransformType='randomOrderAmongEquals';
end

if ~exist('scale01','var')
    scale01=true;
end

[RDMs_bare_utv,nRDMs]=unwrapRDMs(vectorizeRDMs(RDMs));


%% rank transform each RDM
for RDMI=1:nRDMs
    if strcmp(rankTransformType,'randomOrderAmongEquals')
        RDMs_bare_utv(1,:,RDMI)=rankTransform_randomOrderAmongEquals(RDMs_bare_utv(1,:,RDMI),scale01);
    elseif strcmp(rankTransformType,'equalsStayEqual')
        RDMs_bare_utv(1,:,RDMI)=rankTransform_equalsStayEqual(RDMs_bare_utv(1,:,RDMI),scale01);
    end
end


%% resquare & rewrap
if isstruct(RDMs)
    RDMs_rankTransformed=wrapRDMs(RDMs_bare_utv,RDMs);
    if length(RDMs(1).RDM)~=numel(RDMs(1).RDM)
        % they were square, so resquare them
        RDMs_rankTransformed=squareRDMs(RDMs_rankTransformed);
    end
else
    RDMs_rankTransformed=RDMs_bare_utv;
    if length(RDMs(:,:,1))~=numel(RDMs(:,:,1))
        % they were square, so resquare them
        RDMs_rankTransformed=squareRDMs(RDMs_rankTransformed);
    end
end