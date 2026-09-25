function R=searchlight(ds,neighborhoods,varargin)
%SEARCHLIGHT Compute condition RDMs over feature neighborhoods and time windows.
% Reimplements the conceptual workflow in LiSuLab +rsa/+meg/
% MEGSearchlightRDMs_source.m and searchlightMapping_MEG_sensor.m, commit
% c0dfd32f809eb816c4e94ae765feab0e4d12ab5a. No legacy code copied.
% Precision is an externally estimated inverse noise covariance in the
% transformed pattern space, independent of data used for distances.
rsa_meeg.validate_dataset(ds);
if ~iscell(neighborhoods) || isempty(neighborhoods)
    error('rsa_meeg:rdm:Neighborhoods','Neighborhoods must be a nonempty cell array of local feature indices.');
end
p=inputParser;
addParameter(p,'WindowSamples',1); addParameter(p,'StepSamples',1);
addParameter(p,'Metric','correlation'); addParameter(p,'Mode','spatiotemporal');
addParameter(p,'CenterIds',[]); addParameter(p,'FeatureTypes',[]);
addParameter(p,'Precision',[]);
parse(p,varargin{:}); o=p.Results;
metric=lower(string(o.Metric)); mode=lower(string(o.Mode));
if ~isscalar(metric) || ~any(metric==["correlation","sqeuclidean","crossnobis"])
    error('rsa_meeg:rdm:Metric','Metric must be correlation, sqeuclidean, or crossnobis.');
end
if ~isscalar(mode) || ~any(mode==["spatial","temporal","spatiotemporal"])
    error('rsa_meeg:rdm:Mode','Unknown pattern mode.');
end
ncenter=numel(neighborhoods); nf=size(ds.data,2);
center_ids=o.CenterIds;
if isempty(center_ids), center_ids=1:ncenter; end
if numel(center_ids)~=ncenter || numel(unique(center_ids))~=ncenter
    error('rsa_meeg:rdm:CenterIds','CenterIds must be unique and match neighborhoods.');
end
if ~isempty(o.FeatureTypes)
    types=string(o.FeatureTypes);
    if any(ismissing(types)) || isempty(types)
        error('rsa_meeg:rdm:FeatureTypes','Invalid selected feature types.');
    end
else
    types=[];
end
w=rsa_meeg.rdm.windows(ds.sample_times_s,o.WindowSamples,o.StepSamples);
nc=numel(ds.condition_order); np=nc*(nc-1)/2;
if np<1, error('rsa_meeg:rdm:Conditions','At least two distinct conditions are required.'); end
pairs=zeros(np,2); k=0;
for i=1:nc-1
    for j=i+1:nc
        k=k+1; pairs(k,:)=[i j];
    end
end
[~,cond]=ismember(ds.condition_ids,ds.condition_order);
count=accumarray(cond(:),1,[nc,1]);
pair_counts=[count(pairs(:,1)),count(pairs(:,2))];
if metric=="crossnobis"
    if isempty(ds.fold_ids)
        error('rsa_meeg:rdm:Folds','Crossnobis requires explicit independent FoldIds.');
    end
    [folds,~,fold]=unique(ds.fold_ids,'stable'); nfold=numel(folds);
    if nfold<2, error('rsa_meeg:rdm:Folds','Crossnobis requires at least two independent folds.'); end
    for fi=1:nfold
        for ci=1:nc
            if ~any(fold==fi & cond(:)==ci)
                error('rsa_meeg:rdm:MissingCell','Every condition must occur in every fold.');
            end
        end
    end
    if isempty(o.Precision)
        error('rsa_meeg:rdm:Precision','Crossnobis requires externally estimated Precision.');
    end
end
values=nan(np,ncenter,numel(w.indices));
diagnostic=cell(ncenter,numel(w.indices));
feature_indices=cell(1,ncenter);
pattern_dimensions=zeros(ncenter,numel(w.indices));
resolved_types=strings(1,ncenter); resolved_units=strings(1,ncenter);
for c=1:ncenter
    idx=neighborhoods{c};
    if ~isnumeric(idx) || ~isreal(idx) || ~isvector(idx) || isempty(idx) || any(~isfinite(idx)) || any(idx<1 | idx>nf | idx~=floor(idx)) || numel(unique(idx))~=numel(idx)
        error('rsa_meeg:rdm:NeighborhoodIndex','Each neighborhood needs unique local feature indices.');
    end
    idx=idx(:).'; idx=idx(~ds.bad_features(idx));
    if ~isempty(types), idx=idx(ismember(string(ds.feature_types(idx)),types)); end
    if isempty(idx)
        error('rsa_meeg:rdm:EmptyNeighborhood','No good features remain in neighborhood %d.',c);
    end
    if numel(unique(string(ds.feature_types(idx))))>1
        error('rsa_meeg:rdm:MixedTypes','Neighborhood %d mixes feature types with different physical units; select one FeatureTypes value.',c);
    end
    if isfield(ds,'feature_units') && numel(unique(string(ds.feature_units(idx))))>1
        error('rsa_meeg:rdm:MixedUnits','Neighborhood %d mixes signal units; rescale data and record a common FeatureUnits value.',c);
    end
    feature_indices{c}=idx;
    resolved_types(c)=string(ds.feature_types(idx(1)));
    if isfield(ds,'feature_units')
        resolved_units(c)=string(ds.feature_units(idx(1)));
    else
        resolved_units(c)="unspecified";
    end
    for ti=1:numel(w.indices)
        X=makepatterns(ds.data(:,idx,w.indices{ti}),mode);
        ndim=size(X,2);
        pattern_dimensions(c,ti)=ndim;
        if metric=="crossnobis"
            P=precision_for(o.Precision,c,ncenter,ndim);
            M=zeros(nc,ndim,nfold);
            for fi=1:nfold
                for ci=1:nc
                    M(ci,:,fi)=mean(X(fold==fi & cond(:)==ci,:),1);
                end
            end
        else
            M=zeros(nc,ndim);
            for ci=1:nc, M(ci,:)=mean(X(cond(:)==ci,:),1); end
        end
        for pi=1:np
            a=pairs(pi,1); b=pairs(pi,2);
            switch metric
                case 'sqeuclidean'
                    delta=M(a,:)-M(b,:);
                    values(pi,c,ti)=sum(delta.^2)/ndim;
                case 'correlation'
                    xa=M(a,:)-mean(M(a,:)); xb=M(b,:)-mean(M(b,:));
                    den=sqrt(sum(xa.^2)*sum(xb.^2));
                    if den==0
                        diagnostic{c,ti}='Correlation undefined for constant condition pattern.';
                    else
                        values(pi,c,ti)=1-sum(xa.*xb)/den;
                    end
                case 'crossnobis'
                    delta=reshape(M(a,:,:)-M(b,:,:),[ndim,nfold]).';
                    total=0;
                    for f1=1:nfold
                        for f2=1:nfold
                            if f1~=f2, total=total+delta(f1,:)*P*delta(f2,:).'; end
                        end
                    end
                    values(pi,c,ti)=total/(nfold*(nfold-1)*ndim);
            end
        end
    end
end
R=struct();
R.values=values; R.pairs=pairs; R.condition_ids=ds.condition_order;
R.center_ids=center_ids; R.windows=w; R.metric=char(metric); R.mode=char(mode);
R.sample_times_s=ds.sample_times_s;
R.pair_counts=pair_counts; R.diagnostic=diagnostic; R.fold_count=[];
R.feature_indices=feature_indices;
R.pattern_dimensions=pattern_dimensions;
R.feature_types=resolved_types; R.feature_units=resolved_units;
R.reference=ds.reference; R.subject_id=ds.subject_id;
if metric=="correlation"
    R.normalization='Pearson correlation distance';
    R.fold_weighting='not applicable';
else
    R.normalization='per-pattern-dimension';
    if metric=="crossnobis"
        R.fold_weighting='equal ordered independent fold pairs';
    else
        R.fold_weighting='not applicable';
    end
end
if metric=="crossnobis", R.fold_count=nfold; end
end

function X=makepatterns(raw,mode)
n=size(raw,1); f=size(raw,2); t=size(raw,3);
switch mode
    case 'spatial'
        X=reshape(mean(raw,3),n,f);
    case 'temporal'
        X=reshape(mean(raw,2),n,t);
    otherwise
        X=reshape(raw,n,f*t);
end
end

function P=precision_for(raw,c,ncenter,ndim)
if iscell(raw)
    if numel(raw)~=ncenter
        error('rsa_meeg:rdm:Precision','Cell Precision must have one matrix per center.');
    end
    P=raw{c};
else
    P=raw;
end
if ~isnumeric(P) || ~isreal(P) || ~isequal(size(P),[ndim,ndim]) || any(~isfinite(P(:))) || norm(P-P','fro')>1e-10*max(1,norm(P,'fro'))
    error('rsa_meeg:rdm:Precision','Precision must be a finite symmetric matrix matching pattern dimension.');
end
P=(P+P')/2;
if any(eig(P)<=0)
    error('rsa_meeg:rdm:Precision','Precision must be positive definite.');
end
end
