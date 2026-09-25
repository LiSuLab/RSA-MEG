function out=average(subject_rdms,varargin)
%AVERAGE Descriptive fixed-effects mean of aligned subject RDMs.
% Pairwise finite mean; NaN denotes missing. No population inference is made.
% Adapted conceptually from LiSuLab +rsa/+meg/averageSearchlightRDMs.m,
% commit c0dfd32f809eb816c4e94ae765feab0e4d12ab5a. Rewritten API.
p=inputParser; addParameter(p,'AlignedCenters',false); parse(p,varargin{:});
aligned=p.Results.AlignedCenters;
if ~(islogical(aligned)&&isscalar(aligned)) && ...
        ~(isnumeric(aligned)&&isscalar(aligned)&&ismember(aligned,[0,1]))
    error('rsa_meeg:rdm:Alignment','AlignedCenters must be logical scalar or 0/1.');
end
if ~iscell(subject_rdms) || isempty(subject_rdms) || ~all(cellfun(@(r)isstruct(r)&&isscalar(r),subject_rdms))
    error('rsa_meeg:rdm:Subjects','Input must be a nonempty cell array of scalar subject RDM structs.');
end
first=subject_rdms{1}; validate_rdm(first);
nsub=numel(subject_rdms); np=size(first.values,1); nc=size(first.values,2); nt=size(first.values,3);
if nc>1 && ~aligned
    error('rsa_meeg:rdm:Alignment','Multiple centers require AlignedCenters=true to acknowledge a registered common domain.');
end
acc=zeros(np,nc,nt); counts=zeros(np,nc,nt);
subject_ids=cell(1,nsub);
subject_feature_indices=cell(1,nsub);
subject_pattern_dimensions=nan(nc,nt,nsub);
for s=1:nsub
    r=subject_rdms{s}; validate_rdm(r);
    if ~isequal(size3(r.values),[np,nc,nt])
        error('rsa_meeg:rdm:Shape','Subject RDM dimensions differ.');
    end
    if ~isequal(r.center_ids,first.center_ids)
        error('rsa_meeg:rdm:Centers','Center IDs/order differ; align geometry before averaging.');
    end
    if ~same_windows(r.windows,first.windows)
        error('rsa_meeg:rdm:Windows','Explicit sample windows or times differ.');
    end
    for field={'metric','mode','feature_types','feature_units','reference', ...
            'source_tag','geometry_tag','normalization','fold_weighting','sample_times_s'}
        name=field{1};
        if xor(isfield(r,name),isfield(first,name)) || ...
                (isfield(first,name) && ~isequaln(r.(name),first.(name)))
            error('rsa_meeg:rdm:Metadata','%s differs across subjects.',name);
        end
    end
    source_map=pair_map(first.condition_ids,r.condition_ids,r.pairs);
    v=r.values(source_map,:,:);
    good=isfinite(v);
    v(~good)=0;
    acc=acc+v;
    counts=counts+double(good);
    if isfield(r,'subject_id'), subject_ids{s}=r.subject_id; end
    if isfield(r,'feature_indices')
        subject_feature_indices{s}=r.feature_indices;
    end
    if isfield(r,'pattern_dimensions')
        subject_pattern_dimensions(:,:,s)=r.pattern_dimensions;
    end
end
avg=acc./counts; avg(counts==0)=NaN;
out=struct();
out.values=avg; out.valid_counts=counts; out.n_subjects=nsub;
out.pairs=first.pairs; out.condition_ids=first.condition_ids;
out.center_ids=first.center_ids; out.windows=first.windows;
out.metric=first.metric; out.mode=first.mode;
for field={'feature_types','feature_units','reference','source_tag', ...
        'geometry_tag','normalization','fold_weighting','sample_times_s'}
    name=field{1};
    if isfield(first,name), out.(name)=first.(name); end
end
out.subject_ids=subject_ids;
out.subject_feature_indices=subject_feature_indices;
out.subject_pattern_dimensions=subject_pattern_dimensions;
out.analysis_level='descriptive fixed-effects subject RDM mean';
out.assumption=['Each subject contributes equally where its entry is finite. ' ...
    'Centers and windows refer to the same registered domain. ' ...
    'This is not population inference; do not average correlation-derived p-values.'];
end

function validate_rdm(r)
required={'values','pairs','condition_ids','center_ids','windows','metric','mode'};
if ~isstruct(r) || ~isscalar(r) || ~all(isfield(r,required))
    error('rsa_meeg:rdm:Schema','Subject RDM lacks required fields.');
end
v=r.values;
if ~isnumeric(v) || ~isreal(v) || isempty(v) || ndims(v)>3 || any(isinf(v(:)))
    error('rsa_meeg:rdm:Values','RDM values must be real numeric with NaN only for missing entries.');
end
n=numel(r.condition_ids); np=n*(n-1)/2;
if n<2 || ~isvector(r.condition_ids) || numel(unique(string(r.condition_ids)))~=n || ...
        ~isequal(r.pairs,nchoosek(1:n,2)) || size(v,1)~=np
    error('rsa_meeg:rdm:Pairs','RDM pair labels/order are invalid.');
end
if ~isvector(r.center_ids) || numel(r.center_ids)~=size(v,2) || ...
        numel(unique(string(r.center_ids)))~=numel(r.center_ids)
    error('rsa_meeg:rdm:Centers','Center IDs must be unique and match the center axis.');
end
if ~isstruct(r.windows) || ~isscalar(r.windows) || ~all(isfield(r.windows,{'indices','start_s','center_s','end_s'}))
    error('rsa_meeg:rdm:Windows','Explicit time-window indices and times are required.');
end
if isfield(r,'feature_indices') && (~iscell(r.feature_indices) || numel(r.feature_indices)~=size(v,2))
    error('rsa_meeg:rdm:FeatureMetadata','Feature indices must contain one neighborhood per center.');
end
if isfield(r,'pattern_dimensions') && (~isnumeric(r.pattern_dimensions) || ...
        ~isequal(size(r.pattern_dimensions),[size(v,2),size(v,3)]) || ...
        any(r.pattern_dimensions(:)<1 | r.pattern_dimensions(:)~=floor(r.pattern_dimensions(:))))
    error('rsa_meeg:rdm:FeatureMetadata','Pattern dimensions must match centers x windows.');
end
nt=size(v,3); w=r.windows;
if ~iscell(w.indices) || numel(w.indices)~=nt || ...
        ~valid_times(w.start_s,nt) || ~valid_times(w.center_s,nt) || ~valid_times(w.end_s,nt) || ...
        any(w.start_s(:)>w.center_s(:)) || any(w.center_s(:)>w.end_s(:))
    error('rsa_meeg:rdm:Windows','Time-window metadata must match the RDM time axis.');
end
end

function ok=valid_times(t,n)
ok=isnumeric(t)&&isreal(t)&&isvector(t)&&numel(t)==n&&all(isfinite(t(:)));
end

function dims=size3(v)
dims=[size(v,1),size(v,2),size(v,3)];
end

function ok=same_windows(a,b)
ok=isequal(a.indices,b.indices) && isequal(a.start_s,b.start_s) && ...
    isequal(a.center_s,b.center_s) && isequal(a.end_s,b.end_s);
end

function map=pair_map(target_ids,source_ids,source_pairs)
n=numel(target_ids);
if numel(source_ids)~=n || ~isequal(source_pairs,nchoosek(1:n,2))
    error('rsa_meeg:rdm:Conditions','Condition count or pair order differs.');
end
if isnumeric(target_ids) && isnumeric(source_ids)
    [found,loc]=ismember(target_ids,source_ids);
elseif (isstring(target_ids)||iscellstr(target_ids)) && ...
        (isstring(source_ids)||iscellstr(source_ids))
    [found,loc]=ismember(string(target_ids),string(source_ids));
else
    error('rsa_meeg:rdm:Conditions','Condition ID types differ.');
end
if ~all(found) || numel(unique(loc))~=n
    error('rsa_meeg:rdm:Conditions','Condition IDs differ across subjects.');
end
target_pairs=nchoosek(1:n,2); map=zeros(size(target_pairs,1),1);
for k=1:numel(map)
    ab=sort(loc(target_pairs(k,:)));
    map(k)=find(source_pairs(:,1)==ab(1)&source_pairs(:,2)==ab(2),1);
end
end
