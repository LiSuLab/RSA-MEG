function ds = from_stc(files, condition_ids, varargin)
%FROM_STC Load per-observation STCs into a single-subject Dataset.
% FILES is observation x hemisphere cellstr. Hemisphere (e.g. {'lh','rh'})
% is REQUIRED because STC itself contains no hemisphere field. Surfaces is
% an optional cell of matching FreeSurfer triangular surface filenames.
% Separate observations must have identical vertex IDs and time axes.
p=inputParser; p.KeepUnmatched=true;
addParameter(p,'Hemisphere',{}); addParameter(p,'Surfaces',{});
parse(p,varargin{:});
files=cellstr(string(files)); h=cellstr(string(p.Results.Hemisphere));
if isempty(files)||isempty(h)||numel(h)~=size(files,2)||numel(unique(h))~=numel(h)||any(strlength(string(h))==0)
    error('rsa_meeg:io:Hemisphere','Explicit unique hemisphere per FILES column required.');
end
surfs=p.Results.Surfaces;
if ~isempty(surfs)&&numel(surfs)~=numel(h),error('rsa_meeg:io:Surface','One surface per hemisphere required.');end
nobs=size(files,1); blocks=cell(1,numel(h)); ids={}; hemis={}; xyz=[];
for k=1:numel(h)
    first=rsa_meeg.io.read_stc(files{1,k});
    if k==1,times=first.sample_times_s;elseif ~same_times(times,first.sample_times_s),error('rsa_meeg:io:Alignment','Hemisphere times disagree.');end
    nv=numel(first.vertices_zero_based); a=zeros(nobs,nv,numel(times));
    for o=1:nobs
        if o==1,s=first;else,s=rsa_meeg.io.read_stc(files{o,k});end
        if ~isequal(s.vertices_zero_based,first.vertices_zero_based)||~same_times(times,s.sample_times_s)
            error('rsa_meeg:io:Alignment','STC observation vertices/times disagree.');
        end
        a(o,:,:)=s.data;
    end
    blocks{k}=a;
    ids=[ids;cellstr(string(h{k})+":"+string(first.vertices_zero_based))]; %#ok<AGROW>
    hemis=[hemis;repmat(h(k),nv,1)]; %#ok<AGROW>
    if ~isempty(surfs)
        surf=rsa_meeg.io.read_surface(surfs{k}); ix=first.vertices_zero_based+1;
        if any(ix>size(surf.positions_mm,1)),error('rsa_meeg:io:Alignment','STC vertex missing from surface.');end
        xyz=[xyz;surf.positions_mm(ix,:)]; %#ok<AGROW>
    end
end
extra=named_args(p.Unmatched);
ds=rsa_meeg.dataset(cat(2,blocks{:}),times,condition_ids, ...
    'FeatureIds',ids,'FeatureTypes',repmat({'source'},numel(ids),1), ...
    'Hemisphere',hemis,'Positions',xyz,'PositionUnit','mm',extra{:});
ds.provenance=struct('format','MNE-STC','files',{files},'surfaces',{surfs}, ...
    'vertex_convention','file IDs zero-based; local array indices one-based');
end
function tf=same_times(a,b)
tf=numel(a)==numel(b)&&all(abs(a-b)<1e-7);
end
function args=named_args(s)
names=fieldnames(s);args=cell(1,2*numel(names));
for k=1:numel(names),args{2*k-1}=names{k};args{2*k}=s.(names{k});end
end
