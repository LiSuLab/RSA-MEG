function ds = dataset(data, sample_times_s, condition_ids, varargin)
%DATASET Construct an observation x feature x time M/EEG RSA dataset.
% Conceptual adaptation of LiSuLab rsatoolbox +rsa/+meg preparation,
% commit c0dfd32f809eb816c4e94ae765feab0e4d12ab5a; rewritten API.
if ~isnumeric(data) || ~isreal(data) || isempty(data) || ndims(data) > 3 || any(~isfinite(data(:)))
    error('rsa_meeg:dataset:Data','Data must be a nonempty finite real numeric 3-D array.');
end
n = size(data,1); f = size(data,2); t = size(data,3);
if ~isnumeric(sample_times_s) || ~isreal(sample_times_s) || numel(sample_times_s) ~= t || any(~isfinite(sample_times_s(:))) || any(diff(sample_times_s(:)) <= 0)
    error('rsa_meeg:dataset:Times','Sample times must be finite, strictly increasing, and match the time axis.');
end
condition_ids = checklabels(condition_ids,n,'condition_ids');
p = inputParser;
p.FunctionName = 'rsa_meeg.dataset';
addParameter(p,'FoldIds',[]);
addParameter(p,'FeatureIds',1:f);
addParameter(p,'FeatureTypes',repmat("unknown",1,f));
addParameter(p,'FeatureUnits',repmat("unspecified",1,f));
addParameter(p,'Positions',[]);
addParameter(p,'PositionUnit','mm');
addParameter(p,'Hemisphere',repmat("",1,f));
addParameter(p,'BadFeatures',false(1,f));
addParameter(p,'Reference',[]);
addParameter(p,'SubjectId',[]);
parse(p,varargin{:}); o=p.Results;
if isempty(o.FoldIds)
    folds=[];
else
    folds=checklabels(o.FoldIds,n,'FoldIds');
end
ids=checklabels(o.FeatureIds,f,'FeatureIds');
if numel(unique(ids)) ~= f
    error('rsa_meeg:dataset:FeatureIds','FeatureIds must be unique.');
end
types=checktext(o.FeatureTypes,f,'FeatureTypes');
units=checktext(o.FeatureUnits,f,'FeatureUnits');
hemi=checktext(o.Hemisphere,f,'Hemisphere');
bad=o.BadFeatures;
if ~islogical(bad) || numel(bad) ~= f
    error('rsa_meeg:dataset:BadFeatures','BadFeatures must be one logical per feature.');
end
if isempty(o.Positions)
    pos=[];
else
    pos=o.Positions;
    if ~isnumeric(pos) || ~isreal(pos) || ~isequal(size(pos),[f,3]) || any(~isfinite(pos(:)))
        error('rsa_meeg:dataset:Positions','Positions must be finite feature x 3 coordinates.');
    end
    switch lower(char(o.PositionUnit))
        case 'mm', scale=1;
        case 'cm', scale=10;
        case 'm', scale=1000;
        otherwise, error('rsa_meeg:dataset:PositionUnit','PositionUnit must be mm, cm, or m.');
    end
    pos=double(pos)*scale;
end
ds=struct();
ds.data=double(data);
ds.sample_times_s=double(sample_times_s(:).');
ds.condition_ids=condition_ids;
ds.condition_order=unique(condition_ids,'stable');
ds.fold_ids=folds;
ds.feature_ids=ids;
ds.feature_types=types;
ds.feature_units=units;
ds.positions_mm=pos;
ds.hemisphere=hemi;
ds.bad_features=bad(:).';
ds.reference=o.Reference;
ds.subject_id=o.SubjectId;
rsa_meeg.validate_dataset(ds);
end

function x=checklabels(x,n,label)
if ischar(x), x=cellstr(x); end
if ~(isnumeric(x) || isstring(x) || iscellstr(x)) || numel(x) ~= n || isempty(x)
    error('rsa_meeg:dataset:Labels','%s must contain one numeric or text label per item.',label);
end
if isnumeric(x) && (~isreal(x) || any(~isfinite(x(:))))
    error('rsa_meeg:dataset:Labels','%s must contain finite real labels.',label);
end
if isstring(x) && any(ismissing(x(:)))
    error('rsa_meeg:dataset:Labels','%s contains missing text.',label);
end
x=x(:).';
end

function x=checktext(x,n,label)
if ischar(x), x=cellstr(x); end
if ~(isstring(x) || iscellstr(x)) || numel(x) ~= n
    error('rsa_meeg:dataset:Text','%s must contain one text label per feature.',label);
end
x=string(x(:).');
if any(ismissing(x))
    error('rsa_meeg:dataset:Text','%s contains missing text.',label);
end
end
