function validate_dataset(ds)
%VALIDATE_DATASET Validate canonical dataset, including manually built structs.
required={'data','sample_times_s','condition_ids','condition_order','fold_ids', ...
    'feature_ids','feature_types','positions_mm','hemisphere','bad_features','reference','subject_id'};
if ~isstruct(ds) || ~isscalar(ds) || ~all(isfield(ds,required))
    error('rsa_meeg:dataset:Schema','Dataset is missing canonical fields.');
end
x=ds.data;
if ~isnumeric(x) || ~isreal(x) || isempty(x) || ndims(x)>3 || any(~isfinite(x(:)))
    error('rsa_meeg:dataset:Data','Data must be finite real observations x features x time.');
end
n=size(x,1); f=size(x,2); t=size(x,3);
if ~isnumeric(ds.sample_times_s) || ~isreal(ds.sample_times_s) || numel(ds.sample_times_s)~=t || any(~isfinite(ds.sample_times_s(:))) || any(diff(ds.sample_times_s(:))<=0)
    error('rsa_meeg:dataset:Times','Invalid sample_times_s.');
end
checkid(ds.condition_ids,n,'condition_ids');
if ~isequal(ds.condition_order(:),unique(ds.condition_ids(:),'stable'))
    error('rsa_meeg:dataset:ConditionOrder','Condition order must be stable unique IDs.');
end
if ~isempty(ds.fold_ids), checkid(ds.fold_ids,n,'fold_ids'); end
checkid(ds.feature_ids,f,'feature_ids');
if numel(unique(ds.feature_ids))~=f
    error('rsa_meeg:dataset:FeatureIds','Feature IDs must be unique.');
end
if ~(isstring(ds.feature_types)||iscellstr(ds.feature_types)) || numel(ds.feature_types)~=f || any(ismissing(string(ds.feature_types)))
    error('rsa_meeg:dataset:Types','Invalid feature types.');
end
if isfield(ds,'feature_units') && (~(isstring(ds.feature_units)||iscellstr(ds.feature_units)) || numel(ds.feature_units)~=f || any(ismissing(string(ds.feature_units))))
    error('rsa_meeg:dataset:Units','Invalid feature units.');
end
if ~(isstring(ds.hemisphere)||iscellstr(ds.hemisphere)) || numel(ds.hemisphere)~=f || any(ismissing(string(ds.hemisphere)))
    error('rsa_meeg:dataset:Hemisphere','Invalid hemisphere labels.');
end
if ~isempty(ds.positions_mm) && (~isnumeric(ds.positions_mm) || ~isreal(ds.positions_mm) || ~isequal(size(ds.positions_mm),[f,3]) || any(~isfinite(ds.positions_mm(:))))
    error('rsa_meeg:dataset:Positions','Invalid feature coordinates.');
end
if ~islogical(ds.bad_features) || numel(ds.bad_features)~=f
    error('rsa_meeg:dataset:BadFeatures','Invalid bad-feature mask.');
end
end

function checkid(x,n,label)
if ~(isnumeric(x)||isstring(x)||iscellstr(x)) || numel(x)~=n || (isnumeric(x) && (~isreal(x)||any(~isfinite(x(:))))) || (isstring(x)&&any(ismissing(x)))
    error('rsa_meeg:dataset:Labels','Invalid %s.',label);
end
end
