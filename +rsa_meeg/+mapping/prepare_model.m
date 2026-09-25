function [v, ids] = prepare_model(R, model)
%PREPARE_MODEL Validate and align model RDM pairs to neural condition order.
% Numeric models assume R.pairs order. A struct must supply condition_ids
% and values whose rows use lexicographic upper-triangle order in those IDs.
if ~isstruct(R) || ~isfield(R,'values') || ~isfield(R,'pairs') || ~isfield(R,'condition_ids')
    error('rsa_meeg:mapping:InvalidRDM','R requires values, pairs, condition_ids.');
end
ids = R.condition_ids;
n = numel(ids);
np = n*(n-1)/2;
if size(R.pairs,1) ~= np || size(R.pairs,2) ~= 2 || ~isequal(R.pairs, nchoosek(1:n,2))
    error('rsa_meeg:mapping:PairOrder','R.pairs must be lexicographic upper triangle.');
end
if isstruct(model)
    if ~isfield(model,'condition_ids') || ~isfield(model,'values')
        error('rsa_meeg:mapping:ModelLabels','Model struct requires condition_ids and values.');
    end
    mIds = model.condition_ids;
    if numel(mIds) ~= n || numel(unique(string(mIds))) ~= n || numel(unique(string(ids))) ~= n
        error('rsa_meeg:mapping:ModelLabels','Condition labels must be unique and match.');
    end
    [found,loc] = ismember(string(ids),string(mIds));
    if ~all(found)
        error('rsa_meeg:mapping:ModelLabels','Model and neural conditions differ.');
    end
    v = model.values;
    if size(v,1) ~= np
        error('rsa_meeg:mapping:ModelShape','Model pair dimension mismatch.');
    end
    sourcePairs = nchoosek(1:n,2);
    map = zeros(np,1);
    for k=1:np
        a=sort(loc(R.pairs(k,:)));
        map(k)=find(sourcePairs(:,1)==a(1) & sourcePairs(:,2)==a(2),1);
    end
    v=v(map,:,:);
else
    v=model;
    if size(v,1) ~= np
        error('rsa_meeg:mapping:ModelShape','Model pair dimension mismatch.');
    end
end
if ~isnumeric(v) || ~isreal(v) || any(~isfinite(v(:)))
    error('rsa_meeg:mapping:ModelValues','Model values must be finite real numeric.');
end
end
