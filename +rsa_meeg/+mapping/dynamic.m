function out = dynamic(R, model, model_times_s, lags_s, varargin)
%DYNAMIC Compare dynamic RDMs; neural_time = model_time + lag.
% Unmatched neural windows remain NaN. Times are in seconds, not indices.
p=inputParser;
addParameter(p,'Correlation','spearman');
addParameter(p,'Tolerance',1e-9);
parse(p,varargin{:});
[m,ids]=rsa_meeg.mapping.prepare_model(R,model);
mt=model_times_s(:); lag=lags_s(:)';
if ndims(m)>3 || size(m,2)~=numel(mt) || ~isnumeric(mt) || any(~isfinite(mt)) || any(diff(mt)<=0)
    error('rsa_meeg:mapping:ModelTime','ModelTimes must increase and match model dimension 2.');
end
if ~isnumeric(lag) || isempty(lag) || any(~isfinite(lag)) || numel(unique(lag))~=numel(lag)
    error('rsa_meeg:mapping:Lag','Lags must be distinct finite seconds.');
end
tol=p.Results.Tolerance;
if ~isscalar(tol) || ~isfinite(tol) || tol<0, error('rsa_meeg:mapping:Tolerance','Invalid time tolerance.'); end
nt=size(R.values,3); nc=size(R.values,2); nm=size(m,3);
if ~isfield(R.windows,'center_s') || numel(R.windows.center_s)~=nt
    error('rsa_meeg:mapping:NeuralTime','R.windows.center_s is required.');
end
rt=R.windows.center_s(:);
v=nan(nc,nt,nm,numel(lag)); match=zeros(nt,numel(lag));
for a=1:numel(lag)
    for t=1:nt
        candidates=find(abs(mt+lag(a)-rt(t))<=tol);
        if numel(candidates)>1, error('rsa_meeg:mapping:AmbiguousTime','Multiple model times match a neural time.'); end
        if isempty(candidates), continue; end
        q=candidates; match(t,a)=q;
        for j=1:nm
            for c=1:nc
                v(c,t,j,a)=rsa_meeg.mapping.correlation(R.values(:,c,t),m(:,q,j),p.Results.Correlation);
            end
        end
    end
end
out.values=v; out.center_ids=R.center_ids; out.windows=R.windows; out.condition_ids=ids;
out.model_times_s=mt; out.lags_s=lag; out.model_index=match;
out.correlation=char(p.Results.Correlation);
out.assumption='Neural time equals model time plus lag; unmatched points have NaN. RDM edges are dependent; no analytic p-value.';
end
