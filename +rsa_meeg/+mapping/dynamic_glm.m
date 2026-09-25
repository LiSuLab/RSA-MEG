function out=dynamic_glm(R,models,model_times_s,lags_s,varargin)
%DYNAMIC_GLM Descriptive multi-model OLS at explicitly matched model times.
% Neural window time = model time + lag, all in seconds. Each lag is fitted
% separately. Unmatched windows remain NaN. No best-lag search or p-values.
p=inputParser;
addParameter(p,'Tolerance',1e-9); addParameter(p,'Standardize',false);
parse(p,varargin{:}); o=p.Results;
[m,ids]=rsa_meeg.mapping.prepare_model(R,models);
mt=model_times_s(:); lag=lags_s(:)';
if ~isnumeric(mt) || ~isreal(mt) || any(~isfinite(mt)) || isempty(mt) || any(diff(mt)<=0) || size(m,2)~=numel(mt) || ndims(m)>3
    error('rsa_meeg:mapping:ModelTime','ModelTimes must increase and match model dimension 2.');
end
if ~isnumeric(lag) || ~isreal(lag) || isempty(lag) || any(~isfinite(lag)) || numel(unique(lag))~=numel(lag)
    error('rsa_meeg:mapping:Lag','Lags must be distinct finite seconds.');
end
if ~isnumeric(o.Tolerance) || ~isreal(o.Tolerance) || ~isscalar(o.Tolerance) || ~isfinite(o.Tolerance) || o.Tolerance<0
    error('rsa_meeg:mapping:Tolerance','Tolerance must be nonnegative finite seconds.');
end
if ~(islogical(o.Standardize)&&isscalar(o.Standardize)) && ...
        ~(isnumeric(o.Standardize)&&isscalar(o.Standardize)&&ismember(o.Standardize,[0,1]))
    error('rsa_meeg:mapping:Standardize','Standardize must be scalar logical or 0/1.');
end
nt=size(R.values,3); nc=size(R.values,2); nm=size(m,3); nl=numel(lag);
if ~isfield(R.windows,'center_s') || numel(R.windows.center_s)~=nt
    error('rsa_meeg:mapping:NeuralTime','R.windows.center_s is required.');
end
rt=R.windows.center_s(:); np=size(R.values,1);
beta=nan(nc,nt,nm,nl); intercept=nan(nc,nt,nl);
rankValue=nan(nc,nt,nl); condition=nan(nc,nt,nl);
residual=nan(nc,nt,nl); nPairs=zeros(nc,nt,nl); match=zeros(nt,nl);
for a=1:nl
    for t=1:nt
        q=find(abs(mt+lag(a)-rt(t))<=o.Tolerance);
        if numel(q)>1, error('rsa_meeg:mapping:AmbiguousTime','Multiple model times match a neural window.'); end
        if isempty(q), continue; end
        match(t,a)=q;
        predictors=reshape(m(:,q,:),np,nm);
        for c=1:nc
            y=R.values(:,c,t); keep=isfinite(y) & all(isfinite(predictors),2);
            X=predictors(keep,:); y=y(keep); nPairs(c,t,a)=numel(y);
            if numel(y)<nm+1, error('rsa_meeg:mapping:GLMSamples','Too few finite RDM pairs for full-rank GLM.'); end
            if o.Standardize
                sd=std(X,0,1);
                if any(sd==0), error('rsa_meeg:mapping:GLMRank','Constant model predictor.'); end
                X=(X-mean(X,1))./sd;
            end
            X=[ones(numel(y),1),X]; rankValue(c,t,a)=rank(X);
            if rankValue(c,t,a)~=nm+1, error('rsa_meeg:mapping:GLMRank','Model design is rank deficient.'); end
            condition(c,t,a)=cond(X); b=X\y;
            intercept(c,t,a)=b(1); beta(c,t,:,a)=b(2:end);
            residual(c,t,a)=sum((y-X*b).^2);
        end
    end
end
out.beta=beta; out.intercept=intercept; out.rank=rankValue;
out.condition_number=condition; out.residual_ss=residual; out.n_pairs=nPairs;
out.standardized=logical(o.Standardize); out.model_times_s=mt;
out.lags_s=lag; out.model_index=match;
out.center_ids=R.center_ids; out.windows=R.windows; out.condition_ids=ids;
out.assumption=['Neural time equals model time plus lag. Each matched center/window/lag has its own OLS fit. ' ...
    'Dependent RDM edges make coefficients descriptive; no analytic edgewise p-value, best-lag inference, or cross-validation is supplied.'];
end
