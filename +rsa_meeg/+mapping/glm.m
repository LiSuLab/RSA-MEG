function out = glm(R, models, varargin)
%GLM Descriptive OLS across RDM edges; no edgewise sampling inference.
p=inputParser; addParameter(p,'Standardize',false); parse(p,varargin{:});
[m,ids]=rsa_meeg.mapping.prepare_model(R,models);
if ndims(m)>2, error('rsa_meeg:mapping:GLMShape','GLM requires static pairs x models.'); end
if ~(islogical(p.Results.Standardize)&&isscalar(p.Results.Standardize)) && ...
        ~(isnumeric(p.Results.Standardize)&&isscalar(p.Results.Standardize)&&ismember(p.Results.Standardize,[0,1]))
    error('rsa_meeg:mapping:Standardize','Standardize must be scalar logical or 0/1.');
end
nc=size(R.values,2); nt=size(R.values,3); nm=size(m,2);
beta=nan(nc,nt,nm); intercept=nan(nc,nt); rankValue=zeros(nc,nt); condition=nan(nc,nt); residual=nan(nc,nt); nPairs=zeros(nc,nt);
for t=1:nt
    for c=1:nc
        y=R.values(:,c,t); keep=isfinite(y) & all(isfinite(m),2);
        X=m(keep,:); y=y(keep); nPairs(c,t)=numel(y);
        if numel(y)<nm+1, error('rsa_meeg:mapping:GLMSamples','Too few finite RDM pairs for full-rank GLM.'); end
        if p.Results.Standardize
            mu=mean(X,1); sd=std(X,0,1);
            if any(sd==0), error('rsa_meeg:mapping:GLMRank','Constant model predictor.'); end
            X=(X-mu)./sd;
        end
        X=[ones(numel(y),1),X];
        rankValue(c,t)=rank(X);
        if rankValue(c,t)~=nm+1, error('rsa_meeg:mapping:GLMRank','Model design is rank deficient.'); end
        condition(c,t)=cond(X);
        b=X\y;
        intercept(c,t)=b(1); beta(c,t,:)=b(2:end); residual(c,t)=sum((y-X*b).^2);
    end
end
out.beta=beta; out.intercept=intercept; out.rank=rankValue; out.condition_number=condition;
out.residual_ss=residual; out.n_pairs=nPairs; out.standardized=logical(p.Results.Standardize);
out.center_ids=R.center_ids; out.windows=R.windows; out.condition_ids=ids;
out.assumption='OLS coefficients describe dependent RDM edges; no analytic p-value. Standardization applies to predictors within each fit.';
end
