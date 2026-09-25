function out=condition_permutation(R,model,varargin)
%CONDITION_PERMUTATION Fixed-effects RSA label-permutation inference.
% One condition relabeling is applied to ALL model RDMs and all neural
% centers/windows; dependent RDM edges are never shuffled independently.
% This is conditional on the observed neural data, not a subject RFX test.
p=inputParser;
addParameter(p,'Correlation','spearman'); addParameter(p,'Tail','two-sided');
addParameter(p,'NumPermutations',999); addParameter(p,'Seed',1);
addParameter(p,'Exact',false); addParameter(p,'Alpha',.05);
parse(p,varargin{:}); o=p.Results;
tail=lower(char(o.Tail));
if ~ismember(tail,{'positive','negative','two-sided'})
    error('rsa_meeg:inference:Tail','Unknown tail.');
end
if ~isnumeric(o.Alpha) || ~isreal(o.Alpha) || ~isscalar(o.Alpha) || ~isfinite(o.Alpha) || o.Alpha<=0 || o.Alpha>=1
    error('rsa_meeg:inference:Alpha','Alpha must be in (0,1).');
end
if ~isscalar(o.NumPermutations) || ~isnumeric(o.NumPermutations) || ~isreal(o.NumPermutations) || ~isfinite(o.NumPermutations) || o.NumPermutations<1 || o.NumPermutations~=floor(o.NumPermutations)
    error('rsa_meeg:inference:Permutations','NumPermutations must be positive integer.');
end
if ~isscalar(o.Seed) || ~isnumeric(o.Seed) || ~isreal(o.Seed) || ~isfinite(o.Seed) || o.Seed<0 || o.Seed~=floor(o.Seed)
    error('rsa_meeg:inference:Seed','Seed must be nonnegative integer.');
end
if ~isscalar(o.Exact) || (~islogical(o.Exact) && ~ismember(o.Exact,[0,1]))
    error('rsa_meeg:inference:Exact','Exact must be scalar logical.');
end
[m,ids]=rsa_meeg.mapping.prepare_model(R,model);
if ndims(m)>2, error('rsa_meeg:inference:ModelShape','Condition permutation currently requires static pairs x models.'); end
n=numel(ids);
if n<3, error('rsa_meeg:inference:Conditions','At least three conditions are needed for RDM correlation.'); end
if o.Exact && n>7, error('rsa_meeg:inference:ExactSize','Exact condition permutations limited to seven conditions.'); end
observed=rsa_meeg.mapping.compare(R,m,'Correlation',o.Correlation).values;
obs=observed(:)'; valid=isfinite(obs);
if ~any(valid), error('rsa_meeg:inference:Undefined','All observed correlations are undefined.'); end
if o.Exact
    permutations=perms(1:n); B=size(permutations,1);
else
    B=o.NumPermutations;
    prior=rng; cleanup=onCleanup(@()rng(prior));
    rng(double(o.Seed),'twister');
    permutations=zeros(B,n);
    for b=1:B, permutations(b,:)=randperm(n); end
end
pairs=R.pairs; lookup=zeros(n,n); np=size(pairs,1);
for k=1:np, lookup(pairs(k,1),pairs(k,2))=k; lookup(pairs(k,2),pairs(k,1))=k; end
nc=size(R.values,2); nt=size(R.values,3); nm=size(m,2);
nullMax=zeros(B,1);
for b=1:B
    perm=permutations(b,:); pairIndex=zeros(np,1);
    for k=1:np, pairIndex(k)=lookup(perm(pairs(k,1)),perm(pairs(k,2))); end
    shifted=m(pairIndex,:);
    scores=zeros(nc,nt,nm);
    for j=1:nm
        for t=1:nt
            for c=1:nc
                r=rsa_meeg.mapping.correlation(R.values(:,c,t),shifted(:,j),o.Correlation);
                if ~isfinite(r) && valid(sub2ind([nc,nt,nm],c,t,j))
                    error('rsa_meeg:inference:UndefinedNull','A permutation made a defined observed test undefined.');
                end
                scores(c,t,j)=localScore(r,tail);
            end
        end
    end
    nullMax(b)=max(scores(valid));
end
score=arrayfun(@(z)localScore(z,tail),obs); pvals=nan(size(obs));
for k=find(valid)
    if o.Exact, pvals(k)=sum(nullMax>=score(k))/B;
    else, pvals(k)=(1+sum(nullMax>=score(k)))/(B+1); end
    if score(k)==0, pvals(k)=1; end
end
out.statistic=observed; out.p_fwer=reshape(pvals,size(observed));
out.significant=out.p_fwer<=o.Alpha; out.null_max=nullMax;
out.method='condition-label-permutation'; out.tail=tail; out.correlation=char(o.Correlation);
out.seed=o.Seed; out.n_permutations=B; out.family_size=nnz(valid);
out.exact=logical(o.Exact); out.alpha=o.Alpha; out.condition_ids=ids;
out.center_ids=R.center_ids; out.windows=R.windows;
out.assumptions=['Fixed-effects conditional inference: condition labels are exchangeable under the null; ' ...
    'one relabeling applies to every center, time window, and model in this family. ' ...
    'RDM edges are dependent and are not permuted separately. No population-subject inference.'];
end

function s=localScore(r,tail)
switch tail
    case 'positive', s=max(r,0);
    case 'negative', s=max(-r,0);
    otherwise, s=abs(r);
end
end
