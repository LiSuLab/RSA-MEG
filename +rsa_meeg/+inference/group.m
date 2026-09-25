function out = group(values,varargin)
%GROUP Subject-level sign-flip maximum-statistic familywise inference.
% values: subjects x tests. The SAME subject sign is applied to all tests.
p=inputParser;
addParameter(p,'Method','max-t'); addParameter(p,'Tail','two-sided');
addParameter(p,'Adjacency',[]); addParameter(p,'ClusterThreshold',[]);
addParameter(p,'NumPermutations',999); addParameter(p,'Seed',1);
addParameter(p,'Exact',false); addParameter(p,'Alpha',0.05);
parse(p,varargin{:}); o=p.Results;
if ~isnumeric(values) || ~isreal(values) || ~ismatrix(values) || any(~isfinite(values(:)))
    error('rsa_meeg:inference:Values','Values must be a finite real subjects x tests matrix.');
end
[ns,nv]=size(values);
if ns<2 || nv<1, error('rsa_meeg:inference:Size','At least two subjects and one test are required.'); end
method=lower(char(o.Method)); tail=lower(char(o.Tail));
if ~ismember(method,{'max-t','cluster-mass','tfce'}) || ~ismember(tail,{'positive','negative','two-sided'})
    error('rsa_meeg:inference:Options','Unknown method or tail.');
end
if ~isnumeric(o.Alpha) || ~isreal(o.Alpha) || ~isscalar(o.Alpha) || ~isfinite(o.Alpha) || o.Alpha<=0 || o.Alpha>=1
    error('rsa_meeg:inference:Alpha','Alpha must be between zero and one.');
end
adj=[];
if ~strcmp(method,'max-t')
    adj=o.Adjacency;
    if ~isnumeric(adj) && ~islogical(adj), error('rsa_meeg:inference:Adjacency','Adjacency must be a symmetric tests x tests matrix.'); end
    if ~isreal(adj) || ~isequal(size(adj),[nv,nv]) || any(~isfinite(adj(:))) || any(adj(:)<0) || ~isequal(logical(adj),logical(adj')) || any(diag(adj))
        error('rsa_meeg:inference:Adjacency','Adjacency must be finite, symmetric, tests x tests, zero diagonal.');
    end
    adj=logical(adj);
end
if strcmp(method,'cluster-mass') && (~isscalar(o.ClusterThreshold) || ~isnumeric(o.ClusterThreshold) || ~isreal(o.ClusterThreshold) || ~isfinite(o.ClusterThreshold) || o.ClusterThreshold<=0)
    error('rsa_meeg:inference:Threshold','ClusterThreshold must be positive t units.');
end
if ~isnumeric(o.NumPermutations) || ~isreal(o.NumPermutations) || ~isscalar(o.NumPermutations) || ~isfinite(o.NumPermutations) || o.NumPermutations<1 || o.NumPermutations~=floor(o.NumPermutations)
    error('rsa_meeg:inference:Permutations','NumPermutations must be a positive integer.');
end
if ~isscalar(o.Seed) || ~isnumeric(o.Seed) || ~isreal(o.Seed) || ~isfinite(o.Seed) || o.Seed<0 || o.Seed~=floor(o.Seed)
    error('rsa_meeg:inference:Seed','Seed must be a nonnegative integer.');
end
if ~isscalar(o.Exact) || (~islogical(o.Exact) && ~ismember(o.Exact,[0,1]))
    error('rsa_meeg:inference:Exact','Exact must be logical scalar.');
end
if o.Exact && ns>16, error('rsa_meeg:inference:ExactSize','Exact enumeration limited to 16 subjects.'); end
t=localT(values);
[score,labels]=localScore(t,method,tail,adj,o.ClusterThreshold);
if o.Exact
    B=2^ns; flips=ones(B,ns);
    for k=1:ns, flips(:,k)=2*double(bitget(uint32((0:B-1)'),k))-1; end
else
    B=o.NumPermutations;
    prior=rng; cleanup=onCleanup(@()rng(prior));
    rng(double(o.Seed),'twister');
    flips=2*(rand(B,ns)>.5)-1;
end
nullMax=zeros(B,1);
for b=1:B
    tb=localT(values.*flips(b,:)');
    sb=localScore(tb,method,tail,adj,o.ClusterThreshold);
    nullMax(b)=max(sb);
end
if o.Exact
    pvals=arrayfun(@(z)sum(nullMax>=z)/B,score);
else
    pvals=arrayfun(@(z)(1+sum(nullMax>=z))/(B+1),score);
end
pvals(score==0)=1;
out=struct('statistic',t,'enhanced_statistic',score,'p_fwer',pvals,'significant',pvals<=o.Alpha,...
    'null_max',nullMax,'cluster_labels',labels,'method',method,'tail',tail,'seed',o.Seed,...
    'n_permutations',B,'family_size',nv,'exact',logical(o.Exact),'alpha',o.Alpha,...
    'assumptions',['Subjects are independent, under the null their whole-map signs are exchangeable; ' ...
    'one sign per subject applies across the entire tests family. Cluster p applies to clusters, ' ...
    'not individual localization. No single-subject random-effects inference.']);
end

function t=localT(x)
n=size(x,1);
% Column scaling avoids overflow in std for finite values near realmax and
% underflow for tiny units, while preserving the t statistic itself.
scale=max(abs(x),[],1); scale(scale==0)=1;
z=x./scale;
mu=mean(z,1); sd=std(z,0,1);
guard=eps(max(abs(z),[],1));
t=mu./max(sd/sqrt(n),guard);
end

function [score,labels]=localScore(t,method,tail,adj,threshold)
labels=zeros(size(t));
switch method
    case 'max-t'
        switch tail
            case 'positive', score=max(t,0);
            case 'negative', score=max(-t,0);
            otherwise, score=abs(t);
        end
    case 'cluster-mass'
        score=zeros(size(t));
        signs=localSigns(tail);
        for s=signs
            h=s*t; [cc,nc]=localClusters(h>threshold,adj);
            for k=1:nc
                ix=cc==k; score(ix)=sum(h(ix)-threshold); labels(ix)=s*k;
            end
        end
    case 'tfce'
        score=zeros(size(t));
        signs=localSigns(tail);
        for s=signs
            h=max(s*t,0); levels=unique([0,h]);
            for j=1:numel(levels)-1
                lo=levels(j); hi=levels(j+1);
                [cc,nc]=localClusters(h>(lo+hi)/2,adj);
                integral=(hi^3-lo^3)/3; % E=0.5, H=2
                for k=1:nc
                    ix=cc==k; score(ix)=score(ix)+sqrt(sum(ix))*integral;
                end
            end
        end
end
end

function signs=localSigns(tail)
switch tail
    case 'positive', signs=1;
    case 'negative', signs=-1;
    otherwise, signs=[1,-1];
end
end

function [labels,nc]=localClusters(mask,adj)
n=numel(mask); labels=zeros(1,n); nc=0;
for i=find(mask)
    if labels(i)~=0, continue; end
    nc=nc+1; labels(i)=nc; queue=zeros(1,n); queue(1)=i; head=1; last=1;
    while head<=last
        node=queue(head); head=head+1;
        neighbors=find(adj(node,:) & mask & labels==0);
        if isempty(neighbors), continue; end
        labels(neighbors)=nc;
        queue(last+1:last+numel(neighbors))=neighbors; last=last+numel(neighbors);
    end
end
end
