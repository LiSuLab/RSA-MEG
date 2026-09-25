function out=bh(p,alpha)
%BH Benjamini-Hochberg FDR within the supplied p-value family.
if nargin<2, alpha=.05; end
if ~isnumeric(p) || ~isreal(p) || any(~isfinite(p(:))) || any(p(:)<0 | p(:)>1)
    error('rsa_meeg:inference:BHInput','p must contain finite values in [0,1].');
end
if ~isscalar(alpha) || ~isfinite(alpha) || alpha<=0 || alpha>=1
    error('rsa_meeg:inference:BHAlpha','alpha must lie in (0,1).');
end
sz=size(p); m=numel(p);
if m==0, out=struct('reject',false(sz),'p_adjusted',zeros(sz),'threshold',[],'alpha',alpha); return; end
[sp,ord]=sort(p(:)); q=sp.*m./(1:m)'; q=min(1,flipud(cummin(flipud(q))));
adj=zeros(m,1); adj(ord)=q;
k=find(sp<=(1:m)'*alpha/m,1,'last');
if isempty(k), reject=false(sz); threshold=[];
else, threshold=sp(k); reject=reshape(p(:)<=threshold,sz); end
out=struct('reject',reject,'p_adjusted',reshape(adj,sz),'threshold',threshold,'alpha',alpha);
end
