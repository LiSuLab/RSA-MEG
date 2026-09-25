function r = correlation(x,y,method)
%CORRELATION Pairwise-complete descriptive RDM similarity; no edgewise p.
if ~isnumeric(x) || ~isnumeric(y) || ~isreal(x) || ~isreal(y) || ...
        ~isvector(x) || ~isvector(y) || numel(x)~=numel(y)
    error('rsa_meeg:mapping:CorrelationInput','Inputs must be equal-length real numeric vectors.');
end
x=x(:); y=y(:);
good = isfinite(x) & isfinite(y);
x=x(good); y=y(good);
if numel(x)<2
    r=NaN; return
end
switch lower(char(method))
    case 'pearson'
        x=x-mean(x); y=y-mean(y);
        den=sqrt(sum(x.^2)*sum(y.^2));
        if den==0, r=NaN; else, r=sum(x.*y)/den; end
    case 'spearman'
        r=rsa_meeg.mapping.correlation(localRanks(x),localRanks(y),'pearson');
    case {'kendall-a','kendall_taua'}
        c=0;
        for i=1:numel(x)-1
            c=c+sum(sign(x(i)-x(i+1:end)).*sign(y(i)-y(i+1:end)));
        end
        r=c/(numel(x)*(numel(x)-1)/2); % ties contribute zero, no tie correction
    otherwise
        error('rsa_meeg:mapping:Correlation','Unknown correlation type.');
end
end

function z=localRanks(x)
[s,ord]=sort(x(:));
z=zeros(size(x(:))); i=1;
while i<=numel(s)
    j=i;
    while j<numel(s) && s(j+1)==s(i), j=j+1; end
    z(ord(i:j))=(i+j)/2; i=j+1;
end
z=reshape(z,size(x));
end
