function out = compare(R, model, varargin)
%COMPARE Static model-to-neural RDM similarity at each center/window.
p=inputParser;
addParameter(p,'Correlation','spearman');
parse(p,varargin{:});
[m,ids]=rsa_meeg.mapping.prepare_model(R,model);
if ndims(m)>2 || size(m,1)~=size(R.values,1)
    error('rsa_meeg:mapping:StaticShape','Static model must be pairs x models.');
end
nc=size(R.values,2); nt=size(R.values,3); nm=size(m,2);
v=nan(nc,nt,nm);
for j=1:nm
    for t=1:nt
        for c=1:nc
            v(c,t,j)=rsa_meeg.mapping.correlation(R.values(:,c,t),m(:,j),p.Results.Correlation);
        end
    end
end
out.values=v; out.center_ids=R.center_ids; out.windows=R.windows;
out.condition_ids=ids; out.correlation=char(p.Results.Correlation);
out.assumption='RDM edges are dependent; values are descriptive, with no analytic edgewise p-value.';
end
