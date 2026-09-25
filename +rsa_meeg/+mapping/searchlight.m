function out=searchlight(ds,neighborhoods,model,varargin)
%SEARCHLIGHT Batch RDM computation and static model mapping.
% Keeps only one center batch of RDMs in memory. Equivalent to calling
% rsa_meeg.rdm.searchlight on all centers followed by mapping.compare.
p=inputParser;
addParameter(p,'BatchSize',1); addParameter(p,'Correlation','spearman');
addParameter(p,'WindowSamples',1); addParameter(p,'StepSamples',1);
addParameter(p,'Metric','correlation'); addParameter(p,'Mode','spatiotemporal');
addParameter(p,'FeatureTypes',[]); addParameter(p,'Precision',[]);
addParameter(p,'CenterIds',[]);
parse(p,varargin{:}); o=p.Results;
if ~iscell(neighborhoods) || isempty(neighborhoods)
    error('rsa_meeg:mapping:Neighborhoods','Neighborhoods must be nonempty cell array.');
end
nc=numel(neighborhoods);
if ~isnumeric(o.BatchSize) || ~isreal(o.BatchSize) || ~isscalar(o.BatchSize) || ~isfinite(o.BatchSize) || o.BatchSize<1 || o.BatchSize~=floor(o.BatchSize)
    error('rsa_meeg:mapping:BatchSize','BatchSize must be positive integer.');
end
ids=o.CenterIds;
if isempty(ids), ids=1:nc; end
if numel(ids)~=nc
    error('rsa_meeg:mapping:CenterIds','CenterIds must match neighborhoods.');
end
if iscell(o.Precision) && numel(o.Precision)~=nc
    error('rsa_meeg:mapping:Precision','Cell Precision must have one matrix per center.');
end
out=[];
for first=1:o.BatchSize:nc
    ix=first:min(nc,first+o.BatchSize-1);
    P=o.Precision;
    if iscell(P), P=P(ix); end
    R=rsa_meeg.rdm.searchlight(ds,neighborhoods(ix),...
        'WindowSamples',o.WindowSamples,'StepSamples',o.StepSamples,...
        'Metric',o.Metric,'Mode',o.Mode,'CenterIds',ids(ix),...
        'FeatureTypes',o.FeatureTypes,'Precision',P);
    mapped=rsa_meeg.mapping.compare(R,model,'Correlation',o.Correlation);
    if isempty(out)
        out.values=nan(nc,size(mapped.values,2),size(mapped.values,3));
        out.center_ids=ids; out.windows=R.windows; out.condition_ids=R.condition_ids;
        out.correlation=mapped.correlation; out.metric=R.metric; out.mode=R.mode;
        out.pair_counts=R.pair_counts; out.batch_size=o.BatchSize;
        out.assumption=['Static searchlight RDMs are mapped in center batches; RDM edges are dependent. ' ...
            'No analytic edgewise p-values.'];
    end
    out.values(ix,:,:)=mapped.values;
end
end
