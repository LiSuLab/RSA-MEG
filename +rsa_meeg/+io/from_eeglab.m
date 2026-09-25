function ds = from_eeglab(EEG, condition_ids, varargin)
%FROM_EEGLAB Import an already epoched, in-memory EEGLAB EEG struct.
% Does not preprocess, rereference or load external .fdt files. Pass explicit
% FoldIds, Reference and Positions/PositionUnit as appropriate. EEGLAB times
% are milliseconds. Electrode coordinates have no assumed unit or frame.
if ~isstruct(EEG)||~isfield(EEG,'data')||~isnumeric(EEG.data)||~isfield(EEG,'times')
    error('rsa_meeg:io:EEGLAB','Expected in-memory EEG.data and EEG.times.');
end
if ndims(EEG.data)<3 && (~isfield(EEG,'epoch')||numel(EEG.epoch)~=1)
    error('rsa_meeg:io:EEGLAB','Continuous 2-D data are not epoched; a single epoch requires EEG.epoch metadata.');
end
data=permute(EEG.data,[3,1,2]); f=size(data,2); ids=arrayfun(@(k)sprintf('EEG%d',k),1:f,'UniformOutput',false);
if isfield(EEG,'chanlocs')&&numel(EEG.chanlocs)==f&&isfield(EEG.chanlocs,'labels')
    ids={EEG.chanlocs.labels};
end
types=repmat({'eeg'},1,f);
if isfield(EEG,'chanlocs')&&numel(EEG.chanlocs)==f&&isfield(EEG.chanlocs,'type')
    for k=1:f
        value=EEG.chanlocs(k).type;
        if ~isempty(value),types{k}=lower(char(string(value)));end
    end
end
% Explicit arguments override defaults, for example data with EOG channels.
defaults=struct('FeatureIds',{ids},'FeatureTypes',{types});
args=merge_args(defaults,varargin);
ds=rsa_meeg.dataset(data,double(EEG.times(:)')/1000,condition_ids,args{:});
ds.provenance=struct('format','EEGLAB epoched struct','preprocessing','none in adapter');
end
function args=merge_args(s,args)
names=fieldnames(s); supplied=lower(string(args(1:2:end)));
for k=1:numel(names)
    if ~any(supplied==lower(string(names{k}))),args=[{names{k},s.(names{k})},args];end %#ok<AGROW>
end
end
