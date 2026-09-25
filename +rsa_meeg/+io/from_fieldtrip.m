function ds = from_fieldtrip(ft, condition_ids, varargin)
%FROM_FIELDTRIP Import FieldTrip raw trials or timelock rpt_chan_time data.
% Trials must already share a time axis; no implicit resampling/truncation.
% Channel types/positions/units are explicit Dataset arguments. This avoids
% silently treating gradiometers, magnetometers and EEG as one scale.
if ~isstruct(ft)||~isfield(ft,'label')||~isfield(ft,'time')
    error('rsa_meeg:io:FieldTrip','Expected FieldTrip label and time fields.');
end
f=numel(ft.label);
if isfield(ft,'trial')&&iscell(ft.trial)
    if isempty(ft.trial)||~iscell(ft.time)||numel(ft.time)~=numel(ft.trial)
        error('rsa_meeg:io:FieldTrip','Each raw trial needs a time vector.');
    end
    t=double(ft.time{1}(:)'); data=zeros(numel(ft.trial),f,numel(t));
    for k=1:numel(ft.trial)
        tk=double(ft.time{k}(:)');
        if ~isequal(size(ft.trial{k}),[f,numel(t)])||numel(tk)~=numel(t)||any(abs(tk-t)>1e-10)
            error('rsa_meeg:io:Alignment','FieldTrip trials must have matching channels/times.');
        end
        data(k,:,:)=ft.trial{k};
    end
elseif isfield(ft,'trial')&&isnumeric(ft.trial)&&isfield(ft,'dimord')&&strcmp(ft.dimord,'rpt_chan_time')
    data=ft.trial;t=ft.time;
elseif isfield(ft,'avg')&&isfield(ft,'dimord')&&strcmp(ft.dimord,'chan_time')
    data=reshape(ft.avg,1,size(ft.avg,1),size(ft.avg,2));t=ft.time;
else
    error('rsa_meeg:io:FieldTrip','Supported formats: raw trial cells, rpt_chan_time, chan_time avg.');
end
args=varargin;
if ~any(strcmpi(args(1:2:end),'FeatureIds')),args=[{'FeatureIds',ft.label},args];end
ds=rsa_meeg.dataset(data,t,condition_ids,args{:});
ds.provenance=struct('format','FieldTrip in-memory struct','preprocessing','none in adapter');
end
