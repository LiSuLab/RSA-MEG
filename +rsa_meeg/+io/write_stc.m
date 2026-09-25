function write_stc(filename, vertices_zero_based, data, sample_times_s, varargin)
%WRITE_STC Write one hemisphere, preserving zero-based vertex identifiers.
% Explicit Overwrite=true is required to replace an existing file. The STC
% format stores float32 samples and a uniform time axis in milliseconds.
p=inputParser; addParameter(p,'Overwrite',false,@(x)islogical(x)&&isscalar(x));
addParameter(p,'TimeStep',[],@(x)isnumeric(x)&&isscalar(x)&&isfinite(x)&&x>0);
parse(p,varargin{:});
v=double(vertices_zero_based(:)); t=double(sample_times_s(:)');
if isempty(v)||any(~isfinite(v)|v<0|v~=fix(v)|v>2^32-1)||numel(unique(v))~=numel(v)
    error('rsa_meeg:io:Vertices','Unique zero-based uint32 vertex IDs required.');
end
if ~isnumeric(data)||~isreal(data)||ndims(data)>2||size(data,1)~=numel(v)||size(data,2)~=numel(t)||isempty(t)||any(~isfinite(data(:)))||any(~isfinite(t))
    error('rsa_meeg:io:Data','Data must be finite vertex x time, matching times.');
end
if numel(t)==1
    dt=p.Results.TimeStep;
    if isempty(dt), error('rsa_meeg:io:Time','TimeStep required for a single time sample.'); end
else
    dt=t(2)-t(1);
    if dt<=0 || any(abs(diff(t)-dt)>max(1e-12,1e-6*dt))
        error('rsa_meeg:io:Time','STC requires uniformly increasing times.');
    end
end
if any(~isfinite(single(data(:)))) || ~isfinite(single(t(1)*1000)) || ~isfinite(single(dt*1000))
    error('rsa_meeg:io:Range','Values exceed the float32 range of STC.');
end
if isfile(filename)&&~p.Results.Overwrite, error('rsa_meeg:io:Exists','File exists: %s',filename); end
fid=fopen(filename,'wb','ieee-be');
if fid<0, error('rsa_meeg:io:Open','Cannot write STC: %s',filename); end
cleanup=onCleanup(@()fclose(fid));
counts=[fwrite(fid,t(1)*1000,'float32'),fwrite(fid,dt*1000,'float32'), ...
    fwrite(fid,numel(v),'uint32'),fwrite(fid,v,'uint32'), ...
    fwrite(fid,numel(t),'uint32'),fwrite(fid,data,'float32')];
if sum(counts) ~= 4+numel(v)+numel(data), error('rsa_meeg:io:Write','Incomplete STC write.'); end
end
