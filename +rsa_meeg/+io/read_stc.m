function stc = read_stc(filename)
%READ_STC Read one MNE surface STC file without an MNE dependency.
% Vertex identifiers are ZERO-BASED file identifiers, never MATLAB indices.
% Times are seconds; data is vertex x time. Hemisphere is not stored in STC.
% Format reference: mne-tools/mne-python mne/source_estimate.py _read_stc.
% Replaces the unavailable mne_read_stc_file1 in historical RSA-MEG recipes.
fid = fopen(filename, 'rb', 'ieee-be');
if fid < 0, error('rsa_meeg:io:Open', 'Cannot open STC: %s', filename); end
cleanup = onCleanup(@() fclose(fid));
fseek(fid,0,'eof'); bytes=ftell(fid); fseek(fid,0,'bof');
if bytes < 20, error('rsa_meeg:io:STC', 'Truncated STC header.'); end
tmin = fread(fid,1,'float32=>double')/1000;
tstep = fread(fid,1,'float32=>double')/1000;
nv = fread(fid,1,'uint32=>double');
if nv < 1 || 16+4*nv > bytes
    error('rsa_meeg:io:STC', 'Invalid vertex count.');
end
vertices = fread(fid,nv,'uint32=>double');
nt = fread(fid,1,'uint32=>double');
if isempty(nt) || nt < 1 || 16+4*nv+4*nv*nt ~= bytes
    error('rsa_meeg:io:STC', 'STC size disagrees with header.');
end
if ~isfinite(tmin) || ~isfinite(tstep) || tstep <= 0 || numel(unique(vertices)) ~= nv
    error('rsa_meeg:io:STC', 'Invalid STC time or duplicate vertex identifier.');
end
data = fread(fid,[nv,nt],'float32=>double');
if any(~isfinite(data(:))), error('rsa_meeg:io:STC', 'Nonfinite STC data.'); end
stc = struct('vertices_zero_based',vertices,'data',data, ...
    'sample_times_s',tmin+(0:nt-1)*tstep,'tstep_s',tstep, ...
    'filename',char(filename));
end
