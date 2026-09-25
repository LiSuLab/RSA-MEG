function surface = read_surface(filename)
%READ_SURFACE Read a FreeSurfer triangular binary surface (coordinates mm).
% Faces are converted from file zero-based indices to MATLAB 1-based indices.
% Old quad surfaces are deliberately rejected, not reinterpreted as triangles.
fid=fopen(filename,'rb','ieee-be');
if fid<0,error('rsa_meeg:io:Open','Cannot open surface: %s',filename);end
cleanup=onCleanup(@()fclose(fid));
b=fread(fid,3,'uint8=>double');
if numel(b)~=3||dot(b,[65536;256;1])~=16777214
    error('rsa_meeg:io:Surface','Only triangular FreeSurfer surfaces are supported.');
end
fgetl(fid); fgetl(fid);
n=fread(fid,2,'int32=>double');
fseek(fid,0,'cof'); pos=ftell(fid); fseek(fid,0,'eof'); bytes=ftell(fid); fseek(fid,pos,'bof');
if numel(n)~=2||any(n<1)||pos+12*sum(n)>bytes
    error('rsa_meeg:io:Surface','Invalid/truncated surface counts.');
end
xyz=fread(fid,[3,n(1)],'float32=>double')';
tri=fread(fid,[3,n(2)],'int32=>double')'+1;
if any(~isfinite(xyz(:)))||any(tri(:)<1|tri(:)>n(1))
    error('rsa_meeg:io:Surface','Invalid surface coordinates or faces.');
end
surface=struct('positions_mm',xyz,'triangles',tri,'filename',char(filename));
end
