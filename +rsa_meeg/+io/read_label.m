function label = read_label(filename)
%READ_LABEL Read FreeSurfer ASCII label; retain zero-based vertex IDs.
fid=fopen(filename,'rt');
if fid<0,error('rsa_meeg:io:Open','Cannot open label: %s',filename);end
cleanup=onCleanup(@()fclose(fid));
header=fgetl(fid); n=str2double(fgetl(fid));
if ~ischar(header)||~startsWith(header,'#')||~isfinite(n)||n<0||n~=fix(n)
    error('rsa_meeg:io:Label','Invalid FreeSurfer ASCII label header.');
end
a=fscanf(fid,'%f',[5,Inf])';
if size(a,1)~=n||any(~isfinite(a(:)))||any(a(:,1)<0|a(:,1)~=fix(a(:,1)))||numel(unique(a(:,1)))~=n
    error('rsa_meeg:io:Label','Invalid label rows or duplicate vertex IDs.');
end
label=struct('vertices_zero_based',a(:,1),'positions_mm',a(:,2:4),'values',a(:,5));
end
