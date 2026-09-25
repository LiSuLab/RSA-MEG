function result=compare_legacy(nili2014_root,output_json)
%COMPARE_LEGACY Optional Nili 2014 Kendall tau-a mathematical regression.
% Usage:
% result=compare_legacy('/path/to/Nili2014-SoftwareS1', '/tmp/result.json')
% Only the original Engines/rankCorr_Kendall_taua.m is put on the path.
% This function does not call the historical startup, +par, or any I/O jobs.
if nargin<1 || ~(ischar(nili2014_root)||isstring(nili2014_root)) || ~isscalar(string(nili2014_root))
    error('rsa_meeg:regression:LegacyRoot','Supply the Nili2014-SoftwareS1 root directory.');
end
if nargin<2, output_json=''; end
engine=fullfile(char(nili2014_root),'rsatoolbox','Engines');
if ~isfile(fullfile(engine,'rankCorr_Kendall_taua.m'))
    error('rsa_meeg:regression:LegacyRoot','Original Engines/rankCorr_Kendall_taua.m was not found.');
end
oldPath=path; pathCleanup=onCleanup(@()path(oldPath)); %#ok<NASGU>
addpath(engine);
oldRng=rng; rngCleanup=onCleanup(@()rng(oldRng)); %#ok<NASGU>
rng(2014,'twister');

xs={[1 2 2 4], [1 NaN 2 2], [3 3 3 3], [1 2 3], [NaN 1]};
ys={[4 2 2 1], [1 2 2 NaN], [1 2 3 4], [1;2;3], [3 NaN]};
names={'tied','paired-nans','constant','row-vs-column','one-complete'};
for k=1:32
    a=double(randi(5,1,20)); b=double(randi(5,20,1));
    if mod(k,3)==0, a(3)=NaN; b(9)=NaN; end
    xs{end+1}=a; ys{end+1}=b; %#ok<AGROW>
    names{end+1}=sprintf('seeded-%02d',k); %#ok<AGROW>
end
errors=nan(1,numel(xs)); pass=false(1,numel(xs)); failures={};
for k=1:numel(xs)
    reference=rankCorr_Kendall_taua(xs{k},ys{k});
    try
        actual=rsa_meeg.mapping.correlation(xs{k},ys{k},'kendall-a');
        if isnan(reference) && isnan(actual)
            pass(k)=true; errors(k)=0;
        elseif isfinite(reference) && isfinite(actual)
            errors(k)=abs(actual-reference); pass(k)=errors(k)<=1e-12;
        end
        if ~pass(k), failures{end+1}=names{k}; end %#ok<AGROW>
    catch err
        failures{end+1}=sprintf('%s: %s',names{k},err.message); %#ok<AGROW>
    end
end

% This is an independent base-MATLAB formula check, not historical-code
% equivalence: correlation distance of condition patterns is 1-corrcoef.
P=[1 2 3;3 2 1;1 3 2;2 1 3];
ds=rsa_meeg.dataset(P,0,1:size(P,1));
R=rsa_meeg.rdm.searchlight(ds,{1:size(P,2)},'Metric','correlation');
C=corrcoef(P.');
expected=zeros(size(R.pairs,1),1);
for k=1:numel(expected), expected(k)=1-C(R.pairs(k,1),R.pairs(k,2)); end
rdm_error=max(abs(R.values(:,1,1)-expected));

result=struct(); result.reference='Nili2014-SoftwareS1/rsatoolbox/Engines/rankCorr_Kendall_taua.m';
result.legacy_root=char(nili2014_root);
result.matlab_version=version;
result.taua_seed=2014; result.taua_cases=numel(xs);
result.taua_passed=sum(pass); result.taua_max_abs_error=max(errors(isfinite(errors)));
result.taua_failures=failures;
result.independent_correlation_rdm_max_abs_error=rdm_error;
result.independent_correlation_rdm_reference='base MATLAB corrcoef; not a historical-code comparison';
result.passed=all(pass) && rdm_error<=1e-12;
if ~isempty(output_json)
    fid=fopen(char(output_json),'w');
    if fid<0, error('rsa_meeg:regression:Output','Could not open JSON output.'); end
    fileCleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true));
end
if ~result.passed
    error('rsa_meeg:regression:LegacyMismatch','Kendall tau-a or independent correlation RDM regression failed.');
end
end
