function results = run_tests
%RUN_TESTS Execute the release numerical and integration suite.
root=fileparts(mfilename('fullpath'));old=path;cleanup=onCleanup(@()path(old));
addpath(root);addpath(fullfile(root,'examples'));
results=runtests(fullfile(root,'tests'));
disp(table(results));
assert(all([results.Passed]),'rsa_meeg:TestsFailed','One or more tests failed or were incomplete.');
end
