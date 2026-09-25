function setup_rsa_meeg
%SETUP_RSA_MEEG Add only this release root and examples to the MATLAB path.
root=fileparts(mfilename('fullpath'));addpath(root);addpath(fullfile(root,'examples'));
fprintf('RSA-MEEG 0.1.0-dev: %s\n',root);
end
