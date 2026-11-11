function setup_rsa_meeg
%SETUP_RSA_MEEG Add this release root and examples to the MATLAB path.
root=fileparts(mfilename('fullpath'));addpath(root);addpath(fullfile(root,'examples'));
fprintf('RSA-MEEG 0.2.0-v2: %s\n',root);
end
