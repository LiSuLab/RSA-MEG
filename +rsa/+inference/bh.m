function varargout = bh(varargin)
% BH Forwarding alias in +rsa/+inference to rsa_meeg.inference.bh.
%   Part of RSA-MEEG Toolbox v0.2.0-v2.
[varargout{1:nargout}] = rsa_meeg.inference.bh(varargin{:});
end
