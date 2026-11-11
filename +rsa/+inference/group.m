function varargout = group(varargin)
% GROUP Forwarding alias in +rsa/+inference to rsa_meeg.inference.group.
%   Part of RSA-MEEG Toolbox v0.2.0-v2.
[varargout{1:nargout}] = rsa_meeg.inference.group(varargin{:});
end
