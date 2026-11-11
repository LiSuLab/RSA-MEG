function varargout = map(varargin)
% MAP Forwarding alias in +rsa/+plot to rsa_meeg.plot.map.
%   Part of RSA-MEEG Toolbox v0.2.0-v2.
[varargout{1:nargout}] = rsa_meeg.plot.map(varargin{:});
end
