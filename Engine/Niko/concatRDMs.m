function RDMs=concatRDMs(varargin)
% concatenates similarity matrices

for RDMI=1:nargin
    
    if ~isstruct(varargin{RDMI})
        varargin{RDMI}=wrapRDMs(varargin{RDMI});
    end
    
    RDMs(RDMI)=varargin{RDMI};
end