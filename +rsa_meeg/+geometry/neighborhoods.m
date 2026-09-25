function neighbors = neighborhoods(positions_mm,radius_mm,varargin)
%NEIGHBORHOODS Radius neighborhoods using local feature indices.
% Generalizes LiSuLab +rsa/+meg/getNearbySensors.m at c0dfd32f809e.
p=inputParser; addParameter(p,'Hemisphere',[]); addParameter(p,'Centers',[]); parse(p,varargin{:});
if ~isnumeric(positions_mm) || ~isreal(positions_mm) || size(positions_mm,2)~=3 || isempty(positions_mm) || any(~isfinite(positions_mm(:)))
    error('rsa_meeg:geometry:Positions','Positions must be finite N x 3 mm coordinates.');
end
if ~isscalar(radius_mm) || ~isnumeric(radius_mm) || ~isreal(radius_mm) || ~isfinite(radius_mm) || radius_mm<0
    error('rsa_meeg:geometry:Radius','Radius must be nonnegative finite mm.');
end
n=size(positions_mm,1); centers=p.Results.Centers;
if isempty(centers), centers=1:n; end
if ~isnumeric(centers) || any(centers<1 | centers>n | centers~=floor(centers)) || numel(unique(centers))~=numel(centers)
    error('rsa_meeg:geometry:Centers','Centers must be unique local feature indices.');
end
hemi=p.Results.Hemisphere;
if ~isempty(hemi) && (~(isstring(hemi)||iscellstr(hemi)) || numel(hemi)~=n)
    error('rsa_meeg:geometry:Hemisphere','Hemisphere needs one label per feature.');
end
hemi=string(hemi); neighbors=cell(1,numel(centers));
for c=1:numel(centers)
    i=centers(c); d2=sum((positions_mm-positions_mm(i,:)).^2,2);
    keep=d2<=double(radius_mm)^2+eps(double(radius_mm)^2);
    if ~isempty(hemi), keep=keep & (hemi(:)==hemi(i)); end
    neighbors{c}=find(keep).';
end
end
