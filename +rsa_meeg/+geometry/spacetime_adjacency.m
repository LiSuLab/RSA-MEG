function A=spacetime_adjacency(spatialAdj,n_time)
%SPACETIME_ADJACENCY Adjacency for linear index space+(time-1)*n_space.
if (~isnumeric(spatialAdj) && ~islogical(spatialAdj)) || ~isreal(spatialAdj)
    error('rsa_meeg:geometry:Adjacency','Spatial adjacency must be numeric/logical.');
end
[n,m]=size(spatialAdj);
if n~=m || n<1 || any(~isfinite(spatialAdj(:))) || any(spatialAdj(:)<0) || any(diag(spatialAdj)~=0) || ~isequal(logical(spatialAdj),logical(spatialAdj'))
    error('rsa_meeg:geometry:Adjacency','Spatial adjacency must be symmetric square with zero diagonal.');
end
if ~isscalar(n_time) || ~isnumeric(n_time) || ~isreal(n_time) || n_time<1 || n_time~=floor(n_time)
    error('rsa_meeg:geometry:Time','n_time must be a positive integer.');
end
A=kron(speye(n_time),sparse(logical(spatialAdj)));
if n_time>1
    B=sparse(1:n_time-1,2:n_time,1,n_time,n_time);
    A=A+kron(B+B',speye(n));
end
A=logical(A);
end
