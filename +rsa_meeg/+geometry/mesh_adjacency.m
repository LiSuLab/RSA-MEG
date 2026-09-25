function A=mesh_adjacency(triangles,n_vertices)
%MESH_ADJACENCY Sparse undirected adjacency from local 1-based triangular faces.
% Replaces fixed-size LiSuLab +rsa/+meg/calculateMeshAdjacency.m (c0dfd32f809e).
if ~isscalar(n_vertices) || ~isnumeric(n_vertices) || ~isreal(n_vertices) || n_vertices<1 || n_vertices~=floor(n_vertices)
    error('rsa_meeg:geometry:Vertices','n_vertices must be a positive integer.');
end
if ~isnumeric(triangles) || ~isreal(triangles) || size(triangles,2)~=3 || any(~isfinite(triangles(:))) || any(triangles(:)<1 | triangles(:)>n_vertices | triangles(:)~=floor(triangles(:)))
    error('rsa_meeg:geometry:Faces','Triangles must be M x 3 local vertex indices.');
end
if any(triangles(:,1)==triangles(:,2) | triangles(:,1)==triangles(:,3) | triangles(:,2)==triangles(:,3))
    error('rsa_meeg:geometry:Faces','Degenerate triangular face.');
end
ii=[triangles(:,1);triangles(:,2);triangles(:,3)];
jj=[triangles(:,2);triangles(:,3);triangles(:,1)];
A=sparse([ii;jj],[jj;ii],true,n_vertices,n_vertices)>0;
end
