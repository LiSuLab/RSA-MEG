function maps=vecs2maps(vecs,mask)

% USAGE
%       maps=vecs2maps(vecs,mask)
%
% FUNCTION
%       to convert the set of space column vectors vecs into a set of
%       spatial maps of dimensions of array mask. mask is a logical array
%       whose true values must correspond to the spatial (i.e. first)
%       dimension of vecs in number and order.
%
%       the returned maps are zero, wherever the mask is false.
%
%       mask must currently be 3D.
%
%       maps has ndims(mask)+1 dimensions, where the last chooses the
%       space vector.
%
% GENERAL IDEA: This is basically the same as vec2map.m, except that it
%         takes an N x M matrix (columns are vectors) and gives out a 4
%         dimensional array "maps".  "maps" is a number of 3-dimensional
%         statistical maps (like the ones generated in vec2map.m) stacked
%         along the 4th dimension.

mask=logical(mask);
nSpaceVecs=size(vecs,2);
maps=zeros([size(mask) nSpaceVecs]);
cMap=zeros(size(mask));

for spaceVecI=1:nSpaceVecs
    cMap(mask)=vecs(:,spaceVecI);
    maps(:,:,:,spaceVecI)=cMap;
end